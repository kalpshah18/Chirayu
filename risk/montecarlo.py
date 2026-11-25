"""Monte Carlo loss simulation (IMPLEMENTATION PLAN.md §6.8): 500 leases x
20,000 paths, targeting under 10s on a laptop.

- 2-state Markov-switching log rental index (normal / launch-shock regimes),
  with launch jumps, shared across all leases (it's one market).
- Per-lease wear drawn from its archetype's GSU/hour distribution.
- Per-lease failure drawn from the calibrated hazard.
- Lessee default via a one-factor Gaussian copula linked to the market
  regime, so defaults cluster when the market is in a shock regime.

Outputs: EL, VaR99, CVaR99, and the full loss array (for tranching).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MarketParams:
    drift_normal: float
    vol_normal: float
    drift_shock: float
    vol_shock: float
    p_stay_normal: float
    p_stay_shock: float


@dataclass
class LeaseSpec:
    acquisition_cost: float
    gsu_per_hour_mean: float
    gsu_per_hour_std: float
    hours_per_month: float
    term_months: int
    base_rent_monthly: float
    reserve_rate_per_gsu: float
    lam_condition: float
    salvage_frac: float
    hazard_fn: callable  # g (cumulative GSU) -> annualised failure probability
    default_prob_floor: float = 0.01  # lessee's idiosyncratic default probability in a normal market


@dataclass
class McResult:
    losses: np.ndarray          # (n_paths,) total book loss per path, in currency units
    per_lease_losses: np.ndarray  # (n_leases, n_paths)
    expected_loss: float
    var99: float
    cvar99: float


def simulate_market_paths(params: MarketParams, n_months: int, n_paths: int,
                           rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Returns (log_index_paths, regime_paths), each (n_paths, n_months+1)."""
    regime = np.zeros((n_paths, n_months + 1), dtype=int)  # 0 = normal, 1 = shock
    log_idx = np.zeros((n_paths, n_months + 1))

    u = rng.random((n_paths, n_months))
    for m in range(1, n_months + 1):
        prev = regime[:, m - 1]
        stay_prob = np.where(prev == 0, params.p_stay_normal, params.p_stay_shock)
        stay = u[:, m - 1] < stay_prob
        regime[:, m] = np.where(stay, prev, 1 - prev)

        drift_m = np.where(regime[:, m] == 0, params.drift_normal, params.drift_shock) / 12.0
        vol_m = np.where(regime[:, m] == 0, params.vol_normal, params.vol_shock) / np.sqrt(12.0)
        shock = rng.standard_normal(n_paths) * vol_m
        log_idx[:, m] = log_idx[:, m - 1] + drift_m + shock

    return log_idx, regime


def simulate_book(leases: list[LeaseSpec], market: MarketParams, n_paths: int,
                   market_factor_loading: float, seed: int) -> McResult:
    rng = np.random.default_rng(seed)
    n_leases = len(leases)
    max_term = max(l.term_months for l in leases)

    log_idx, regime = simulate_market_paths(market, max_term, n_paths, rng)
    index_level = np.exp(log_idx)

    # One-factor Gaussian copula for lessee default: Z_i = sqrt(rho)*M + sqrt(1-rho)*eps_i.
    # M is linked to the market regime (shock regime => worse common factor draw).
    market_factor = rng.standard_normal(n_paths)
    shock_frac = regime[:, -1].astype(float)  # fraction of path in shock at term, proxy for stress
    market_factor = market_factor - market_factor_loading * shock_frac  # shock pulls M down (worse)

    per_lease_losses = np.zeros((n_leases, n_paths))

    for i, lease in enumerate(leases):
        T = lease.term_months
        terminal_index = index_level[:, T]

        # GSU accumulation: per-month draws from the archetype distribution, summed to term.
        gsu_draws = rng.normal(lease.gsu_per_hour_mean, lease.gsu_per_hour_std, size=(n_paths, T))
        gsu_draws = np.clip(gsu_draws, 0.0, None) * lease.hours_per_month
        g_T = gsu_draws.sum(axis=1)

        p_fail = np.clip(lease.hazard_fn(g_T), 0.0, 1.0)
        fails = rng.random(n_paths) < p_fail

        market_value_T = lease.acquisition_cost * terminal_index  # normalised: index starts at 1
        condition = np.exp(-lease.lam_condition * g_T)
        salvage = lease.salvage_frac * market_value_T
        v_T = np.where(fails, salvage, market_value_T * condition)

        # Lessee default: idiosyncratic factor correlated with the market factor.
        rho = market_factor_loading
        idio = rng.standard_normal(n_paths)
        z = np.sqrt(rho) * market_factor + np.sqrt(max(0.0, 1 - rho)) * idio
        # norminv(p) = sqrt(2) * erfinv(2p - 1); default when the common/idiosyncratic
        # factor draw falls below the threshold implied by the lessee's default probability.
        default_threshold = np.sqrt(2) * _erfinv(2 * lease.default_prob_floor - 1)
        defaults = z < default_threshold

        reserve_collected = lease.reserve_rate_per_gsu * g_T
        expected_v_if_no_default = lease.acquisition_cost  # book value baseline for loss computation
        loss_if_default = np.clip(expected_v_if_no_default - v_T - reserve_collected, 0.0, None)

        per_lease_losses[i] = np.where(defaults | fails, loss_if_default, 0.0)

    total_losses = per_lease_losses.sum(axis=0)
    el = float(np.mean(total_losses))
    var99 = float(np.percentile(total_losses, 99))
    tail = total_losses[total_losses >= var99]
    cvar99 = float(np.mean(tail)) if len(tail) else var99

    return McResult(losses=total_losses, per_lease_losses=per_lease_losses,
                     expected_loss=el, var99=var99, cvar99=cvar99)


def _erfinv(x: np.ndarray) -> np.ndarray:
    from scipy.special import erfinv
    return erfinv(x)
