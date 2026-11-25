"""Lease pricing (IMPLEMENTATION PLAN.md §6.8).

- Market value M(a, s): PV of the GPU's remaining market rental stream at the
  current rental rate (a simple DCF; risk/montecarlo.py uses the full
  regime-switching index for simulation, this is the point-estimate used in
  deterministic pricing).
- Reserve rate: r = -(1/(1+y)^T) * d E[V_T]/dg, a finite-difference derivative
  of the residual value with respect to cumulative wear.
- Base rent B solves: K = sum_m (B + r*E[GSU_m]) / (1+y)^m + E[V_T]/(1+y)^T
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from .residual import ResidualParams, expected_residual_value


def market_value_dcf(rental_rate_per_hour: float, hours_per_month: float,
                      remaining_months: int, annual_yield: float) -> float:
    """PV of the remaining rental stream at today's rate — the market
    component M(a, s) of the residual value model.
    """
    monthly_rate = rental_rate_per_hour * hours_per_month
    m = annual_yield / 12.0
    months = np.arange(1, remaining_months + 1)
    return float(np.sum(monthly_rate / (1 + m) ** months))


def reserve_rate(market_value: float, p_fail_fn, params: ResidualParams, g0: float, T_months: int,
                  annual_yield: float, eps: float = 1e-3) -> float:
    """Finite-difference reserve rate at wear level g0: the per-GSU reserve
    that offsets the marginal drop in expected residual value.
    """
    y_m = annual_yield / 12.0

    def ev(g):
        p_fail = p_fail_fn(g)
        return expected_residual_value(np.array([market_value]), np.array([g]), np.array([p_fail]), params)[0]

    d_ev_dg = (ev(g0 + eps) - ev(g0 - eps)) / (2 * eps)
    discount = 1.0 / (1 + y_m) ** T_months
    return float(-discount * d_ev_dg)


@dataclass
class PricingResult:
    base_rent_monthly: float
    reserve_rate_per_gsu: float
    effective_rent_per_month: np.ndarray  # B + r * E[GSU_m], length T
    flat_worst_case_rent: float


def solve_base_rent(acquisition_cost: float, expected_gsu_per_month: np.ndarray, reserve_rate_val: float,
                     expected_terminal_value: float, annual_yield: float) -> float:
    """Solves K = sum_m (B + r*E[GSU_m]) / (1+y)^m + E[V_T]/(1+y)^T for B."""
    T = len(expected_gsu_per_month)
    y_m = annual_yield / 12.0
    months = np.arange(1, T + 1)
    discount = 1.0 / (1 + y_m) ** months
    terminal_pv = expected_terminal_value / (1 + y_m) ** T

    reserve_pv = np.sum(reserve_rate_val * expected_gsu_per_month * discount)

    def residual(B: float) -> float:
        rent_pv = np.sum(B * discount)
        return (rent_pv + reserve_pv + terminal_pv) - acquisition_cost

    lo, hi = 0.0, acquisition_cost
    return float(brentq(residual, lo, hi))


def price_lease(acquisition_cost: float, rental_rate_per_hour: float, hours_per_month: float,
                 expected_gsu_per_month: np.ndarray, p_fail_at_term, params: ResidualParams,
                 annual_yield: float) -> PricingResult:
    T = len(expected_gsu_per_month)
    market_value_now = market_value_dcf(rental_rate_per_hour, hours_per_month, T, annual_yield)
    g_T = float(np.sum(expected_gsu_per_month))
    p_fail_T = p_fail_at_term(g_T)
    e_v_T = expected_residual_value(
        np.array([market_value_now]), np.array([g_T]), np.array([p_fail_T]), params
    )[0]

    r = reserve_rate(market_value_now, p_fail_at_term, params, g0=g_T / 2, T_months=T, annual_yield=annual_yield)
    B = solve_base_rent(acquisition_cost, expected_gsu_per_month, r, e_v_T, annual_yield)
    effective = B + r * expected_gsu_per_month

    # Flat worst-case lease: reserve priced at the max plausible GSU/month
    # instead of the archetype's actual expectation — what a lessor without
    # condition data would have to charge everyone.
    worst_case_gsu = np.full(T, float(np.max(expected_gsu_per_month)) * 1.5)
    B_worst = solve_base_rent(acquisition_cost, worst_case_gsu, r, e_v_T, annual_yield)

    return PricingResult(
        base_rent_monthly=B,
        reserve_rate_per_gsu=r,
        effective_rent_per_month=effective,
        flat_worst_case_rent=B_worst + r * float(np.max(worst_case_gsu)),
    )
