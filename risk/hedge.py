"""Minimum-variance hedge of the India rental curve against public US index
futures (CME/Silicon Data H100 Rental Index Futures, or ICE/Ornn OCPI), plus
a bootstrap confidence interval since public US points are sparse
(IMPLEMENTATION PLAN.md §6.8). Labelled preliminary when the sample is small.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HedgeResult:
    hedge_ratio: float      # h* = rho * sigma_V / sigma_F
    correlation: float
    basis_share: float      # 1 - rho^2: fraction of variance not explained by the hedge
    ci_low: float
    ci_high: float
    n_obs: int
    preliminary: bool


def min_variance_hedge(india_returns: np.ndarray, us_index_returns: np.ndarray,
                        n_bootstrap: int = 2000, seed: int = 0,
                        preliminary_threshold: int = 30) -> HedgeResult:
    india_returns = np.asarray(india_returns, dtype=float)
    us_index_returns = np.asarray(us_index_returns, dtype=float)
    if len(india_returns) != len(us_index_returns):
        raise ValueError("series must be the same length (paired observations)")

    n = len(india_returns)
    rho = float(np.corrcoef(india_returns, us_index_returns)[0, 1])
    sigma_v = float(np.std(india_returns, ddof=1))
    sigma_f = float(np.std(us_index_returns, ddof=1))
    h_star = rho * sigma_v / sigma_f if sigma_f > 0 else 0.0
    basis_share = 1 - rho ** 2

    rng = np.random.default_rng(seed)
    boot_h = np.empty(n_bootstrap)
    idx_all = np.arange(n)
    for b in range(n_bootstrap):
        idx = rng.choice(idx_all, size=n, replace=True)
        v, f = india_returns[idx], us_index_returns[idx]
        sv, sf = np.std(v, ddof=1), np.std(f, ddof=1)
        r = np.corrcoef(v, f)[0, 1] if sf > 0 and sv > 0 else 0.0
        boot_h[b] = r * sv / sf if sf > 0 else 0.0

    ci_low, ci_high = np.percentile(boot_h, [2.5, 97.5])

    return HedgeResult(
        hedge_ratio=h_star, correlation=rho, basis_share=basis_share,
        ci_low=float(ci_low), ci_high=float(ci_high), n_obs=n,
        preliminary=n < preliminary_threshold,
    )
