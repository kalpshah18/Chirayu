"""Residual value model (IMPLEMENTATION PLAN.md §6.8, doc §"Financial model").

    E[V_T] = M(a_T, s_T) * C(g_T) * (1 - p_f(g_T)) + p_f(g_T) * S

M is the market value from the rental DCF (risk/pricing.py's market path),
C(g) = exp(-lambda*g) is the condition factor in cumulative GSU, p_f(g) is
failure probability from the calibrated hazard, and S is salvage value.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ResidualParams:
    lam: float            # lambda in C(g) = exp(-lambda*g)
    salvage_frac: float   # S as a fraction of original market value


def fit_lambda_from_haircut(hours: float, haircut_low: float, haircut_high: float,
                             gsu_per_hour: float = 1.0) -> float:
    """Calibrates lambda so that C(g) at g = hours*gsu_per_hour matches the
    midpoint of a reseller's reported haircut range (configs/wear_params.yaml
    residual_value_condition_factor.lambda_h100.calibration_points).
    """
    haircut_mid = (haircut_low + haircut_high) / 2.0
    c_target = 1.0 - haircut_mid
    g = hours * gsu_per_hour
    if g <= 0 or not (0 < c_target < 1):
        raise ValueError("invalid calibration inputs")
    return float(-np.log(c_target) / g)


def condition_factor(g: np.ndarray | float, lam: float) -> np.ndarray | float:
    return np.exp(-lam * np.asarray(g, dtype=float))


def expected_residual_value(market_value: np.ndarray, g: np.ndarray, p_fail: np.ndarray,
                             params: ResidualParams) -> np.ndarray:
    c = condition_factor(g, params.lam)
    salvage = params.salvage_frac * market_value
    return market_value * c * (1 - p_fail) + p_fail * salvage
