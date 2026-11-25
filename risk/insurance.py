"""Residual-value insurance premium pricing (IMPLEMENTATION PLAN.md §6.8):

    premium = (1 + theta) * E[(G - V_T)^+] / (1 + y)^T

where G is the guaranteed residual value, V_T is the (simulated) actual
residual value, and theta is the insurer's margin. Priced with and without
condition data to show the same value-of-data effect as tranching.py.
"""
from __future__ import annotations

import numpy as np


def insurance_premium(guaranteed_value: float, simulated_v_T: np.ndarray, theta: float,
                       annual_yield: float, term_months: int) -> float:
    shortfall = np.clip(guaranteed_value - simulated_v_T, 0.0, None)
    expected_shortfall = float(np.mean(shortfall))
    discount = (1 + annual_yield / 12.0) ** term_months
    return (1 + theta) * expected_shortfall / discount


def compare_with_without_data(guaranteed_value: float, simulated_v_T_with_data: np.ndarray,
                               simulated_v_T_without_data: np.ndarray, theta: float,
                               annual_yield: float, term_months: int) -> dict:
    with_data = insurance_premium(guaranteed_value, simulated_v_T_with_data, theta, annual_yield, term_months)
    without_data = insurance_premium(guaranteed_value, simulated_v_T_without_data, theta, annual_yield, term_months)
    return {
        "premium_with_data": with_data,
        "premium_without_data": without_data,
        "premium_reduction": without_data - with_data,
        "premium_reduction_pct": (without_data - with_data) / without_data * 100 if without_data else 0.0,
    }
