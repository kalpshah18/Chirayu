"""Detector 1: power divergence (IMPLEMENTATION PLAN.md §6.4).

Fits P_psu(t) = alpha + beta * (sum P_gpu(t) + sum P_cpu(t)) + eps(t) on a
clean calibration window, then runs a two-sided CUSUM on the normalised
residual of held-out data. alpha absorbs fans/memory/network/conversion
loss; beta >= 1 absorbs PSU efficiency. Summit's reported ~11% average
offset between PSU sums and switchboard meters (§2.2) is exactly why this
is a fitted offset-and-gain model and not a raw equality check.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DivergenceFit:
    alpha: float
    beta: float
    residual_std: float


def total_reported_power(df: pd.DataFrame) -> np.ndarray:
    gpu_cols = [c for c in df.columns if c.startswith("gpu") and c.endswith("_power")]
    total = df[gpu_cols].sum(axis=1, skipna=True).to_numpy(dtype=float)
    cpu_cols = [c for c in df.columns if c.startswith("cpu") and c.endswith("_power")]
    if cpu_cols:
        total = total + df[cpu_cols].sum(axis=1, skipna=True).to_numpy(dtype=float)
    return total


def witness_power(df: pd.DataFrame) -> np.ndarray:
    p0 = df.get("ps0_input_power", pd.Series(np.nan, index=df.index)).to_numpy(dtype=float)
    p1 = df.get("ps1_input_power", pd.Series(np.nan, index=df.index)).to_numpy(dtype=float)
    return np.nan_to_num(p0, nan=0.0) + np.nan_to_num(p1, nan=0.0)


def fit_divergence(calibration_df: pd.DataFrame) -> DivergenceFit:
    x = total_reported_power(calibration_df)
    y = witness_power(calibration_df)
    valid = np.isfinite(x) & np.isfinite(y) & (y > 0)
    x, y = x[valid], y[valid]
    if len(x) < 10:
        raise ValueError("not enough clean rows to calibrate the divergence model")

    beta, alpha = np.polyfit(x, y, 1)
    resid = y - (alpha + beta * x)
    return DivergenceFit(alpha=float(alpha), beta=float(beta), residual_std=float(np.std(resid, ddof=1)))


def cusum(residuals_z: np.ndarray, k: float = 0.5, h: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """Two-sided CUSUM on a z-scored residual series. k is the allowance
    (slack) in std units, h is the decision threshold. Returns (statistic,
    alarm_mask) where alarm_mask[i] is True once either side exceeds h.
    """
    n = len(residuals_z)
    s_hi = np.zeros(n)
    s_lo = np.zeros(n)
    for i in range(1, n):
        s_hi[i] = max(0.0, s_hi[i - 1] + residuals_z[i] - k)
        s_lo[i] = min(0.0, s_lo[i - 1] + residuals_z[i] + k)
    alarm = (s_hi > h) | (s_lo < -h)
    stat = np.where(s_hi > -s_lo, s_hi, s_lo)
    return stat, alarm


def score(df: pd.DataFrame, fit: DivergenceFit, k: float = 0.5, h: float = 5.0) -> dict:
    x = total_reported_power(df)
    y = witness_power(df)
    predicted = fit.alpha + fit.beta * x
    resid = y - predicted
    z = np.where(fit.residual_std > 0, resid / fit.residual_std, 0.0)
    # Treat rows with no witness reading as "cannot score", not as agreement.
    has_witness = y > 0
    z = np.where(has_witness, z, 0.0)
    stat, alarm = cusum(z, k=k, h=h)
    alarm = alarm & has_witness
    return {
        "residual": resid,
        "z": z,
        "cusum_stat": stat,
        "alarm": alarm,
        "witness_silent": ~has_witness,
    }
