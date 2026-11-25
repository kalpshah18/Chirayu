"""Detector 2: thermal consistency (IMPLEMENTATION PLAN.md §6.4).

Per-GPU first-order thermal model:

    T(t+dt) = T(t) + (dt/tau) * (T_amb + R*P(t) - T(t))

fit on clean data, i.e. a discrete RC thermal circuit where R*P(t) is the
steady-state temperature rise driven by power. Summit's reported near-linear
power-to-temperature relation (~62W power spread, ~15.8C temperature spread
in a 4,608-node job; §2.2) is the empirical basis. Faking power without
matching temperature — or the reverse — shows up as a residual here, which
is what makes the "consistent_fake" attack in attacks/inject.py the hard
case: it has to get both channels mutually consistent to hide.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


@dataclass
class ThermalFit:
    tau_steps: float   # time constant, in sample steps
    T_amb: float
    R: float            # deg C per watt, steady-state gain
    residual_std: float


def _predict_next(T_and_P, tau, T_amb, R):
    T, P = T_and_P
    return T + (1.0 / tau) * (T_amb + R * P - T)


def fit_thermal(temp_c: np.ndarray, power_w: np.ndarray) -> ThermalFit:
    valid = np.isfinite(temp_c) & np.isfinite(power_w)
    T, P = temp_c[valid], power_w[valid]
    if len(T) < 20:
        raise ValueError("not enough clean rows to calibrate the thermal model")

    T_now, P_now, T_next = T[:-1], P[:-1], T[1:]
    p0 = (10.0, float(np.min(T_now)), 0.1)
    bounds = ([1.0, -20.0, 0.0], [10_000.0, 60.0, 2.0])
    try:
        popt, _ = curve_fit(_predict_next, (T_now, P_now), T_next, p0=p0, bounds=bounds, maxfev=5000)
    except RuntimeError:
        # Fall back to a plain linear steady-state fit (T ~= T_amb + R*P) if the
        # dynamic fit doesn't converge (e.g. on a very short or noisy window).
        R, T_amb = np.polyfit(P, T, 1)
        popt = (50.0, T_amb, R)
    tau, T_amb, R = popt
    pred_next = _predict_next((T_now, P_now), tau, T_amb, R)
    resid = T_next - pred_next
    return ThermalFit(tau_steps=float(tau), T_amb=float(T_amb), R=float(R), residual_std=float(np.std(resid, ddof=1)) or 1.0)


def score(temp_c: np.ndarray, power_w: np.ndarray, fit: ThermalFit) -> dict:
    T, P = temp_c, power_w
    valid = np.isfinite(T) & np.isfinite(P)
    pred = np.full_like(T, np.nan, dtype=float)
    pred[1:] = np.where(
        valid[:-1] & valid[1:],
        _predict_next((T[:-1], P[:-1]), fit.tau_steps, fit.T_amb, fit.R),
        np.nan,
    )
    resid = T - pred
    z = resid / fit.residual_std
    return {"predicted": pred, "residual": resid, "z": z}
