"""GPU Stress Unit (GSU): the per-use wear metric (IMPLEMENTATION PLAN.md §6.6,
configs/wear_params.yaml). Three terms:

  1. time-at-temperature-and-power (Arrhenius-type acceleration, regime-dependent Ea)
  2. thermal cycling (rainflow-counted, Coffin-Manson-style)
  3. memory-error growth (where logged)

1 GSU = 1 hour at reference conditions (T_ref, P_ref, no cycling), so a
reference trace of exactly T_ref/P_ref for N hours with no cycling gives
N GSU.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .rainflow_cycles import rainflow_count

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "wear_params.yaml"


@dataclass
class GsuParams:
    T_ref_c: float
    P_ref_w: float
    Ea_cool_ev: float
    Ea_hot_ev: float
    regime_boundary_c: float
    power_exponent_n: float
    delta_T_ref_c: float
    w_c: float
    m_exponent: float
    w_m: float
    k_boltzmann_ev: float

    @classmethod
    def from_yaml(cls, path: Path = DEFAULT_CONFIG_PATH, tdp_w: float = 700.0) -> "GsuParams":
        cfg = yaml.safe_load(Path(path).read_text())
        ref = cfg["reference"]
        tat = cfg["time_at_temperature_power"]
        cyc = cfg["thermal_cycling"]
        mem = cfg["memory_error_growth"]
        return cls(
            T_ref_c=ref["T_ref_c"],
            P_ref_w=ref["P_ref_frac_tdp"] * tdp_w,
            Ea_cool_ev=tat["activation_energy_ev"]["cool_regime"]["value"],
            Ea_hot_ev=tat["activation_energy_ev"]["hot_regime"]["value"],
            regime_boundary_c=tat["activation_energy_ev"]["regime_boundary_c"],
            power_exponent_n=tat["power_exponent_n"]["value"],
            delta_T_ref_c=cyc["delta_T_ref_c"],
            w_c=cyc["weight_w_c"]["value"],
            m_exponent=cyc["exponent_m"]["value"],
            w_m=mem["weight_w_m"]["value"],
            k_boltzmann_ev=tat["boltzmann_ev_per_k"],
        )


def _ea_for_temp(temp_c: np.ndarray, params: GsuParams) -> np.ndarray:
    """Regime-dependent activation energy: low below the boundary (Summit-like,
    liquid-cooled, weak temperature effect), high above it (Titan-like,
    air-cooled, strong temperature effect). See configs/wear_params.yaml.
    """
    return np.where(temp_c < params.regime_boundary_c, params.Ea_cool_ev, params.Ea_hot_ev)


def time_at_temp_power_term(temp_c: np.ndarray, power_w: np.ndarray, dt_hours: np.ndarray,
                             params: GsuParams) -> float:
    temp_c = np.asarray(temp_c, dtype=float)
    power_w = np.asarray(power_w, dtype=float)
    dt_hours = np.asarray(dt_hours, dtype=float)

    valid = np.isfinite(temp_c) & np.isfinite(power_w)
    if not np.any(valid):
        return 0.0

    T_k = temp_c[valid] + 273.15
    T_ref_k = params.T_ref_c + 273.15
    Ea = _ea_for_temp(temp_c[valid], params)

    accel_thermal = np.exp((Ea / params.k_boltzmann_ev) * (1.0 / T_ref_k - 1.0 / T_k))
    accel_power = np.power(np.clip(power_w[valid] / params.P_ref_w, 1e-6, None), params.power_exponent_n)

    return float(np.sum(dt_hours[valid] * accel_thermal * accel_power))


def thermal_cycling_term(temp_c: np.ndarray, params: GsuParams) -> float:
    cycles = rainflow_count(np.asarray(temp_c, dtype=float))
    if not cycles:
        return 0.0
    total = sum(c.count * (c.range / params.delta_T_ref_c) ** params.m_exponent for c in cycles)
    return float(params.w_c * total)


def memory_error_term(remapped_rows_delta: float, params: GsuParams) -> float:
    if remapped_rows_delta is None or not np.isfinite(remapped_rows_delta):
        return 0.0
    return float(params.w_m * remapped_rows_delta)


def gsu(temp_c: np.ndarray, power_w: np.ndarray, dt_hours: np.ndarray,
        remapped_rows_delta: float = 0.0, params: GsuParams | None = None) -> dict:
    """Computes total GSU and the three component terms for one device over
    one window (e.g. one lease-month).
    """
    params = params or GsuParams.from_yaml()
    t1 = time_at_temp_power_term(temp_c, power_w, dt_hours, params)
    t2 = thermal_cycling_term(temp_c, params)
    t3 = memory_error_term(remapped_rows_delta, params)
    return {"gsu": t1 + t2 + t3, "time_at_temp_power": t1, "thermal_cycling": t2, "memory_error": t3}


def reference_trace_gsu_per_hour(params: GsuParams | None = None) -> float:
    """Sanity check: a trace held exactly at (T_ref, P_ref) with no cycling
    gives 1.0 GSU per hour (IMPLEMENTATION PLAN.md §6.6 acceptance criterion).
    """
    params = params or GsuParams.from_yaml()
    temp = np.full(2, params.T_ref_c)
    power = np.full(2, params.P_ref_w)
    dt = np.array([1.0, 1.0])
    result = gsu(temp, power, dt, remapped_rows_delta=0.0, params=params)
    return result["gsu"] / np.sum(dt)
