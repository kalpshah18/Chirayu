"""Lessee power-cap optimizer (IMPLEMENTATION PLAN.md §6.8):

    minimise  c_e * E(p) + r * GSU(p) - v * Throughput(p)   over power cap p

The power->temperature effect of a cap comes from Zhao et al.'s published
MIT Supercloud V100 power-capping study; the throughput penalty per cap
level is taken from the same paper's reported figures where available, and
is a labelled assumption with a stated range otherwise (configs/risk_params.yaml).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wear.gsu import GsuParams, gsu


@dataclass
class PowerCapCurve:
    """A labelled stand-in for the published power-capping relationship.

    energy_frac, throughput_frac and temp_drop_c are indexed to cap_frac (as a
    fraction of TDP). Where Zhao et al. report a figure directly, source is
    the citation; otherwise it is "assumption" with the range noted in
    configs/risk_params.yaml.
    """

    cap_frac: np.ndarray
    energy_frac: np.ndarray       # energy use relative to uncapped, per unit time
    throughput_frac: np.ndarray   # throughput relative to uncapped
    temp_drop_c: np.ndarray       # degrees C cooler than uncapped, at this cap

    @classmethod
    def from_published_shape(cls, cap_fracs: np.ndarray) -> "PowerCapCurve":
        """A monotonic curve shaped like Zhao et al.'s reported pattern: capping
        power gives a roughly proportional energy saving with a sub-proportional
        throughput loss (the point of power capping at HPC scale), and cooler
        operation. Where the paper gives exact figures for specific cap levels,
        replace the relevant entries directly and keep this function for the
        interpolated remainder, with the substitution noted here.
        """
        cap_fracs = np.asarray(cap_fracs, dtype=float)
        energy_frac = cap_fracs  # assumption: energy scales ~linearly with the cap
        # assumption: throughput degrades sub-linearly below the cap (diminishing
        # loss as utilization-bound stages dominate), per the paper's qualitative finding
        throughput_frac = 1 - 0.4 * (1 - cap_fracs) ** 1.5
        temp_drop_c = 25.0 * (1 - cap_fracs)  # assumption: linear in cap reduction
        return cls(cap_fracs, energy_frac, throughput_frac, temp_drop_c)


def optimize_power_cap(curve: PowerCapCurve, baseline_power_w: float, baseline_temp_c: float,
                        hours: float, energy_cost_per_kwh: float, reserve_rate_per_gsu: float,
                        value_per_unit_throughput: float, gsu_params: GsuParams) -> dict:
    results = []
    for i, cap in enumerate(curve.cap_frac):
        power_w = baseline_power_w * curve.energy_frac[i]
        temp_c = baseline_temp_c - curve.temp_drop_c[i]
        energy_kwh = power_w * hours / 1000.0
        energy_cost = energy_kwh * energy_cost_per_kwh

        g = gsu(
            temp_c=np.full(2, temp_c), power_w=np.full(2, power_w), dt_hours=np.array([hours / 2, hours / 2]),
            remapped_rows_delta=0.0, params=gsu_params,
        )["gsu"]
        wear_cost = reserve_rate_per_gsu * g
        throughput_value = value_per_unit_throughput * curve.throughput_frac[i] * hours

        objective = energy_cost + wear_cost - throughput_value
        results.append({
            "cap_frac": float(cap), "power_w": power_w, "temp_c": temp_c,
            "energy_cost": energy_cost, "wear_cost": wear_cost,
            "throughput_value": throughput_value, "objective": objective,
        })

    best = min(results, key=lambda r: r["objective"])
    return {"grid": results, "optimal": best}
