"""Securitization tranching (IMPLEMENTATION PLAN.md §6.8): quantile
attachment points for a target exceedance probability, plus the
"value of data" comparison (same simulation with per-device GSU known vs.
drawn from a wide prior).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Tranche:
    name: str
    attachment: float    # loss level at which this tranche starts absorbing losses
    exceedance_prob: float


def attachment_point(losses: np.ndarray, exceedance_prob: float) -> float:
    """P(L > A) <= exceedance_prob  =>  A is the (1 - exceedance_prob) quantile."""
    return float(np.quantile(losses, 1 - exceedance_prob))


def build_tranches(losses: np.ndarray, target_exceedance: dict[str, float]) -> list[Tranche]:
    return [Tranche(name=name, attachment=attachment_point(losses, p), exceedance_prob=p)
            for name, p in target_exceedance.items()]


def value_of_data(losses_with_data: np.ndarray, losses_without_data: np.ndarray,
                   target_exceedance: dict[str, float], notional: float,
                   spread_senior: float, spread_mezzanine: float) -> dict:
    """Funding saving from having real per-device condition data vs. pricing
    everyone off a wide prior: Saving ~= delta_A_senior * N * (s_mezz - s_senior) * D.

    A smaller senior attachment point with data (because losses are more
    predictable, i.e. a tighter distribution) means a larger share of the
    book can be funded at the cheaper senior spread instead of the mezzanine
    spread — this is the funding saving in basis points.
    """
    a_with = attachment_point(losses_with_data, target_exceedance["senior"])
    a_without = attachment_point(losses_without_data, target_exceedance["senior"])
    delta_a = a_without - a_with  # positive if data tightens the senior attachment point
    spread_diff = spread_mezzanine - spread_senior
    saving = delta_a * spread_diff  # currency units, per unit of duration D folded into spreads already
    saving_bps_of_notional = (saving / notional) * 10_000 if notional else 0.0

    return {
        "attachment_with_data": a_with,
        "attachment_without_data": a_without,
        "delta_attachment": delta_a,
        "funding_saving": saving,
        "funding_saving_bps": saving_bps_of_notional,
    }
