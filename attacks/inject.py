"""Injects synthetic spoofs into real replayed Summit telemetry, so tamper
detection is evaluated on real fleet data rather than one staged machine
(IMPLEMENTATION PLAN.md §6.5). Operates on the raw DataFrame a ReplayBackend
loads, before it is sampled, which is simplest for batch evaluation in
eval/tamper_eval.py; the same transforms apply equally to the live per-sample
path for the demo's "inject live" moment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

AttackKind = Literal[
    "underreport_load", "underreport_heat", "consistent_fake", "drop_data", "replay", "swap_identity"
]


@dataclass
class AttackSpec:
    kind: AttackKind
    severity: float          # k for underreport_load; degrees C for underreport_heat; unused otherwise
    start_frac: float = 0.3  # fraction into the series where the attack begins
    duration_frac: float = 0.4


def _gpu_cols(df: pd.DataFrame, suffix: str) -> list[str]:
    return [c for c in df.columns if c.startswith("gpu") and c.endswith(suffix)]


def inject(df: pd.DataFrame, spec: AttackSpec, rng: np.random.Generator | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    """Returns (attacked_df, is_attacked) where is_attacked is a boolean mask
    over rows, for scoring detection delay and false-alarm rate.
    """
    rng = rng or np.random.default_rng(0)
    df = df.copy()
    n = len(df)
    start = int(n * spec.start_frac)
    end = min(n, int(n * (spec.start_frac + spec.duration_frac)))
    mask = np.zeros(n, dtype=bool)
    mask[start:end] = True

    power_cols = _gpu_cols(df, "_power")
    temp_cols = _gpu_cols(df, "_core_temp")

    if spec.kind == "underreport_load":
        for c in power_cols:
            df.loc[mask, c] = df.loc[mask, c] * spec.severity

    elif spec.kind == "underreport_heat":
        for c in temp_cols:
            df.loc[mask, c] = df.loc[mask, c] - spec.severity

    elif spec.kind == "consistent_fake":
        # Scale power down, and shift temperature to roughly what the fitted
        # power->temperature model (ledger/thermal.py) would predict for that
        # lower power — the hardest case, since it fools a naive rule that
        # only checks "power dropped, temperature didn't".
        k = spec.severity if spec.severity else 0.8
        approx_slope_c_per_w = 15.8 / 62.0  # Summit's reported spread (§2.2), used as a rough scale
        for pc, tc in zip(power_cols, temp_cols):
            delta_p = df.loc[mask, pc] * (1 - k)
            df.loc[mask, pc] = df.loc[mask, pc] * k
            df.loc[mask, tc] = df.loc[mask, tc] - delta_p * approx_slope_c_per_w

    elif spec.kind == "drop_data":
        for c in power_cols + temp_cols:
            df.loc[mask, c] = np.nan

    elif spec.kind == "replay":
        replay_len = end - start
        src_start = max(0, start - replay_len)
        for c in power_cols + temp_cols:
            df.loc[start:end - 1, c] = df.loc[src_start:src_start + replay_len - 1, c].to_numpy()

    elif spec.kind == "swap_identity":
        # Splice in another node's values (here: a shuffled slice of this
        # file) under the same device id — identity continuity is what
        # breaks, which ledger/verify.py's chain check catches independent
        # of the physical-plausibility detectors.
        shuffled_idx = rng.permutation(np.arange(start, end))
        for c in power_cols + temp_cols:
            df.loc[start:end - 1, c] = df.loc[shuffled_idx, c].to_numpy()

    else:
        raise ValueError(f"unknown attack kind: {spec.kind}")

    return df, mask


ATTACK_GRID: list[AttackSpec] = [
    AttackSpec("underreport_load", k) for k in (0.9, 0.8, 0.6)
] + [
    AttackSpec("underreport_heat", d) for d in (3.0, 5.0, 10.0)
] + [
    AttackSpec("consistent_fake", 0.8),
    AttackSpec("drop_data", 0.0, duration_frac=0.1),
    AttackSpec("replay", 0.0, duration_frac=0.1),
    AttackSpec("swap_identity", 0.0, duration_frac=0.1),
]
