"""Rainflow cycle counting for thermal cycling stress (IMPLEMENTATION PLAN.md
§6.6). A minimal ASTM E1049-85-style implementation (four-point algorithm),
so the wear engine has no hard external dependency beyond numpy; swap in the
`rainflow` package if a faster/validated implementation is preferred later.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Cycle:
    range: float
    mean: float
    count: float  # 1.0 for a full cycle, 0.5 for a half (residual) cycle


def _turning_points(series: np.ndarray) -> np.ndarray:
    s = np.asarray(series, dtype=float)
    if len(s) < 3:
        return s
    d = np.diff(s)
    nz_idx = np.where(d != 0)[0]
    if len(nz_idx) < 2:
        return s
    signs = np.sign(d[nz_idx])
    keep = [nz_idx[0]]  # first non-flat segment start
    for i in range(1, len(signs)):
        if signs[i] != signs[i - 1]:
            keep.append(nz_idx[i])
    keep.append(nz_idx[-1] + 1)  # final point
    idx = sorted(set([0] + keep))
    return s[idx]


def rainflow_count(series: np.ndarray) -> list[Cycle]:
    """Four-point rainflow counting. Returns full and half cycles."""
    points = _turning_points(np.asarray(series, dtype=float))
    if len(points) < 3:
        return []

    stack: list[float] = []
    cycles: list[Cycle] = []

    for x in points:
        stack.append(x)
        while len(stack) >= 4:
            s0, s1, s2, s3 = stack[-4], stack[-3], stack[-2], stack[-1]
            r1 = abs(s1 - s0)
            r2 = abs(s2 - s1)
            r3 = abs(s3 - s2)
            if r2 <= r1 and r2 <= r3:
                cycles.append(Cycle(range=r2, mean=(s1 + s2) / 2, count=1.0))
                del stack[-3:-1]
            else:
                break

    # Remaining stack: count as half-cycles (standard rainflow residual handling).
    for i in range(len(stack) - 1):
        cycles.append(Cycle(range=abs(stack[i + 1] - stack[i]), mean=(stack[i] + stack[i + 1]) / 2, count=0.5))

    return cycles


def cycle_count_total(series: np.ndarray) -> float:
    return sum(c.count for c in rainflow_count(series))
