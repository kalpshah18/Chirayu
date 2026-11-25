"""Indexed rent variant and reserve-account bookkeeping (IMPLEMENTATION PLAN.md
§6.8): B_m = B * I_m / I_0, and tracking the accumulated wear reserve against
realised GSU draws (for the dashboard's "book" and "lease pricer" pages).
"""
from __future__ import annotations

import numpy as np


def indexed_rent(base_rent: float, index_path: np.ndarray) -> np.ndarray:
    """B_m = B * I_m / I_0 — rent that tracks a published rental index instead
    of staying flat, so the lessor isn't stuck below market if rental prices
    rise.
    """
    index_path = np.asarray(index_path, dtype=float)
    return base_rent * index_path / index_path[0]


class ReserveAccount:
    """Accumulates r * GSU_realised each month and releases it against the
    terminal expected-value shortfall at lease end.
    """

    def __init__(self, reserve_rate_per_gsu: float):
        self.rate = reserve_rate_per_gsu
        self.balance = 0.0
        self.history: list[float] = []

    def accrue(self, gsu_this_month: float) -> float:
        contribution = self.rate * gsu_this_month
        self.balance += contribution
        self.history.append(self.balance)
        return contribution

    def settle(self, shortfall: float) -> float:
        """Draws down the reserve against a realised residual-value shortfall.
        Returns the amount still uncovered (0 if the reserve was sufficient).
        """
        covered = min(self.balance, shortfall)
        self.balance -= covered
        return shortfall - covered
