"""The Chirayu India rental index: daily median per SKU, and its basis against
public US index points (Silicon Data, Ornn) — IMPLEMENTATION PLAN.md §6.7.
"""
from __future__ import annotations

import pandas as pd


def daily_median_index(quotes: pd.DataFrame, region_filter: str = "india") -> pd.Series:
    """`quotes` has columns: date, sku, provider, term, region, usd_per_gpu_hr.
    Returns a Series indexed by (date, sku) of the daily median price.
    """
    df = quotes[quotes["region"].str.lower() == region_filter]
    return df.groupby(["date", "sku"])["usd_per_gpu_hr"].median()


def basis_vs_us(india_index: pd.Series, us_index_points: pd.Series) -> pd.Series:
    """India minus US, aligned on (date, sku). Sparse by construction since
    public US index points are infrequent (§6.8 hedge notes this is preliminary).
    """
    aligned = pd.concat({"india": india_index, "us": us_index_points}, axis=1).dropna()
    return aligned["india"] - aligned["us"]
