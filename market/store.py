"""Parquet snapshot storage for daily market data (IMPLEMENTATION PLAN.md
§6.7: "GitHub Actions cron commits Parquet snapshots daily").
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_STORE_DIR = Path(__file__).resolve().parents[1] / "data" / "market"


def append_snapshot(df: pd.DataFrame, store_dir: Path = DEFAULT_STORE_DIR) -> Path:
    store_dir.mkdir(parents=True, exist_ok=True)
    date = df["date"].iloc[0] if len(df) else "unknown"
    path = store_dir / f"{date}.parquet"
    df.to_parquet(path, index=False)
    return path


def load_range(store_dir: Path = DEFAULT_STORE_DIR) -> pd.DataFrame:
    files = sorted(store_dir.glob("*.parquet"))
    if not files:
        return pd.DataFrame(columns=["date", "sku", "provider", "term", "region", "usd_per_gpu_hr"])
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
