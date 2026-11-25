"""Replay backend: walks a reduced Summit/ClusterWise-shaped Parquet file and
exposes it through the same TelemetryBackend interface a live NVML backend
would use. This is what drives the finale demo (IMPLEMENTATION PLAN.md §6.2).

Expected Parquet schema per node-month file (see data/README.md and
data/fetch_summit.py):

    timestamp            (UTC, ISO-8601 or pandas Timestamp)
    hostname
    gpu{0..N-1}_core_temp
    gpu{0..N-1}_mem_temp
    gpu{0..N-1}_power
    ps0_input_power, ps1_input_power   (independent witness channel, §6.3)
    xid                  (nullable int, error code if any)
    label                (nullable str, failure-event label if any)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from . import DeviceInfo, Sample


class ReplayBackend:
    def __init__(self, parquet_path: str | Path, sku: str = "V100", speedup: float = 1.0):
        self.path = Path(parquet_path)
        self.df = pd.read_parquet(self.path)
        if "timestamp" not in self.df.columns:
            raise ValueError(f"{self.path} is missing a 'timestamp' column")
        self.df = self.df.sort_values("timestamp").reset_index(drop=True)
        self.hostname = str(self.df["hostname"].iloc[0]) if "hostname" in self.df.columns else self.path.stem
        self.sku = sku
        self.speedup = speedup  # cosmetic only: how fast a live demo would replay this

        self._n_gpus = self._count_gpus()
        self._cursor: dict[str, int] = {}

    def _count_gpus(self) -> int:
        n = 0
        while f"gpu{n}_power" in self.df.columns:
            n += 1
        if n == 0:
            raise ValueError(f"{self.path} has no gpu{{n}}_power columns")
        return n

    def devices(self) -> list[DeviceInfo]:
        return [
            DeviceInfo(id=f"{self.hostname}:gpu{i}", sku=self.sku, site="summit-replay")
            for i in range(self._n_gpus)
        ]

    def _slot(self, dev: DeviceInfo) -> int:
        return int(dev.id.rsplit("gpu", 1)[1])

    def sample(self, dev: DeviceInfo) -> Sample:
        i = self._slot(dev)
        idx = self._cursor.get(dev.id, 0)
        if idx >= len(self.df):
            idx = len(self.df) - 1  # hold at the last row rather than raising mid-demo
        row = self.df.iloc[idx]
        self._cursor[dev.id] = idx + 1

        def g(col: str) -> Optional[float]:
            val = row.get(col)
            return None if pd.isna(val) else float(val)

        ts = row["timestamp"]
        t_iso = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)

        xid_val = row.get("xid")
        xid = None if pd.isna(xid_val) else int(xid_val)

        return Sample(
            t=t_iso,
            temp_core_c=g(f"gpu{i}_core_temp"),
            temp_mem_c=g(f"gpu{i}_mem_temp"),
            power_w=g(f"gpu{i}_power"),
            xid=xid,
        )

    def witness_input_power_w(self, idx: Optional[int] = None) -> Optional[float]:
        """Node-level AC input power from the power supplies — the independent
        witness channel. Not part of TelemetryBackend; used by witness/replay_witness.py.
        """
        if idx is None:
            idx = max(self._cursor.values(), default=0) - 1
        idx = max(0, min(idx, len(self.df) - 1))
        row = self.df.iloc[idx]
        p0 = row.get("ps0_input_power")
        p1 = row.get("ps1_input_power")
        if pd.isna(p0) and pd.isna(p1):
            return None
        return float(0 if pd.isna(p0) else p0) + float(0 if pd.isna(p1) else p1)

    def __len__(self) -> int:
        return len(self.df)
