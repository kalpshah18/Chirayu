"""Generates a physically-plausible synthetic node-month file with the same
schema the real fetch scripts produce, so the rest of the pipeline (agent,
ledger, wear, risk) can be built and tested before — or without — a working
network path to Globus/HuggingFace/S3 from a given machine.

This is NOT a substitute for the real datasets at evaluation time. Every
acceptance criterion in IMPLEMENTATION PLAN.md §6 that says "on real Summit
nodes" or "on MIT Supercloud traces" must be run against data/fetch_summit.py
and data/fetch_supercloud.py output, not this file. Tests that only need a
schema-correct fixture (chain verification, divergence detector math, GSU
math) use this generator so they don't require network access or a 240 GB
download to run in CI.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_node_month(
    hostname: str = "synth:n01",
    n_gpus: int = 6,
    n_steps: int = 8640,          # 1 day at 10s cadence
    start: str = "2020-08-14T00:00:00Z",
    freq_s: int = 10,
    base_temp_c: float = 40.0,
    base_power_w: float = 200.0,
    psu_offset_frac: float = 0.11,   # Summit's reported ~11% PSU-vs-switchboard offset (§2.2)
    cpu_power_w: float = 150.0,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start=start, periods=n_steps, freq=f"{freq_s}s", tz="UTC")

    data: dict[str, np.ndarray] = {"timestamp": timestamps, "hostname": [hostname] * n_steps}

    total_gpu_power = np.zeros(n_steps)
    for i in range(n_gpus):
        # Slow utilization cycle (job-like) plus noise, so there's real
        # cycling for the rainflow counter and the thermal model to see.
        phase = rng.uniform(0, 2 * np.pi)
        util = 0.5 + 0.4 * np.sin(np.linspace(0, 6 * np.pi, n_steps) + phase)
        power = base_power_w * (0.3 + 0.7 * util) + rng.normal(0, 3, n_steps)
        power = np.clip(power, 20, None)
        # Near-linear power->temperature coupling, per the Summit finding (§2.2).
        temp_core = base_temp_c + 0.08 * (power - base_power_w) + rng.normal(0, 0.5, n_steps)
        temp_mem = temp_core + 3 + rng.normal(0, 0.3, n_steps)

        data[f"gpu{i}_power"] = power
        data[f"gpu{i}_core_temp"] = temp_core
        data[f"gpu{i}_mem_temp"] = temp_mem
        total_gpu_power += power

    data["cpu0_power"] = np.full(n_steps, cpu_power_w) + rng.normal(0, 5, n_steps)

    node_true_power = total_gpu_power + data["cpu0_power"] + 40.0  # fans/memory/network/conversion loss
    # Two PSUs split the load roughly evenly, with the fitted offset/gain
    # Summit itself shows against switchboard meters.
    psu_total = node_true_power * (1 + psu_offset_frac)
    data["ps0_input_power"] = psu_total / 2 + rng.normal(0, 2, n_steps)
    data["ps1_input_power"] = psu_total / 2 + rng.normal(0, 2, n_steps)

    data["xid"] = [None] * n_steps
    data["label"] = [None] * n_steps
    data["allocation_id"] = ["synthetic-job-0"] * n_steps

    return pd.DataFrame(data)


if __name__ == "__main__":
    import sys

    out_path = sys.argv[1] if len(sys.argv) > 1 else "data/reduced/synth_sample_node.parquet"
    from pathlib import Path

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df = make_synthetic_node_month()
    df.to_parquet(out_path, index=False)
    print(f"wrote {len(df)} rows to {out_path}")
