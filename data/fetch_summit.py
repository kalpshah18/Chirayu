"""Fetches a reduced extract of Summit power/thermal telemetry
(IMPLEMENTATION PLAN.md §2.1, §6.1), from the ClusterWise Hugging Face mirror
by default (easier format than raw OLCF Globus access), falling back to the
OLCF DOI page's Globus endpoint.

Run this from an environment with real network access to Hugging Face —
Kaggle, Colab, or the IITGN CPU cluster, per §5's compute plan. It is not
expected to run inside a sandboxed agent environment with an allowlisted
network; use data/synth.py there instead to exercise the rest of the
pipeline against a schema-correct fixture.

Usage:
    python data/fetch_summit.py --nodes 200 --out data/reduced/summit/
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

COLUMNS = (
    "timestamp,hostname,"
    + ",".join(f"gpu{i}_core_temp,gpu{i}_mem_temp,gpu{i}_power" for i in range(6))
    + ",ps0_input_power,ps1_input_power,xid,label,allocation_id"
).split(",")


def fetch(n_nodes: int, out_dir: Path, seed: int = 0) -> None:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "Install the 'datasets' package (pip install datasets) on a machine "
            "with network access to Hugging Face, then re-run this script.\n"
            f"Original error: {e}"
        )

    ds = load_dataset("MachaParfait/ClusterWise", streaming=True, split="train")

    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stratify by cabinet: Summit hostnames encode position (e.g. aXXnYY), so a
    # random sample of hostnames seen in the stream already stratifies roughly
    # by cabinet, provided the stream isn't pre-sorted by node.
    buffers: dict[str, list[dict]] = {}
    seen_hosts: set[str] = set()
    target_hosts: set[str] | None = None

    for row in ds:
        host = row.get("hostname")
        if host is None:
            continue
        if target_hosts is None:
            seen_hosts.add(host)
            if len(seen_hosts) >= n_nodes * 20:  # build a candidate pool before sampling
                target_hosts = set(rng.sample(sorted(seen_hosts), k=min(n_nodes, len(seen_hosts))))
            continue
        if host not in target_hosts:
            continue
        buffers.setdefault(host, []).append({k: row.get(k) for k in COLUMNS if k in row})

    import pandas as pd

    for host, rows in buffers.items():
        df = pd.DataFrame(rows)
        safe_name = host.replace(":", "_").replace("/", "_")
        df.to_parquet(out_dir / f"{safe_name}.parquet", index=False)

    print(f"wrote {len(buffers)} node files to {out_dir}")
    print("Record exact files and row counts in data/README.md, per §6.1's acceptance criterion.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=200)
    parser.add_argument("--out", type=Path, default=Path("data/reduced/summit"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    fetch(args.nodes, args.out, args.seed)
