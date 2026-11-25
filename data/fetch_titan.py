"""Clones olcf/TitanGPULife and locates its cleaned per-GPU summary file
(IMPLEMENTATION PLAN.md §2.1, §6.1), used for hot-regime hazard calibration
in wear/calibrate_titan.ipynb.

Usage:
    python data/fetch_titan.py --out data/reduced/titan/
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/olcf/TitanGPULife"


def fetch(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    clone_dir = out_dir / "TitanGPULife"
    if clone_dir.exists():
        print(f"{clone_dir} already exists, pulling latest")
        subprocess.run(["git", "-C", str(clone_dir), "pull"], check=True)
    else:
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(clone_dir)], check=True)

    candidates = list(clone_dir.rglob("*.csv")) + list(clone_dir.rglob("*summary*"))
    if not candidates:
        raise SystemExit(
            f"No obvious per-GPU summary file found under {clone_dir}. "
            "Inspect the repo manually and update data/README.md with the exact path used."
        )
    print("Candidate summary files found:")
    for c in candidates:
        print(" -", c)
    print("Pick the per-GPU summary file and record it, with its DOI "
          "(10.13139/ORNLNCCS/1657202), in data/README.md.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/reduced/titan"))
    args = parser.parse_args()
    fetch(args.out)
