"""Fetches the labelled subset of the MIT Supercloud dataset (IMPLEMENTATION
PLAN.md §2.1, §6.1) and resamples its 100ms series to 1s (mean for power and
utilization, max for temperature), for use in wear/profiles.ipynb.

Requires network access to the public S3 bucket (`--no-sign-request`, no AWS
credentials needed) or the AWS CLI installed. Run on a machine with real
network access; see data/synth.py for a schema-correct local fixture.

Usage:
    python data/fetch_supercloud.py --out data/reduced/supercloud/
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

S3_PREFIX = "s3://mit-supercloud-dataset/datacenter-challenge/202201/"


def fetch_labelled_subset(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["aws", "s3", "cp", S3_PREFIX, str(out_dir), "--recursive", "--no-sign-request",
           "--exclude", "*", "--include", "*labelled*"]
    print("running:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(
            "aws s3 cp failed. Make sure the AWS CLI is installed and this machine "
            f"has network access to S3.\nstderr:\n{result.stderr}"
        )
    print(result.stdout)


def resample_to_1s(raw_csv: Path, out_parquet: Path) -> None:
    import pandas as pd

    df = pd.read_csv(raw_csv, parse_dates=["timestamp"])
    df = df.set_index("timestamp")

    agg = {}
    for col in df.columns:
        lc = col.lower()
        if "temp" in lc:
            agg[col] = "max"
        elif "power" in lc or "util" in lc or "mem" in lc:
            agg[col] = "mean"
        else:
            agg[col] = "first"

    resampled = df.resample("1s").agg(agg).dropna(how="all")
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    resampled.reset_index().to_parquet(out_parquet, index=False)
    print(f"wrote {len(resampled)} rows (from {len(df)}) to {out_parquet}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/reduced/supercloud"))
    parser.add_argument("--resample-only", type=Path, default=None,
                         help="skip the S3 fetch and just resample an already-downloaded CSV")
    args = parser.parse_args()

    if args.resample_only:
        resample_to_1s(args.resample_only, args.out / (args.resample_only.stem + ".parquet"))
    else:
        fetch_labelled_subset(args.out)
        print("Now resample each downloaded CSV with --resample-only <path>.")
