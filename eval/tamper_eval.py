"""Evaluates tamper detection across many real (or, for local dev, synthetic)
node files and every attack in attacks.ATTACK_GRID (IMPLEMENTATION PLAN.md
§6.5). This is the strongest technical result in the project: measured on
fleet data, not staged.

Usage:
    python -m eval.tamper_eval --nodes-dir data/reduced/summit --out eval_results.csv
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from attacks.inject import ATTACK_GRID, inject
from ledger.divergence import fit_divergence, score as divergence_score


@dataclass
class AttackResult:
    node: str
    attack_kind: str
    severity: float
    detected_within_1h: bool
    detected_within_24h: bool
    detection_delay_steps: float | None
    n_false_alarms_clean: int


def evaluate_node(df: pd.DataFrame, step_seconds: float = 10.0,
                   k: float = 0.5, h: float = 5.0) -> list[AttackResult]:
    results: list[AttackResult] = []
    n = len(df)
    calib_end = n // 3  # calibrate on the first third, "clean" by construction here

    calib = df.iloc[:calib_end]
    try:
        fit = fit_divergence(calib)
    except ValueError:
        return results  # not enough clean rows in this node file to calibrate

    # Clean false-alarm rate, on the held-out (non-calibration) clean data.
    clean_eval = df.iloc[calib_end:]
    clean_scored = divergence_score(clean_eval, fit, k=k, h=h)
    n_false_alarms = int(np.sum(clean_scored["alarm"]))
    node_days_clean = len(clean_eval) * step_seconds / 86400.0

    steps_per_hour = 3600.0 / step_seconds
    rng = np.random.default_rng(0)

    for spec in ATTACK_GRID:
        attacked_df, attack_mask = inject(df, spec, rng=rng)
        scored = divergence_score(attacked_df, fit, k=k, h=h)
        alarm = scored["alarm"]

        attack_start_idx = int(np.argmax(attack_mask)) if attack_mask.any() else None
        if attack_start_idx is None:
            continue

        alarms_after_start = np.where(alarm[attack_start_idx:])[0]
        delay_steps = float(alarms_after_start[0]) if len(alarms_after_start) else None

        detected_1h = delay_steps is not None and delay_steps <= steps_per_hour
        detected_24h = delay_steps is not None and delay_steps <= steps_per_hour * 24

        results.append(AttackResult(
            node="", attack_kind=spec.kind, severity=spec.severity,
            detected_within_1h=detected_1h, detected_within_24h=detected_24h,
            detection_delay_steps=delay_steps,
            n_false_alarms_clean=n_false_alarms,
        ))

    return results, node_days_clean


def evaluate_directory(nodes_dir: Path, step_seconds: float = 10.0) -> pd.DataFrame:
    rows = []
    total_clean_node_days = 0.0
    for path in sorted(Path(nodes_dir).glob("*.parquet")):
        df = pd.read_parquet(path)
        out = evaluate_node(df, step_seconds=step_seconds)
        if not out:
            continue
        results, node_days_clean = out
        total_clean_node_days += node_days_clean
        for r in results:
            r.node = path.stem
            rows.append(r.__dict__)

    result_df = pd.DataFrame(rows)
    if len(result_df):
        result_df.attrs["total_clean_node_days"] = total_clean_node_days
    return result_df


def summarize(result_df: pd.DataFrame) -> pd.DataFrame:
    if result_df.empty:
        return result_df
    summary = result_df.groupby(["attack_kind", "severity"]).agg(
        n_nodes=("node", "nunique"),
        detection_rate_1h=("detected_within_1h", "mean"),
        detection_rate_24h=("detected_within_24h", "mean"),
        median_delay_steps=("detection_delay_steps", "median"),
    ).reset_index()
    total_clean_node_days = result_df.attrs.get("total_clean_node_days", np.nan)
    total_false_alarms = result_df.groupby("node")["n_false_alarms_clean"].first().sum()
    false_alarms_per_1000_node_days = (
        total_false_alarms / total_clean_node_days * 1000 if total_clean_node_days else np.nan
    )
    summary["false_alarms_per_1000_node_days"] = false_alarms_per_1000_node_days
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("eval_results.csv"))
    args = parser.parse_args()

    df = evaluate_directory(args.nodes_dir)
    df.to_csv(args.out, index=False)
    print(summarize(df).to_string())
