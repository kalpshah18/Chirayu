"""Builds the synthetic 500-lease book (IMPLEMENTATION PLAN.md §6.8), with
three lessees mapped to archetypes (configs/lessees.yaml), an H100/B200 mix,
24/36-month terms, seeded for reproducibility. Lessee behaviour (GSU/hour
distribution per archetype) is set by wear/profiles.ipynb from real MIT
Supercloud jobs; book.py only samples from it.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


@dataclass
class BookLease:
    lease_id: str
    lessee_id: str
    archetype: str
    sku: str
    term_months: int
    gsu_per_hour_mean: float
    gsu_per_hour_std: float
    acquisition_cost: float


SKU_COST = {"H100": 2_500_000.0, "B200": 4_000_000.0}  # INR, illustrative


def build_book(n_leases: int | None = None, seed: int | None = None) -> list[BookLease]:
    lessees_cfg = yaml.safe_load((CONFIG_DIR / "lessees.yaml").read_text())
    risk_cfg = yaml.safe_load((CONFIG_DIR / "risk_params.yaml").read_text())

    n_leases = n_leases or risk_cfg["book"]["n_leases"]
    seed = seed if seed is not None else risk_cfg["book"]["seed"]
    rng = np.random.default_rng(seed)

    archetypes = lessees_cfg["archetypes"]
    lessees = lessees_cfg["lessees"]
    lessee_ids = list(lessees.keys())
    weights = np.array([lessees[lid]["n_gpus"] for lid in lessee_ids], dtype=float)
    weights = weights / weights.sum()

    sku_names = list(risk_cfg["book"]["sku_mix"].keys())
    sku_weights = np.array(list(risk_cfg["book"]["sku_mix"].values()), dtype=float)
    terms = risk_cfg["book"]["term_months_options"]

    leases = []
    for i in range(n_leases):
        lessee_id = rng.choice(lessee_ids, p=weights)
        archetype_name = lessees[lessee_id]["archetype"]
        arch = archetypes[archetype_name]
        sku = rng.choice(sku_names, p=sku_weights)
        term = int(rng.choice(terms))

        leases.append(BookLease(
            lease_id=f"L{i:04d}",
            lessee_id=lessee_id,
            archetype=archetype_name,
            sku=sku,
            term_months=term,
            gsu_per_hour_mean=arch["gsu_per_hour_mean"],
            gsu_per_hour_std=arch["gsu_per_hour_std"],
            acquisition_cost=SKU_COST[sku],
        ))

    return leases
