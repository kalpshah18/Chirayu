"""Condition-priced GPU leases, calibrated on real fleet data.

Data: ORNL Titan GPU lifetimes (Ostrouchov et al., SC20; github.com/olcf/TitanGPULife),
one record per GPU: 30,207 NVIDIA K20X cards, >100,000 GPU-years, with cabinet
position (cage 0 = bottom, nearest the cool air inlet; cage 2 = top, where air
leaves the cabinet hottest).

Question answered: if identical GPUs are run under different operating
conditions, how much does that change lease-end value, the rent a lessor must
charge, and the loss a lender must absorb? Position is the condition variable
here because it is what Titan recorded; Chirayu replaces it with measured
per-GPU stress (GSU) in a live lease.

Outputs (analysis/out/): km_by_cage.pdf, rent_by_condition.pdf, tail_loss.pdf, results.json
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "public" / "titan" / "gc_summary_loc.csv"
OUT = ROOT / "analysis" / "out"
OUT.mkdir(parents=True, exist_ok=True)

# Lease assumptions (labelled; see configs/risk_params.yaml for the rest)
COST = 30_000.0          # USD per H100-class GPU, assumption
TERM_M = 36              # lease term, months
YIELD = 0.12             # lessor's target annual yield (configs/risk_params.yaml)
MKT_FRAC_T = 0.30        # market value of a healthy GPU at lease end, fraction of cost (assumption)
SALVAGE_FRAC_T = 0.05    # value of a failed GPU at lease end, fraction of cost (assumption)
N_LEASES = 500
N_PATHS = 50_000
SEED = 20261128

CAGE_NAME = {0: "Bottom (cool inlet)", 1: "Middle", 2: "Top (hot exhaust)"}
COLORS = {0: "#2b6cb0", 1: "#b7791f", 2: "#c53030"}

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})


def load() -> pd.DataFrame:
    d = pd.read_csv(DATA)
    d["E"] = d["dead"].astype(int)
    return d


def survival(d: pd.DataFrame) -> dict:
    res = {}
    fig, ax = plt.subplots(figsize=(4.1, 2.55))
    kmf = KaplanMeierFitter()
    for c in (0, 1, 2):
        s = d[d.cage == c]
        kmf.fit(s.years, s.E, label=CAGE_NAME[c])
        kmf.plot_survival_function(ax=ax, ci_show=True, color=COLORS[c], lw=1.6)
        res[c] = {
            "n_gpus": int(len(s)),
            "failures": int(s.E.sum()),
            "gpu_years": float(s.years.sum()),
            "fail_per_100_gpu_years": float(100 * s.E.sum() / s.years.sum()),
            "surv_3y": float(kmf.predict(3.0)),
            "surv_4y": float(kmf.predict(4.0)),
        }
    ax.axvline(3.0, color="0.5", ls=":", lw=1)
    ax.text(3.05, 0.04, "lease\nends", color="0.35", fontsize=7.5)
    ax.set_xlabel("Years in service")
    ax.set_ylabel("Share of GPUs still working")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, 5.5)
    ax.legend(frameon=False, loc="lower left", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT / "km_by_cage.pdf")
    plt.close(fig)
    return res


def cox(d: pd.DataFrame) -> dict:
    x = pd.get_dummies(d[["cage", "node", "batch"]].astype(str), drop_first=True).astype(float)
    x["years"] = d.years
    x["E"] = d.E
    cph = CoxPHFitter()
    cph.fit(x, "years", "E")
    s = cph.summary
    out = {}
    for k in ("cage_1", "cage_2"):
        out[k] = {"hr": float(s.loc[k, "exp(coef)"]),
                  "lo": float(s.loc[k, "exp(coef) lower 95%"]),
                  "hi": float(s.loc[k, "exp(coef) upper 95%"])}
    out["concordance"] = float(cph.concordance_index_)
    out["n_events"] = int(d.E.sum())
    return out


def monthly_rent(p_fail: float) -> tuple[float, float]:
    """Level monthly rent B that returns the lessor's cost at YIELD:
    COST = sum_m B/(1+y)^m + E[V_T]/(1+y)^T, with
    E[V_T] = (1-p_f)*MKT + p_f*SALVAGE  (residual model of risk/residual.py
    with the condition factor folded into the failure term)."""
    y = YIELD / 12
    ann = (1 - (1 + y) ** -TERM_M) / y
    ev = COST * ((1 - p_fail) * MKT_FRAC_T + p_fail * SALVAGE_FRAC_T)
    b = (COST - ev / (1 + y) ** TERM_M) / ann
    return b, ev


def pricing(surv: dict) -> dict:
    out = {}
    pf = {c: 1 - surv[c]["surv_3y"] for c in surv}
    weights = {c: surv[c]["n_gpus"] for c in surv}
    pf_pool = sum(pf[c] * weights[c] for c in pf) / sum(weights.values())
    flat, ev_flat = monthly_rent(pf_pool)
    for c in (0, 1, 2):
        b, ev = monthly_rent(pf[c])
        out[c] = {"p_fail_36m": pf[c], "rent": b, "ev_T": ev,
                  "flat_minus_fair": flat - b,
                  "over_term": (flat - b) * TERM_M}
    out["flat"] = {"p_fail_36m": pf_pool, "rent": flat, "ev_T": ev_flat}

    fig, ax = plt.subplots(figsize=(6.3, 1.75))
    ys = [0, 1, 2]
    vals = [out[c]["rent"] for c in (0, 1, 2)]
    ax.axvline(flat, color="0.25", ls="--", lw=1)
    ax.text(flat, -0.75, f"flat rent ${flat:,.0f}", fontsize=7.5, ha="center", va="bottom")
    for y, v, c in zip(ys, vals, (0, 1, 2)):
        ax.plot([flat, v], [y, y], color=COLORS[c], lw=2)
        ax.plot(v, y, "o", color=COLORS[c], ms=7)
        side = -1 if v < flat else 1
        ax.text(v + side * 2.5, y, f"${v:,.0f}/month   ({100*pf[c]:.1f}% fail by month 36)",
                va="center", ha="right" if side < 0 else "left", fontsize=7.5)
    ax.set_yticks(ys, [CAGE_NAME[c] for c in (0, 1, 2)])
    ax.set_ylim(2.6, -1.0)
    ax.set_xlim(min(vals) - 45, max(vals) + 45)
    ax.set_xlabel("Break-even monthly rent per GPU, USD (36-month lease, 12% yield, \\$30k GPU)")
    fig.tight_layout()
    fig.savefig(OUT / "rent_by_condition.pdf")
    plt.close(fig)
    return out


def book_losses(surv: dict, price: dict) -> dict:
    """Lender's view of a 500-GPU lease book.

    Each GPU either survives the term (worth MKT) or fails (worth SALVAGE).
    The lessor has priced in an expected residual; loss = expected residual
    priced in minus realised residual, summed over the book.

    Scenario A, flat pricing + adverse selection: careful lessees (cool) find
    the flat rent too high and leave; the book fills with hot operators.
    Scenario B, flat pricing, representative mix (what the lessor assumed).
    Scenario C, condition pricing: each lease's residual is priced from its
    own measured condition, so the expected shortfall is zero by construction
    for every mix; only true randomness remains.
    """
    rng = np.random.default_rng(SEED)
    pf = {c: 1 - surv[c]["surv_3y"] for c in (0, 1, 2)}
    gap = COST * (MKT_FRAC_T - SALVAGE_FRAC_T)
    tot = sum(surv[c]["n_gpus"] for c in (0, 1, 2))
    mixes = {"representative": {c: surv[c]["n_gpus"] / tot for c in (0, 1, 2)},  # Titan's own fleet mix
             "adverse": {0: 0.05, 1: 0.25, 2: 0.70}}
    pf_flat = price["flat"]["p_fail_36m"]

    def simulate(mix, priced_pf_fn):
        counts = {c: N_LEASES * mix[c] for c in mix}
        loss = np.zeros(N_PATHS)
        for c, n in counts.items():
            fails = rng.binomial(int(round(n)), pf[c], size=N_PATHS)
            n = int(round(n))
            expected_fails_priced = n * priced_pf_fn(c)
            loss += (fails - expected_fails_priced) * gap
        return loss

    res = {}
    scen = {
        "flat_representative": simulate(mixes["representative"], lambda c: pf_flat),
        "flat_adverse": simulate(mixes["adverse"], lambda c: pf_flat),
        "condition_adverse": simulate(mixes["adverse"], lambda c: pf[c]),
    }
    book_value = N_LEASES * COST
    for k, v in scen.items():
        res[k] = {"expected_loss": float(v.mean()),
                  "var99": float(np.quantile(v, 0.99)),
                  "expected_loss_pct_book": float(100 * v.mean() / book_value),
                  "var99_pct_book": float(100 * np.quantile(v, 0.99) / book_value)}

    fig, ax = plt.subplots(figsize=(6.3, 2.0))
    style = {"flat_adverse": ("Flat rent, hot-heavy book", "#c53030"),
             "flat_representative": ("Flat rent, assumed mix", "0.45"),
             "condition_adverse": ("Condition-priced, hot-heavy book", "#2b6cb0")}
    from scipy.stats import gaussian_kde
    grid = np.linspace(-250, 450, 600)
    for k, (lab, col) in style.items():
        dens = gaussian_kde(scen[k] / 1e3, bw_method=0.25)(grid)
        ax.fill_between(grid, dens, color=col, alpha=0.25, lw=0)
        ax.plot(grid, dens, color=col, lw=1.4, label=lab)
        v = res[k]["var99"] / 1e3
        ax.axvline(v, color=col, lw=1, ls="--")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Residual-value loss on a 500-GPU book at lease end, USD thousands (dashed: 99th percentile)")
    ax.set_yticks([])
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "tail_loss.pdf")
    plt.close(fig)
    res["book_value"] = book_value
    res["mixes"] = mixes
    return res


def main():
    d = load()
    surv = survival(d)
    hr = cox(d)
    price = pricing(surv)
    loss = book_losses(surv, price)
    results = {
        "dataset": {"n_gpus": int(len(d)), "gpu_years": float(d.years.sum()),
                    "failures": int(d.E.sum())},
        "survival_by_cage": surv,
        "cox_hazard_ratios_vs_bottom_cage": hr,
        "pricing": {str(k): v for k, v in price.items()},
        "book": loss,
        "assumptions": {"cost_usd": COST, "term_months": TERM_M, "annual_yield": YIELD,
                        "market_frac_at_T": MKT_FRAC_T, "salvage_frac_at_T": SALVAGE_FRAC_T,
                        "n_leases": N_LEASES, "n_paths": N_PATHS, "seed": SEED},
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2, default=float))
    print(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
