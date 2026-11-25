# Chirayu

A pricing, risk and financing desk for GPU operating leases written out of
GIFT IFSC. Prices leases, marks every leased GPU each month, simulates
lease-end losses, hedges market risk and sizes funding tranches — all on top
of per-GPU condition data the lessee cannot quietly edit.

Built for the GIFT FutureLab Grand Challenge 2026–27 finale. See
[`IMPLEMENTATION PLAN.md`](IMPLEMENTATION%20PLAN.md) for the full design,
milestones and citations.

## What is real, what is published, what is synthetic

- **Real fleet data (once fetched):** Summit power/thermal telemetry and MIT
  Supercloud job traces, via `data/fetch_summit.py` and
  `data/fetch_supercloud.py`. Not yet fetched in this environment — see
  `data/README.md`.
- **Published results, cited directly:** failure-rate anchors (NCSA Delta,
  Meta RSC, Llama 3), Summit's thermal findings, power-capping curves. See
  `IMPLEMENTATION PLAN.md` §2.2 and `configs/wear_params.yaml` /
  `configs/risk_params.yaml`, where every parameter is tagged with its source
  or labelled `assumption`.
- **Synthetic, for development only:** `data/synth.py` generates a
  schema-correct fixture so the pipeline runs and is testable without network
  access to Globus/Hugging Face/S3. It is explicitly *not* a substitute for
  the real evaluation runs required by the plan's acceptance criteria.

## Status

This is the first working cut of the core pipeline (M0-level scope): the
signed telemetry chain, the witness channel, both tamper detectors, the GSU
wear engine, and the full risk/pricing engine all run end to end and are unit
tested. Not yet built: the React dashboard (`web/`), the real data fetch runs
against Summit/Supercloud/Titan, the Cox-model calibration notebooks, and the
live market scrapers beyond the interface in `market/scrapers/`.

## Layout

See `IMPLEMENTATION PLAN.md` §4 for the full repository layout and the
rationale behind each module.

```
agent/      telemetry backends (replay + NVML), signed hash chain, sampler
witness/    independent power-witness channel (Summit PSU data; physical design for the pilot)
attacks/    spoof/replay/drop/swap injectors, used to evaluate detection
ledger/     verification, the two tamper detectors, Merkle roots, a minimal FastAPI service
wear/       the GPU Stress Unit (GSU) model and rainflow cycle counting
risk/       residual value, pricing, Monte Carlo, tranching, hedge, insurance, the lessee optimizer, the book
market/     rental-price normalization and index construction
data/       fetch scripts for the three real datasets, plus a synthetic fixture generator
eval/       tamper-detection evaluation across many nodes and attack types
configs/    every model parameter, each tagged with its source or "assumption"
tests/      unit tests for the chain, GSU, pricing, divergence detector, tranching, Monte Carlo
```

## Running it

```bash
pip install -e ".[dev]"
pytest
```

To exercise the full chain end to end on synthetic data (no network needed):

```python
from data.synth import make_synthetic_node_month
from agent.backends.replay import ReplayBackend
from agent.chain import SigningKey, verify_chain
from agent.sampler import Sampler

make_synthetic_node_month(n_steps=500, seed=1).to_parquet("node.parquet", index=False)
backend = ReplayBackend("node.parquet")
device = backend.devices()[0]
key = SigningKey.generate()
sampler = Sampler(backend, device, key)

records = [r for batch in sampler.run_batches(10) for r in batch]
assert verify_chain(records, key.public_hex()) == []
```

To run the ledger service locally:

```bash
uvicorn ledger.app:app --reload
```

## Known gaps to close before the finale

- Tamper-detector thresholds (`ledger/divergence.py`'s CUSUM `h`, `k`) are
  currently illustrative. §6.4 is explicit that these are "chosen on held-out
  clean data for a target false-alarm rate" — that calibration has to run
  against real Summit data, not the synthetic fixture, before the
  acceptance targets in §6.5 mean anything.
- The three real datasets haven't been fetched yet in this environment; see
  `data/README.md` for exactly what's needed and where to run each script.
- `wear/profiles.ipynb`, `wear/calibrate_summit.ipynb` and
  `wear/calibrate_titan.ipynb` don't exist yet — the archetype distributions
  in `configs/lessees.yaml` are placeholders until those notebooks run on
  real data.
