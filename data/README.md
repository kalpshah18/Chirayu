# Data

Three real public datasets, plus a synthetic generator for local development
and tests that don't need the real thing (see `synth.py`'s docstring for
exactly which tests those are).

## Real datasets

| Dataset | Licence | Fetch script | Status |
| --- | --- | --- | --- |
| OLCF Summit power/thermal (via ClusterWise, Hugging Face mirror) | CC BY 4.0 | `fetch_summit.py` | Not yet run — needs a machine with Hugging Face network access (Kaggle/Colab CPU session or the IITGN CPU cluster, per IMPLEMENTATION PLAN.md §5) |
| MIT Supercloud labelled subset | Public (MIT, via AWS Open Data) | `fetch_supercloud.py` | Not yet run — needs AWS CLI + S3 access |
| ORNL Titan GPU lifetimes | Public (ORNL) | `fetch_titan.py` | Not yet run — needs GitHub access |

**Once each script has been run, record here:**
- the exact files/nodes selected and their row counts (§6.1 acceptance criterion),
- the dataset's citation and licence,
- any known caveats (e.g. Summit's missing spring/early-summer 2020 temperature readings, noted in IMPLEMENTATION PLAN.md §2.2).

## Citations

- Shin, W. et al. "Long Term Per-Component Power and Thermal Measurements of the OLCF Summit System." DOI 10.13139/OLCF/1861393.
- ClusterWise (Summit-derived), CC BY 4.0. https://huggingface.co/datasets/MachaParfait/ClusterWise
- Samsi, S. et al. "The MIT Supercloud Dataset." IEEE HPEC 2021. https://arxiv.org/abs/2108.02037
- Ostrouchov, G. et al. "GPU Lifetimes on Titan Supercomputer." SC20. DOI 10.13139/ORNLNCCS/1657202.

## Target reduced-data footprint

Per IMPLEMENTATION PLAN.md §5: < 5 GB total.
- Summit: 200–500 nodes x 1 month at 10s, random + stratified by cabinet.
- MIT Supercloud: labelled subset (3,430 jobs) first.

## Synthetic fixture (`synth.py`)

Generates one node-month Parquet file with the same column schema the fetch
scripts produce (gpu{N}_power/core_temp/mem_temp, ps0/ps1_input_power, xid,
label, allocation_id), with a physically-plausible near-linear power-to-
temperature coupling and the ~11% PSU-vs-reported offset Summit itself shows.
Used only where a test needs a schema-correct fixture and not the real
dataset's actual values — never as a substitute for the real evaluation runs
required by IMPLEMENTATION PLAN.md §6.5 and §6.6.

```
python data/synth.py data/reduced/synth_sample_node.parquet
```
