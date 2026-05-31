# Reproducibility guide

This repository is the closed-loop bridge between the generative screening code
in `invDesMobility` and the first-principles mobility runtime in
`2d-mobility`.

## Scope

The bridge reproduces the campaign-level orchestration:

- extraction of trusted mobility feedback from completed `2d-mobility` runs;
- construction of feedback-aware DiffCSP and ALIGNN datasets;
- launch of the next generation and screening round in `invDesMobility`;
- publication of strict90 top-k candidates to MongoDB;
- launch of the downstream `2d-mobility` validation batch;
- writing per-round manifests and progress summaries.

It does not run VASP directly and does not assign mobility labels. Mobility
labels are assigned by the downstream `2d-mobility` workflow.

## Clone layout

Place the three repositories side by side:

```bash
git clone https://github.com/DreamLufei/2d-mobility.git
git clone https://github.com/DreamLufei/invDesMobility.git
git clone https://github.com/DreamLufei/invdesmobility_loop.git
```

Typical side-by-side paths:

```text
/path/to/2d-mobility
/path/to/invDesMobility
/path/to/invdesmobility_loop
```

Use CLI flags if your paths differ.

## Install

```bash
cd invdesmobility_loop
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest -q
```

The bridge calls scripts in the two companion repositories, so their
environments must also be installed.

## Required services and secrets

Required for a real closed-loop round:

- MongoDB reachable by the bridge and `2d-mobility`;
- the `2d-mobility` environment configured with LLM, PostgreSQL and VASP
  settings;
- `invDesMobility` environments installed for DiffCSP, ALIGNN, MEGNET and
  PhononBench/MatterSim;
- a completed parent `2d-mobility` batch directory used as feedback.

Local secrets may be placed in:

```bash
~/.config/invdesmobility/secrets.env
```

Do not commit this file.

## Run one closed-loop round

```bash
cd invdesmobility_loop

python -m invdesmobility closed-loop-round \
  --round-index 4 \
  --feedback-batch-root /path/to/completed/2d-mobility/batch \
  --invdes-root /path/to/invDesMobility \
  --two-d-mobility-root /path/to/2d-mobility \
  --total-samples 100000 \
  --samples-per-job 1000 \
  --top-k 10 \
  --gpu-list 0,1,2,3
```

For long runs:

```bash
bash scripts/launch_closed_loop_round_tmux.sh \
  4 \
  /path/to/completed/2d-mobility/batch
```

Monitor:

```bash
bash scripts/show_closed_loop_round_progress.sh loop_04
watch -n 30 'bash scripts/show_closed_loop_round_progress.sh loop_04'
```

## Dry-run and audit modes

Build round outputs without publishing to MongoDB:

```bash
python -m invdesmobility closed-loop-round \
  --round-index 4 \
  --feedback-batch-root /path/to/completed/batch \
  --publish-dry-run \
  --skip-downstream-run
```

Launch downstream validation in dry-run mode:

```bash
python -m invdesmobility closed-loop-round \
  --round-index 4 \
  --feedback-batch-root /path/to/completed/batch \
  --downstream-dry-run
```

## Important outputs

Each round writes a local directory under:

```text
runs/loop_<NN>__closed_loop_round/
```

Key files and folders:

- `01_trusted_feedback/trusted_channels.csv`;
- `01_trusted_feedback/trusted_materials.csv`;
- `01_trusted_feedback/rejected_feedback.csv`;
- `01_trusted_feedback/feedback_summary.json`;
- feedback-augmented DiffCSP and ALIGNN datasets in the companion
  `invDesMobility` tree;
- generation/screening manifests in the companion `invDesMobility/06_runs/`
  tree;
- downstream `2d-mobility` batch summary.

For manuscript reproduction, deposit round manifests, trusted feedback CSVs,
top-k CIFs and downstream validation summaries in a DOI-backed data archive.

## Public-release and NCS notes

Nature Computational Science requires code, protocols and data sufficient to
replicate and build on the published claims. This bridge repository should
contain orchestration code and configuration only. Do not commit:

- private credentials or MongoDB connection strings;
- VASP `POTCAR` files;
- raw VASP run directories;
- generated candidate pools or large run logs;
- local `.env` files.

Recommended manuscript Code Availability wording:

```text
The closed-loop orchestration bridge is available at
https://github.com/DreamLufei/invdesmobility_loop.
```
