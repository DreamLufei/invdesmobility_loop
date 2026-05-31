# invdesmobility_loop

`invdesmobility_loop` is a lightweight orchestration bridge for closed-loop
inverse design. It connects candidate generation and surrogate screening in
`invDesMobility` with first-principles mobility validation in `2d-mobility`.

The bridge extracts trusted mobility feedback from completed validation
batches, prepares feedback-aware training data, launches the next generation
and screening round, publishes selected candidates for validation and records
round-level manifests.

## What Is Included

- `invdesmobility/closed_loop.py`: closed-loop round orchestration.
- `invdesmobility/batch_launcher.py`: MongoDB-backed bridge into
  `2d-mobility` batch runs.
- `scripts/launch_closed_loop_round_tmux.sh`: tmux launcher for long-running
  rounds.
- `scripts/show_closed_loop_round_progress.sh`: read-only progress summary.
- `tests/`: unit tests for CLI, source loading, MongoDB handoff and
  closed-loop helper behavior.

Generated round directories under `runs/` are intentionally excluded from the
public repository.

## Requirements

- Python 3.11 or newer.
- The companion repositories available locally:
  - `invDesMobility`
  - `2d-mobility`
- MongoDB reachable by both this bridge and `2d-mobility`.
- Installed scientific environments required by `invDesMobility`.
- Configured VASP, PostgreSQL, MongoDB and LLM settings required by
  `2d-mobility`.

## Installation

```bash
git clone https://github.com/DreamLufei/invdesmobility_loop.git
cd invdesmobility_loop

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the lightweight tests:

```bash
python -m pytest -q
```

## Configuration

Pass repository paths explicitly on the command line, or set them in the
environment used by your launcher:

```bash
export INVDES_ROOT=/path/to/invDesMobility
export TWO_D_MOBILITY_ROOT=/path/to/2d-mobility
```

Local secrets for long-running launchers can be stored outside the repository:

```bash
~/.config/invdesmobility/secrets.env
```

Do not commit credential files. The bridge itself does not require API keys
unless it launches downstream workflows that need them.

## Running One Round

```bash
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

For long-running cluster use:

```bash
bash scripts/launch_closed_loop_round_tmux.sh \
  4 \
  /path/to/completed/2d-mobility/batch
```

Monitor:

```bash
bash scripts/show_closed_loop_round_progress.sh loop_04
```

## Dry Runs

Build round outputs without publishing candidates or launching downstream VASP
validation:

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

## Feedback Semantics

Only trusted validation output is used as positive feedback. Failed
calculations, rejected physics checks and low-confidence channels are retained
as audit records but are not used as positive fine-tuning examples.

Generated structures are deduplicated against the current generated pool, the
reference source library and previously submitted or trusted feedback
structures. This prevents repeated validation of the same composition/structure
family across rounds.

## Public-Release Boundaries

This repository intentionally excludes:

- local `runs/` directories and long campaign logs;
- private MongoDB connection strings;
- API keys and other private environment values;
- VASP inputs or raw calculation folders.

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for an additional checklist.

## License

The license terms are defined by the `LICENSE` file when one is added to the
repository.
