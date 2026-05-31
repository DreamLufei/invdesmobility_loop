#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

ROUND_INPUT="${1:-${ROUND_ID:-}}"
if [[ -z "${ROUND_INPUT}" ]]; then
  echo "Usage: bash ${0##*/} <round_id|round_index>" >&2
  exit 2
fi

if [[ "${ROUND_INPUT}" =~ ^[0-9]+$ ]]; then
  printf -v ROUND_ID 'loop_%02d' "${ROUND_INPUT}"
else
  ROUND_ID="${ROUND_INPUT}"
fi

"${PYTHON_BIN:-$(command -v python)}" - "${ROOT_DIR}" "${ROUND_ID}" <<'PY'
from __future__ import annotations

import json
import os
import subprocess
import sys

from pathlib import Path


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return dict(json.loads(path.read_text(encoding="utf-8")) or {})
    except Exception:
        return {}


def fmt_bool(value: bool) -> str:
    return "yes" if value else "no"


def tmux_session_status(name: str) -> str:
    if not name:
        return "missing"
    try:
        completed = subprocess.run(
            ["tmux", "has-session", "-t", name],
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return "unknown"
    return "alive" if completed.returncode == 0 else "missing"


def tail_line(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    except Exception:
        return ""
    return lines[-1] if lines else ""


def first_path(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def main() -> int:
    root_dir = Path(sys.argv[1])
    round_id = sys.argv[2]
    round_root = root_dir / "runs" / f"{round_id}__closed_loop_round"
    control_dir = round_root / "_control"
    runner_log = first_path(
        control_dir / "resume_after_publish_fix.log",
        control_dir / "closed_loop_runner.log",
        root_dir / "runs" / f"{round_id}__closed_loop_runner.log",
    )
    launcher_context = load_json(
        first_path(
            control_dir / "tmux_context.json",
            root_dir / "runs" / f"{round_id}__closed_loop_tmux_context.json",
        )
    )
    session_name = str(launcher_context.get("session_name") or f"invdes-{round_id}")
    runner_session_name = str(launcher_context.get("runner_session_name") or f"{session_name}-runner")

    print(f"round_id: {round_id}")
    print(f"round_root: {round_root}")
    print(f"monitor_session: {session_name} ({tmux_session_status(session_name)})")
    print(f"runner_session: {runner_session_name} ({tmux_session_status(runner_session_name)})")
    print(f"runner_log: {runner_log}")
    print()

    if not round_root.exists():
        print("round_root_exists: no")
        if runner_log.exists():
            print(f"runner_log_last: {tail_line(runner_log)}")
        return 0

    print("round_root_exists: yes")
    feedback_summary = load_json(round_root / "01_trusted_feedback" / "feedback_summary.json")
    reference_manifest = load_json(round_root / "02_reference_pool_for_dedup" / "reference_manifest.json")
    alignn_dataset_manifest_path = first_path(
        round_root / "03_alignn_dataset" / "manifest.json",
        round_root / "03_alignn_round_dataset_manifest.json",
    )
    diffcsp_dataset_manifest_path = first_path(
        round_root / "04_diffcsp_dataset" / "manifest.json",
        round_root / "04_diffcsp_round_dataset_manifest.json",
    )
    generator_ckpt_path = first_path(
        round_root / "05_generator_training" / "filtered.ckpt",
        round_root / "05_generator_filtered.ckpt",
    )
    generator_train_log_path = first_path(
        round_root / "05_generator_training" / "train.log",
        round_root / "05_generator_train.log",
    )
    alignn_train_log_path = first_path(
        round_root / "06_alignn_training" / "train.log",
        round_root / "06_alignn_train.log",
    )
    publish_manifest_path = first_path(
        round_root / "07_publish" / "manifest.json",
        round_root / "07_publish_manifest.json",
    )
    alignn_dataset_manifest = load_json(alignn_dataset_manifest_path)
    diffcsp_dataset_manifest = load_json(diffcsp_dataset_manifest_path)
    publish_manifest = load_json(publish_manifest_path)
    loop_manifest = load_json(round_root / "loop_manifest.json")

    checkpoints = {
        "trusted_feedback": (round_root / "01_trusted_feedback" / "trusted_materials.csv").exists(),
        "reference_pool": (round_root / "02_reference_pool_for_dedup" / "reference_manifest.json").exists(),
        "alignn_dataset": alignn_dataset_manifest_path.exists(),
        "diffcsp_dataset": diffcsp_dataset_manifest_path.exists(),
        "generator_ckpt": generator_ckpt_path.exists(),
        "alignn_ckpt": (Path(os.environ.get("INVDES_ROOT", str(root_dir.parent / "invDesMobility"))) / "04_models" / "04_alignn_mobility" / f"alignn_mobility_round_{round_id.split('_')[-1]}" / "best_model.pt").exists(),
        "publish_manifest": publish_manifest_path.exists(),
        "loop_manifest": (round_root / "loop_manifest.json").exists(),
    }
    print("stage_checkpoints:")
    for key, value in checkpoints.items():
        print(f"  {key}: {fmt_bool(value)}")
    print()

    if feedback_summary:
        print("feedback_summary:")
        counts = dict(feedback_summary.get("counts") or {})
        for key in ("trusted_channel_count", "trusted_material_count", "rejected_entry_count"):
            if key in counts:
                print(f"  {key}: {counts.get(key)}")
        print(f"  batch_root: {feedback_summary.get('batch_root')}")
        print()

    if reference_manifest:
        counts = dict(reference_manifest.get("counts") or {})
        print("reference_pool:")
        print(f"  base_reference_count: {counts.get('base_reference_count')}")
        historical_count = counts.get("historical_computed_reference_count", counts.get("feedback_reference_count"))
        print(f"  historical_computed_reference_count: {historical_count}")
        print(f"  total_reference_count: {counts.get('total_reference_count')}")
        print()

    if alignn_dataset_manifest:
        print("alignn_dataset:")
        print(f"  total_rows: {alignn_dataset_manifest.get('total_rows')}")
        print(f"  feedback_rows: {alignn_dataset_manifest.get('feedback_rows')}")
        print()

    if diffcsp_dataset_manifest:
        print("diffcsp_dataset:")
        print(f"  unique_material_count: {diffcsp_dataset_manifest.get('unique_material_count')}")
        print(f"  train_row_count: {diffcsp_dataset_manifest.get('train_row_count')}")
        print(f"  feedback_weight: {diffcsp_dataset_manifest.get('feedback_weight')}")
        print()

    if publish_manifest:
        counts = dict(publish_manifest.get("counts") or {})
        print("publish_manifest:")
        for key in ("source_rows", "publishable_rows", "inserted_count", "updated_count", "skipped_count"):
            if key in counts:
                print(f"  {key}: {counts.get(key)}")
        print()

    downstream_summary = None
    if loop_manifest:
        downstream = dict(loop_manifest.get("downstream_batch") or {})
        summary_path = dict(downstream.get("final_state") or {}).get("batch", {}).get("summary_path")
        if summary_path:
            downstream_summary = load_json(Path(summary_path))
        print("loop_manifest:")
        print(f"  generation_run_id: {dict(loop_manifest.get('generation') or {}).get('run_id')}")
        print(f"  screening_run_id: {dict(loop_manifest.get('screening') or {}).get('run_id')}")
        print(f"  downstream_batch_tag: {downstream.get('batch_tag')}")
        print(f"  downstream_returncode: {downstream.get('returncode')}")
        print()

    if downstream_summary:
        print("downstream_summary:")
        for key in ("processed", "succeeded", "failed", "skipped", "scientifically_passed", "scientifically_warning", "scientifically_failed"):
            if key in downstream_summary:
                print(f"  {key}: {downstream_summary.get(key)}")
        print()

    for label, path in (
        ("runner_log_last", runner_log),
        ("generator_train_last", generator_train_log_path),
        ("alignn_train_last", alignn_train_log_path),
    ):
        line = tail_line(path)
        if line:
            print(f"{label}: {line}")

    return 0


raise SystemExit(main())
PY
