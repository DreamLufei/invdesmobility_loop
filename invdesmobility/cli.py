from __future__ import annotations

import argparse
import json
import sys

from pathlib import Path
from typing import Any

from . import DEFAULT_2D_MOBILITY_ROOT, DEFAULT_INVDES_ROOT, DEFAULT_SOURCE_RUN_LINK, PROJECT_ROOT
from .batch_launcher import launch_batch
from .closed_loop import ClosedLoopError, run_closed_loop_round
from .models import BatchLaunchResult, RunSource, SyncSummary
from .mongo_sync import MONGO_COLLECTION_NAME, MONGO_DB_NAME, open_collection, sync_source_items
from .preflight import PreflightError, run_preflight
from .source_loader import resolve_run_source


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="invDesMobility to 2d-mobility orchestration CLI")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Ingest the resolved ranked CIF set and launch 2d-mobility batch")
    run_parser.add_argument(
        "--source-run-link",
        default=DEFAULT_SOURCE_RUN_LINK,
        help="Symlink or run directory pointing at the invDesMobility pipeline output.",
    )
    run_parser.add_argument(
        "--mongo-db",
        default=MONGO_DB_NAME,
        help=f"MongoDB database name (default: {MONGO_DB_NAME}).",
    )
    run_parser.add_argument(
        "--mongo-collection",
        default=MONGO_COLLECTION_NAME,
        help=f"MongoDB collection name (default: {MONGO_COLLECTION_NAME}).",
    )
    run_parser.add_argument(
        "--expected-cif-count",
        type=int,
        default=None,
        help="Optional exact ranked-CIF count check before launch; omit to auto-accept the resolved source size.",
    )
    run_parser.add_argument(
        "--batch-tag",
        default=None,
        help="Optional explicit 2d-mobility batch tag. Useful when launching from tmux and tracking a known run root.",
    )
    run_parser.add_argument("--force", action="store_true", help="Reset the current run's materials back to pending.")
    run_parser.add_argument(
        "--dry-run-batch",
        action="store_true",
        help="Launch 2d-mobility with --dry-run after ingesting the Mongo documents.",
    )

    loop_parser = subparsers.add_parser(
        "closed-loop-round",
        help="Run one full closed-loop round by orchestrating invDesMobility generation/screening and 2d-mobility batch execution.",
    )
    loop_parser.add_argument("--round-index", type=int, required=True, help="Closed-loop round index, e.g. 1 for loop_01.")
    loop_parser.add_argument("--round-id", default=None, help="Optional explicit round id. Defaults to loop_<NN>.")
    loop_parser.add_argument(
        "--parent-round-id",
        default=None,
        help="Optional parent round id. Defaults to round_00_bootstrap for round 1, otherwise loop_<NN-1>.",
    )
    loop_parser.add_argument(
        "--feedback-source-round-id",
        default=None,
        help="Optional round id label to stamp on the extracted feedback snapshot. Defaults to parent round id.",
    )
    loop_parser.add_argument(
        "--feedback-batch-root",
        required=True,
        help="Completed 2d-mobility batch root used as the high-confidence feedback source.",
    )
    loop_parser.add_argument(
        "--invdes-root",
        default=DEFAULT_INVDES_ROOT,
        help=f"invDesMobility repository root (default: {DEFAULT_INVDES_ROOT}).",
    )
    loop_parser.add_argument(
        "--two-d-mobility-root",
        default=DEFAULT_2D_MOBILITY_ROOT,
        help=f"2d-mobility repository root (default: {DEFAULT_2D_MOBILITY_ROOT}).",
    )
    loop_parser.add_argument("--mongo-uri", default=None, help="Optional Mongo URI override. Defaults to the 2d-mobility env.")
    loop_parser.add_argument("--mongo-db", default=MONGO_DB_NAME, help=f"Mongo database name (default: {MONGO_DB_NAME}).")
    loop_parser.add_argument(
        "--mongo-collection",
        default=MONGO_COLLECTION_NAME,
        help=f"Mongo collection name (default: {MONGO_COLLECTION_NAME}).",
    )
    loop_parser.add_argument("--total-samples", type=int, default=100000, help="Number of structures to generate this round.")
    loop_parser.add_argument("--samples-per-job", type=int, default=1000, help="Structures generated per DiffCSP worker job.")
    loop_parser.add_argument(
        "--num-batches-to-samples",
        type=int,
        default=1,
        help="DiffCSP generation multiplier passed through to run_multigpu.sh.",
    )
    loop_parser.add_argument("--top-k", type=int, default=10, help="How many ranked candidates to send into strict90.")
    loop_parser.add_argument("--gpu-list", default="0,1,2,3", help="GPU list for generation, e.g. 0,1,2,3.")
    loop_parser.add_argument(
        "--phononbench-gpu-list",
        default=None,
        help="Optional GPU list for PhononBench screening. Defaults to the generation GPU list.",
    )
    loop_parser.add_argument("--phononbench-dim", default="2 2 2", help="PhononBench supercell dimension string.")
    loop_parser.add_argument(
        "--phononbench-subparts-per-gpu",
        type=int,
        default=1,
        help="PhononBench workload sharding per GPU.",
    )
    loop_parser.add_argument("--phononbench-model", default="mattersim-v1", help="PhononBench backend model id.")
    loop_parser.add_argument("--bandgap-threshold", type=float, default=0.2, help="Bandgap threshold in eV.")
    loop_parser.add_argument(
        "--formation-threshold",
        type=float,
        default=0.0,
        help="Formation energy threshold in eV/atom. Candidates must be below this value.",
    )
    loop_parser.add_argument(
        "--phonon-imag-threshold",
        type=float,
        default=0.1,
        help="Maximum allowed imaginary phonon magnitude.",
    )
    loop_parser.add_argument(
        "--target-crystal-system",
        default="orthorhombic",
        help="Crystal-system filter for the screening pipeline.",
    )
    loop_parser.add_argument(
        "--strict90-max-angle-deviation-deg",
        type=float,
        default=0.6,
        help="Maximum angle deviation allowed before strict90 snapping rejects a candidate.",
    )
    loop_parser.add_argument(
        "--feedback-weight",
        type=int,
        default=12,
        help="Training oversampling weight for trusted relaxed feedback structures in the generator dataset.",
    )
    loop_parser.add_argument(
        "--min-train-rows",
        type=int,
        default=10000,
        help="Minimum DiffCSP train-row count after weighting and repetition.",
    )
    loop_parser.add_argument("--generator-warm-start-ckpt", default=None, help="Optional generator checkpoint override.")
    loop_parser.add_argument("--alignn-restart-model-path", default=None, help="Optional ALIGNN restart checkpoint override.")
    loop_parser.add_argument("--batch-tag", default=None, help="Optional explicit 2d-mobility batch tag for this round.")
    loop_parser.add_argument(
        "--feedback-snapshot-id",
        default=None,
        help="Optional explicit feedback snapshot id written into loop metadata.",
    )
    loop_parser.add_argument("--generator-model-id", default=None, help="Optional explicit generator model id.")
    loop_parser.add_argument("--alignn-model-id", default=None, help="Optional explicit ALIGNN model id.")
    loop_parser.add_argument(
        "--publish-dry-run",
        action="store_true",
        help="Build the loop and write manifests without inserting new Mongo documents.",
    )
    loop_parser.add_argument(
        "--skip-downstream-run",
        action="store_true",
        help="Skip the 2d-mobility batch launch after publishing strict90 survivors.",
    )
    loop_parser.add_argument(
        "--downstream-dry-run",
        action="store_true",
        help="Launch 2d-mobility in dry-run mode after publishing round candidates.",
    )

    return parser


def _summary_path_for(batch_result: BatchLaunchResult) -> Path:
    return Path(batch_result.runs_root) / "ingestion_summary.json"


def _stderr_excerpt(stderr: str, *, max_lines: int = 20) -> str:
    lines = [line for line in str(stderr or "").splitlines() if line.strip()]
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[-max_lines:])


def _build_summary(
    *,
    source: RunSource,
    sync_summary: SyncSummary,
    batch_result: BatchLaunchResult,
    dry_run_batch: bool,
    mongo_db: str,
    mongo_collection: str,
) -> dict[str, Any]:
    batch_payload = {
        "batch_tag": batch_result.batch_tag,
        "runs_root": batch_result.runs_root,
        "command": list(batch_result.command),
        "returncode": batch_result.returncode,
        "dry_run_batch": bool(dry_run_batch),
    }
    final_state = dict(batch_result.final_state or {})
    if final_state:
        batch_payload["final_state"] = final_state
        summary_path = dict(final_state.get("batch") or {}).get("summary_path")
        if summary_path:
            batch_payload["summary_path"] = summary_path
    if batch_result.returncode != 0:
        batch_payload["stderr_excerpt"] = _stderr_excerpt(batch_result.stderr)
    return {
        "run_id": source.run_id,
        "source_code": source.source_code,
        "run_link": source.run_link,
        "run_root": source.run_root,
        "top10_dir": source.top10_dir,
        "top10_csv_path": source.top10_csv_path,
        "manifest_path": source.manifest_path,
        "cif_count": len(source.items),
        "source_items": [
            {
                "rank": item.rank,
                "material_id": item.material_id,
                "cif_name": item.cif_name,
            }
            for item in source.items
        ],
        "ingestion": {
            "mongo_db": mongo_db,
            "mongo_collection": mongo_collection,
            "upserted_count": sync_summary.upserted_count,
            "skipped_count": sync_summary.skipped_count,
            "force_reset_count": sync_summary.force_reset_count,
            "selected_material_ids": list(sync_summary.selected_material_ids),
        },
        "batch": batch_payload,
    }


def _write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def run_command(args: argparse.Namespace) -> int:
    source = resolve_run_source(getattr(args, "source_run_link", DEFAULT_SOURCE_RUN_LINK))
    mongo_db = getattr(args, "mongo_db", MONGO_DB_NAME)
    mongo_collection = getattr(args, "mongo_collection", MONGO_COLLECTION_NAME)
    expected_cif_count = getattr(args, "expected_cif_count", None)
    force = bool(getattr(args, "force", False))
    dry_run_batch = bool(getattr(args, "dry_run_batch", False))
    batch_tag = getattr(args, "batch_tag", None)

    preflight = run_preflight(
        source=source,
        two_d_mobility_root=DEFAULT_2D_MOBILITY_ROOT,
        expected_cif_count=expected_cif_count,
    )
    with open_collection(
        preflight.effective_env["MONGO_URI"],
        db_name=mongo_db,
        collection_name=mongo_collection,
    ) as collection:
        sync_summary = sync_source_items(collection, source, force=force)

    batch_result = launch_batch(
        two_d_mobility_root=preflight.two_d_mobility_root,
        base_env=preflight.effective_env,
        run_id=source.run_id,
        source_code=source.source_code,
        mongo_db=mongo_db,
        mongo_collection=mongo_collection,
        batch_tag=batch_tag,
        force=force,
        dry_run_batch=dry_run_batch,
        runs_root_base=PROJECT_ROOT / "runs",
    )
    summary = _build_summary(
        source=source,
        sync_summary=sync_summary,
        batch_result=batch_result,
        dry_run_batch=dry_run_batch,
        mongo_db=mongo_db,
        mongo_collection=mongo_collection,
    )
    _write_summary(_summary_path_for(batch_result), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if batch_result.returncode == 0 else int(batch_result.returncode)


def closed_loop_round_command(args: argparse.Namespace) -> int:
    payload = run_closed_loop_round(
        round_index=int(args.round_index),
        round_id=getattr(args, "round_id", None),
        parent_round_id=getattr(args, "parent_round_id", None),
        feedback_source_round_id=getattr(args, "feedback_source_round_id", None),
        feedback_batch_root=str(args.feedback_batch_root),
        invdes_root=str(getattr(args, "invdes_root", DEFAULT_INVDES_ROOT)),
        two_d_mobility_root=str(getattr(args, "two_d_mobility_root", DEFAULT_2D_MOBILITY_ROOT)),
        mongo_uri=getattr(args, "mongo_uri", None),
        mongo_db=str(getattr(args, "mongo_db", MONGO_DB_NAME)),
        mongo_collection=str(getattr(args, "mongo_collection", MONGO_COLLECTION_NAME)),
        total_samples=int(getattr(args, "total_samples", 100000)),
        samples_per_job=int(getattr(args, "samples_per_job", 1000)),
        num_batches_to_samples=int(getattr(args, "num_batches_to_samples", 1)),
        top_k=int(getattr(args, "top_k", 10)),
        gpu_list=str(getattr(args, "gpu_list", "0,1,2,3")),
        phononbench_gpu_list=getattr(args, "phononbench_gpu_list", None),
        phononbench_dim=str(getattr(args, "phononbench_dim", "2 2 2")),
        phononbench_subparts_per_gpu=int(getattr(args, "phononbench_subparts_per_gpu", 1)),
        phononbench_model=str(getattr(args, "phononbench_model", "mattersim-v1")),
        bandgap_threshold=float(getattr(args, "bandgap_threshold", 0.2)),
        formation_threshold=float(getattr(args, "formation_threshold", 0.0)),
        phonon_imag_threshold=float(getattr(args, "phonon_imag_threshold", 0.1)),
        target_crystal_system=str(getattr(args, "target_crystal_system", "orthorhombic")),
        strict90_max_angle_deviation_deg=float(getattr(args, "strict90_max_angle_deviation_deg", 0.6)),
        feedback_weight=int(getattr(args, "feedback_weight", 12)),
        min_train_rows=int(getattr(args, "min_train_rows", 10000)),
        generator_warm_start_ckpt=getattr(args, "generator_warm_start_ckpt", None),
        alignn_restart_model_path=getattr(args, "alignn_restart_model_path", None),
        batch_tag=getattr(args, "batch_tag", None),
        feedback_snapshot_id=getattr(args, "feedback_snapshot_id", None),
        generator_model_id=getattr(args, "generator_model_id", None),
        alignn_model_id=getattr(args, "alignn_model_id", None),
        publish_dry_run=bool(getattr(args, "publish_dry_run", False)),
        skip_downstream_run=bool(getattr(args, "skip_downstream_run", False)),
        downstream_dry_run=bool(getattr(args, "downstream_dry_run", False)),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "run":
        handler = run_command
    elif args.command == "closed-loop-round":
        handler = closed_loop_round_command
    else:
        parser.print_help(sys.stderr)
        return 1
    try:
        return handler(args)
    except (FileNotFoundError, ValueError, RuntimeError, PreflightError, ClosedLoopError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
