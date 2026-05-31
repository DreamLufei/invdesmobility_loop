from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys

from pathlib import Path
from typing import Any, Callable, Sequence

from pymatgen.core import Structure

from . import DEFAULT_2D_MOBILITY_ROOT, DEFAULT_INVDES_ROOT, PROJECT_ROOT
from .batch_launcher import launch_batch
from .preflight import load_effective_env


class ClosedLoopError(RuntimeError):
    pass


CommandRunner = Callable[[Sequence[str], str | None, dict[str, str] | None], Any]


def format_round_suffix(round_index: int) -> str:
    if int(round_index) < 0:
        raise ClosedLoopError(f"round_index must be >= 0, got {round_index}")
    return f"{int(round_index):02d}"


def default_round_id(round_index: int) -> str:
    return f"loop_{format_round_suffix(round_index)}"


def default_parent_round_id(round_index: int) -> str:
    if int(round_index) <= 1:
        return "round_00_bootstrap"
    return default_round_id(int(round_index) - 1)


def _sanitize_token(raw: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(raw or "")).strip("_")
    return token or "item"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ClosedLoopError(f"expected JSON object at {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _assert_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise ClosedLoopError(f"missing {label}: {path}")


def _prepare_round_root(round_root: Path) -> None:
    if not round_root.exists():
        round_root.mkdir(parents=True, exist_ok=False)
        return
    if not round_root.is_dir():
        raise ClosedLoopError(f"target round root is not a directory: {round_root}")
    allowed_names = {"_control"}
    existing_names = {child.name for child in round_root.iterdir()}
    disallowed = sorted(existing_names - allowed_names)
    if disallowed:
        raise ClosedLoopError(
            f"target round root already exists with non-control contents: {round_root} ({', '.join(disallowed)})"
        )


def _clear_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _recursive_cif_paths(root: Path) -> list[Path]:
    seen: set[Path] = set()
    paths: list[Path] = []
    for pattern in ("*.cif", "*.CIF"):
        for path in root.rglob(pattern):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            paths.append(path)
    return sorted(paths)


def _copy_feedback_archive(source_dir: Path, target_dir: Path) -> None:
    _clear_dir(target_dir)
    for name in ("trusted_channels.csv", "trusted_materials.csv", "rejected_feedback.csv", "feedback_summary.json"):
        src = source_dir / name
        if src.exists():
            shutil.copy2(src, target_dir / name)
    source_cif_dir = source_dir / "trusted_relaxed_cif"
    target_cif_dir = target_dir / "trusted_relaxed_cif"
    target_cif_dir.mkdir(parents=True, exist_ok=True)
    if source_cif_dir.exists():
        for cif_path in sorted(source_cif_dir.glob("*.cif")):
            shutil.copy2(cif_path, target_cif_dir / cif_path.name)


def _archive_batch_submitted_structures(batch_root: Path, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for child in list(output_dir.iterdir()):
        if child.is_file() or child.is_symlink():
            child.unlink()

    written: list[Path] = []
    for material_dir in sorted(path for path in batch_root.iterdir() if path.is_dir()):
        poscar_path = material_dir / "POSCAR"
        if not poscar_path.exists():
            continue
        structure = Structure.from_file(str(poscar_path))
        output_path = output_dir / f"{_sanitize_token(material_dir.name)}.cif"
        structure.to(fmt="cif", filename=str(output_path))
        written.append(output_path)
    return written


def _iter_feedback_material_csvs(feedback_archive_root: Path) -> list[Path]:
    return sorted(feedback_archive_root.glob("*/trusted_materials.csv"))


def _iter_feedback_relaxed_cifs(feedback_archive_root: Path) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for csv_path in _iter_feedback_material_csvs(feedback_archive_root):
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                candidate = Path(str(row.get("relaxed_cif_path") or "").strip())
                if not candidate.exists():
                    continue
                resolved = candidate.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                paths.append(candidate)
    return sorted(paths)


def _iter_submitted_reference_cifs(feedback_archive_root: Path) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for cif_path in sorted(feedback_archive_root.glob("*/submitted_reference_cif/*.cif")):
        resolved = cif_path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        paths.append(cif_path)
    return paths


def build_combined_reference_dir(
    *,
    base_reference_root: Path,
    feedback_archive_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    _assert_exists(base_reference_root, "base reference root")
    _clear_dir(output_dir)

    linked_targets: set[Path] = set()
    base_count = 0
    historical_count = 0
    feedback_count = 0

    def link_many(paths: Sequence[Path], prefix: str) -> int:
        linked = 0
        for idx, source_path in enumerate(sorted(paths), start=1):
            resolved = source_path.resolve()
            if resolved in linked_targets:
                continue
            linked_targets.add(resolved)
            suffix = source_path.suffix.lower() or ".cif"
            stem = _sanitize_token(source_path.stem)
            dst = output_dir / f"{prefix}_{idx:05d}__{stem}{suffix}"
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            dst.symlink_to(resolved)
            linked += 1
        return linked

    base_count += link_many(_recursive_cif_paths(base_reference_root), "base")
    historical_count += link_many(_iter_submitted_reference_cifs(feedback_archive_root), "history")
    feedback_count += link_many(_iter_feedback_relaxed_cifs(feedback_archive_root), "feedback")

    manifest = {
        "output_dir": str(output_dir),
        "base_reference_root": str(base_reference_root),
        "feedback_archive_root": str(feedback_archive_root),
        "counts": {
            "base_reference_count": base_count,
            "historical_computed_reference_count": historical_count,
            "feedback_reference_count": feedback_count,
            "total_reference_count": base_count + historical_count + feedback_count,
        },
    }
    _write_json(output_dir / "reference_manifest.json", manifest)
    return manifest


def _run_checked(
    command: Sequence[str | os.PathLike[str]],
    *,
    cwd: str | os.PathLike[str] | None = None,
    env: dict[str, str] | None = None,
    runner: CommandRunner | None = None,
) -> Any:
    normalized_command = tuple(str(part) for part in command)
    normalized_cwd = str(cwd) if cwd is not None else None
    merged_env = dict(os.environ)
    if env:
        merged_env.update({str(key): str(value) for key, value in env.items()})

    if runner is not None:
        return runner(normalized_command, normalized_cwd, merged_env)

    completed = subprocess.run(
        list(normalized_command),
        cwd=normalized_cwd,
        env=merged_env,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ClosedLoopError(
            f"command failed with returncode={completed.returncode}: {' '.join(normalized_command)}"
        )
    return completed


def _invdes_layout(invdes_root: Path, round_suffix: str, generator_model_id: str, alignn_model_id: str) -> dict[str, Path]:
    datasets_root = invdes_root / "03_datasets"
    models_root = invdes_root / "04_models"
    runs_root = invdes_root / "06_runs"
    step09_root = invdes_root / "05_steps" / "09_closed_loop_feedback"
    return {
        "invdes_root": invdes_root,
        "datasets_root": datasets_root,
        "models_root": models_root,
        "runs_root": runs_root,
        "source_cif_dir": datasets_root / "01_source_cif" / "high_quality_280",
        "source_cif_reference_root": datasets_root / "01_source_cif",
        "extract_feedback_script": step09_root / "extract_trusted_feedback.py",
        "build_alignn_dataset_script": step09_root / "build_alignn_round_dataset.py",
        "build_diffcsp_dataset_script": step09_root / "build_diffcsp_round_dataset.py",
        "publish_round_candidates_script": step09_root / "publish_round_candidates_to_mongo.py",
        "alignn_train_script": invdes_root / "05_steps" / "06_alignn_mobility_rank" / "train_best.sh",
        "generator_train_script": invdes_root / "05_steps" / "02_finetune_generator" / "run.sh",
        "generate_script": invdes_root / "05_steps" / "03_generate_structures" / "run_multigpu.sh",
        "screening_script": invdes_root / "05_steps" / "08_orchestration" / "run_dedup_orthorhombic_semiconductor_pipeline.sh",
        "alignn_round_dataset_dir": datasets_root / "03_alignn_mobility_dataset" / f"round_{round_suffix}",
        "alignn_round_dir": models_root / "04_alignn_mobility" / alignn_model_id,
        "alignn_base_model_ckpt": models_root / "04_alignn_mobility" / "mobility_reg_v1_bs8_lr5e4_wu200_nw4" / "best_model.pt",
        "alignn_base_model_config": models_root / "04_alignn_mobility" / "mobility_reg_v1_bs8_lr5e4_wu200_nw4" / "config.json",
        "diffcsp_round_dataset_name": f"mobility2d_feedback_round_{round_suffix}",
        "generator_round_dir": models_root / "01_diffcsp_generator" / "finetuned" / generator_model_id,
        "generator_base_ckpt": models_root / "01_diffcsp_generator" / "finetuned" / "mobility2d_highquality280_ft_v1" / "best.ckpt",
    }


def _default_generator_warm_start(layout: dict[str, Path], round_index: int) -> Path:
    if int(round_index) > 1:
        previous = layout["models_root"] / "01_diffcsp_generator" / "finetuned" / f"generator_round_{format_round_suffix(int(round_index) - 1)}" / "best.ckpt"
        if previous.exists():
            return previous
    return layout["generator_base_ckpt"]


def _default_alignn_restart(layout: dict[str, Path], round_index: int) -> Path:
    if int(round_index) > 1:
        previous_dir = layout["models_root"] / "04_alignn_mobility" / f"alignn_mobility_round_{format_round_suffix(int(round_index) - 1)}"
        for candidate_name in ("current_model.pt", "best_model.pt"):
            candidate = previous_dir / candidate_name
            if candidate.exists():
                return candidate
    base_dir = layout["alignn_base_model_ckpt"].parent
    current_model = base_dir / "current_model.pt"
    if current_model.exists():
        return current_model
    return layout["alignn_base_model_ckpt"]


def _find_latest_strict90_csv(screening_run_root: Path) -> Path:
    candidates = sorted(screening_run_root.glob("*/**/*_candidates_strict90.csv"))
    if not candidates:
        candidates = sorted(screening_run_root.rglob("*_candidates_strict90.csv"))
    if not candidates:
        raise ClosedLoopError(f"missing strict90 merged CSV under {screening_run_root}")
    return candidates[-1]


def run_closed_loop_round(
    *,
    round_index: int,
    feedback_batch_root: str | os.PathLike[str],
    round_id: str | None = None,
    parent_round_id: str | None = None,
    feedback_source_round_id: str | None = None,
    invdes_root: str | os.PathLike[str] = DEFAULT_INVDES_ROOT,
    two_d_mobility_root: str | os.PathLike[str] = DEFAULT_2D_MOBILITY_ROOT,
    mongo_db: str = "materials_database",
    mongo_collection: str = "invdesmobility",
    mongo_uri: str | None = None,
    total_samples: int = 100000,
    samples_per_job: int = 1000,
    num_batches_to_samples: int = 1,
    top_k: int = 10,
    gpu_list: str = "0,1,2,3",
    phononbench_gpu_list: str | None = None,
    phononbench_dim: str = "2 2 2",
    phononbench_subparts_per_gpu: int = 1,
    phononbench_model: str = "mattersim-v1",
    bandgap_threshold: float = 0.2,
    formation_threshold: float = 0.0,
    phonon_imag_threshold: float = 0.1,
    target_crystal_system: str = "orthorhombic",
    strict90_max_angle_deviation_deg: float = 0.6,
    feedback_weight: int = 12,
    min_train_rows: int = 10000,
    generator_warm_start_ckpt: str | os.PathLike[str] | None = None,
    alignn_restart_model_path: str | os.PathLike[str] | None = None,
    publish_dry_run: bool = False,
    skip_downstream_run: bool = False,
    downstream_dry_run: bool = False,
    batch_tag: str | None = None,
    feedback_snapshot_id: str | None = None,
    generator_model_id: str | None = None,
    alignn_model_id: str | None = None,
    command_runner: CommandRunner | None = None,
) -> dict[str, Any]:
    normalized_round_index = int(round_index)
    normalized_round_id = str(round_id or default_round_id(normalized_round_index)).strip()
    normalized_parent_round_id = str(parent_round_id or default_parent_round_id(normalized_round_index)).strip()
    normalized_feedback_source_round_id = str(feedback_source_round_id or normalized_parent_round_id).strip()
    round_suffix = format_round_suffix(normalized_round_index)
    normalized_generator_model_id = str(generator_model_id or f"generator_round_{round_suffix}").strip()
    normalized_alignn_model_id = str(alignn_model_id or f"alignn_mobility_round_{round_suffix}").strip()
    normalized_feedback_snapshot_id = str(
        feedback_snapshot_id or f"feedback_snapshot_up_to_{normalized_feedback_source_round_id}"
    ).strip()
    normalized_batch_tag = str(batch_tag or f"{normalized_round_id}__mobility_batch").strip()

    normalized_invdes_root = Path(invdes_root).expanduser().resolve()
    normalized_two_d_root = Path(two_d_mobility_root).expanduser().resolve()
    normalized_feedback_batch_root = Path(feedback_batch_root).expanduser().resolve()
    _assert_exists(normalized_feedback_batch_root, "feedback batch root")
    _assert_exists(normalized_invdes_root, "invDesMobility root")
    _assert_exists(normalized_two_d_root, "2d-mobility root")

    layout = _invdes_layout(
        normalized_invdes_root,
        round_suffix,
        normalized_generator_model_id,
        normalized_alignn_model_id,
    )
    for label in (
        "extract_feedback_script",
        "build_alignn_dataset_script",
        "build_diffcsp_dataset_script",
        "publish_round_candidates_script",
        "alignn_train_script",
        "generator_train_script",
        "generate_script",
        "screening_script",
        "source_cif_dir",
        "source_cif_reference_root",
        "alignn_base_model_config",
    ):
        _assert_exists(layout[label], label)

    two_d_env = load_effective_env(normalized_two_d_root)
    effective_mongo_uri = str(mongo_uri or two_d_env.get("MONGO_URI") or "").strip()
    if not effective_mongo_uri and publish_dry_run:
        effective_mongo_uri = "mongodb://dry-run.invalid"
    if not effective_mongo_uri:
        raise ClosedLoopError("MONGO_URI is required for closed-loop publishing")

    round_root = PROJECT_ROOT / "runs" / f"{normalized_round_id}__closed_loop_round"
    _prepare_round_root(round_root)

    feedback_archive_root = PROJECT_ROOT / "runs" / "_feedback_archive"
    feedback_archive_dir = feedback_archive_root / normalized_feedback_source_round_id
    submitted_reference_archive_dir = feedback_archive_dir / "submitted_reference_cif"
    control_dir = round_root / "_control"
    alignn_dataset_dir = round_root / "03_alignn_dataset"
    diffcsp_dataset_dir = round_root / "04_diffcsp_dataset"
    generator_training_dir = round_root / "05_generator_training"
    alignn_training_dir = round_root / "06_alignn_training"
    publish_dir = round_root / "07_publish"
    control_dir.mkdir(parents=True, exist_ok=True)
    alignn_dataset_dir.mkdir(parents=True, exist_ok=True)
    diffcsp_dataset_dir.mkdir(parents=True, exist_ok=True)
    generator_training_dir.mkdir(parents=True, exist_ok=True)
    alignn_training_dir.mkdir(parents=True, exist_ok=True)
    publish_dir.mkdir(parents=True, exist_ok=True)
    feedback_dir = round_root / "01_trusted_feedback"
    reference_pool_dir = round_root / "02_reference_pool_for_dedup"
    alignn_dataset_manifest = alignn_dataset_dir / "manifest.json"
    diffcsp_dataset_manifest = diffcsp_dataset_dir / "manifest.json"
    generator_ckpt_report = generator_training_dir / "ckpt_compat.json"
    generator_filtered_ckpt = generator_training_dir / "filtered.ckpt"
    generator_train_log = generator_training_dir / "train.log"
    alignn_prepare_log = alignn_training_dir / "prepare.log"
    alignn_patch_log = alignn_training_dir / "patch.log"
    alignn_train_log = alignn_training_dir / "train.log"
    publish_manifest_path = publish_dir / "manifest.json"
    downstream_runs_root = round_root / "09_2d_mobility_batch"
    loop_manifest_path = round_root / "loop_manifest.json"

    normalized_generator_warm_start = Path(
        generator_warm_start_ckpt or _default_generator_warm_start(layout, normalized_round_index)
    ).expanduser().resolve()
    normalized_alignn_restart = Path(
        alignn_restart_model_path or _default_alignn_restart(layout, normalized_round_index)
    ).expanduser().resolve()
    _assert_exists(normalized_generator_warm_start, "generator warm-start checkpoint")
    _assert_exists(normalized_alignn_restart, "ALIGNN restart checkpoint")

    _run_checked(
        [
            sys.executable,
            str(layout["extract_feedback_script"]),
            "--batch-root",
            str(normalized_feedback_batch_root),
            "--output-dir",
            str(feedback_dir),
            "--round-id",
            normalized_feedback_source_round_id,
            "--batch-id",
            normalized_feedback_batch_root.name,
        ],
        cwd=PROJECT_ROOT,
        runner=command_runner,
    )
    _copy_feedback_archive(feedback_dir, feedback_archive_dir)
    _archive_batch_submitted_structures(normalized_feedback_batch_root, submitted_reference_archive_dir)

    feedback_csvs = _iter_feedback_material_csvs(feedback_archive_root)
    if not feedback_csvs:
        raise ClosedLoopError(f"no trusted feedback CSVs found under {feedback_archive_root}")

    reference_manifest = build_combined_reference_dir(
        base_reference_root=layout["source_cif_reference_root"],
        feedback_archive_root=feedback_archive_root,
        output_dir=reference_pool_dir,
    )

    alignn_dataset_command = [
        sys.executable,
        str(layout["build_alignn_dataset_script"]),
        "--output-dir",
        str(layout["alignn_round_dataset_dir"]),
        "--manifest-path",
        str(alignn_dataset_manifest),
        "--base-cif-dir",
        str(layout["source_cif_dir"]),
        "--base-labels",
        str(layout["source_cif_dir"] / "id_prop.csv"),
    ]
    for feedback_csv in feedback_csvs:
        alignn_dataset_command.extend(["--feedback-csv", str(feedback_csv)])
    _run_checked(alignn_dataset_command, cwd=PROJECT_ROOT, runner=command_runner)

    _run_checked(
        ["bash", str(layout["alignn_train_script"])],
        cwd=normalized_invdes_root,
        env={
            "ALIGNN_MOBILITY_SKIP_PREPARE": "1",
            "ALIGNN_MOBILITY_DATA_DIR_OVERRIDE": str(layout["alignn_round_dataset_dir"]),
            "ALIGNN_MOBILITY_OUTPUT_DIR": str(layout["alignn_round_dir"]),
            "ALIGNN_MOBILITY_CONFIG_OVERRIDE": str(layout["alignn_base_model_config"]),
            "ALIGNN_MOBILITY_RESTART_MODEL_PATH": str(normalized_alignn_restart),
            "ALIGNN_MOBILITY_PREPARE_LOG_PATH": str(alignn_prepare_log),
            "ALIGNN_MOBILITY_PATCH_LOG_PATH": str(alignn_patch_log),
            "ALIGNN_MOBILITY_TRAIN_LOG_PATH": str(alignn_train_log),
        },
        runner=command_runner,
    )
    _assert_exists(layout["alignn_round_dir"] / "best_model.pt", "round ALIGNN checkpoint")

    diffcsp_dataset_command = [
        sys.executable,
        str(layout["build_diffcsp_dataset_script"]),
        "--dataset-name",
        str(layout["diffcsp_round_dataset_name"]),
        "--manifest-path",
        str(diffcsp_dataset_manifest),
        "--base-cif-dir",
        str(layout["source_cif_dir"]),
        "--feedback-weight",
        str(int(feedback_weight)),
        "--min-train-rows",
        str(int(min_train_rows)),
    ]
    for feedback_csv in feedback_csvs:
        diffcsp_dataset_command.extend(["--feedback-csv", str(feedback_csv)])
    _run_checked(diffcsp_dataset_command, cwd=PROJECT_ROOT, runner=command_runner)

    _run_checked(
        ["bash", str(layout["generator_train_script"])],
        cwd=normalized_invdes_root,
        env={
            "DATASET_NAME": str(layout["diffcsp_round_dataset_name"]),
            "CKPT_PATH": str(normalized_generator_warm_start),
            "EXP_NAME": normalized_generator_model_id,
            "FINETUNED_DIR": str(layout["generator_round_dir"]),
            "REPORT_PATH": str(generator_ckpt_report),
            "FILTERED_CKPT_PATH": str(generator_filtered_ckpt),
            "LOG_PATH": str(generator_train_log),
        },
        runner=command_runner,
    )
    _assert_exists(layout["generator_round_dir"] / "best.ckpt", "round generator checkpoint")

    generation_run_id = f"{normalized_round_id}__generated_{int(total_samples)}_structures__from_{normalized_generator_model_id}"
    _run_checked(
        ["bash", str(layout["generate_script"])],
        cwd=normalized_invdes_root,
        env={
            "MODEL_PATH": str(layout["generator_round_dir"]),
            "RUN_ID": generation_run_id,
            "TOTAL_SAMPLES": str(int(total_samples)),
            "SAMPLES_PER_JOB": str(int(samples_per_job)),
            "NUM_BATCHES_TO_SAMPLES": str(int(num_batches_to_samples)),
            "GPU_LIST": str(gpu_list),
            "LABEL_PREFIX": normalized_round_id,
            "CONVERT_TO_CIF": "1",
        },
        runner=command_runner,
    )
    generation_run_root = layout["runs_root"] / generation_run_id
    generation_cif_dir = generation_run_root / "03_generate_structures" / "generated_cif"
    _assert_exists(generation_cif_dir, "generated CIF dir")

    screening_run_id = f"{normalized_round_id}__screen_from_{generation_run_id}"
    _run_checked(
        ["bash", str(layout["screening_script"])],
        cwd=normalized_invdes_root,
        env={
            "INPUT_SOURCE_DIR": str(generation_cif_dir),
            "SOURCE_RUN_LABEL": generation_run_id,
            "SOURCE_RUN_ID": generation_run_id,
            "RUN_NAME": screening_run_id,
            "TARGET_CRYSTAL_SYSTEM": str(target_crystal_system),
            "DEDUP_MODE": "formula",
            "DEDUP_REFERENCE_DIR": str(reference_pool_dir),
            "DEDUP_BANDGAP_THRESHOLD": str(float(bandgap_threshold)),
            "DEDUP_FORMATION_ENERGY_THRESHOLD": str(float(formation_threshold)),
            "PHONONBENCH_IMAG_THRESHOLD": str(float(phonon_imag_threshold)),
            "PHONONBENCH_DIM": str(phononbench_dim),
            "PHONONBENCH_GPU_LIST": str(phononbench_gpu_list or gpu_list),
            "PHONONBENCH_SUBPARTS_PER_GPU": str(int(phononbench_subparts_per_gpu)),
            "PHONONBENCH_MODEL": str(phononbench_model),
            "TOP_K": str(int(top_k)),
            "STRICT90_MAX_ANGLE_DEVIATION_DEG": str(float(strict90_max_angle_deviation_deg)),
            "ALIGNN_MOBILITY_MODEL_CONFIG_OVERRIDE": str(layout["alignn_round_dir"] / "config.json"),
            "ALIGNN_MOBILITY_MODEL_CKPT_OVERRIDE": str(layout["alignn_round_dir"] / "best_model.pt"),
        },
        runner=command_runner,
    )
    screening_run_root = layout["runs_root"] / screening_run_id
    _assert_exists(screening_run_root, "screening run root")
    strict90_csv = _find_latest_strict90_csv(screening_run_root)

    publish_command = [
        sys.executable,
        str(layout["publish_round_candidates_script"]),
        "--strict90-csv",
        str(strict90_csv),
        "--output-manifest",
        str(publish_manifest_path),
        "--mongo-uri",
        effective_mongo_uri,
        "--mongo-db",
        str(mongo_db),
        "--mongo-collection",
        str(mongo_collection),
        "--round-index",
        str(normalized_round_index),
        "--round-id",
        normalized_round_id,
        "--parent-round-id",
        normalized_parent_round_id,
        "--generator-model-id",
        normalized_generator_model_id,
        "--alignn-model-id",
        normalized_alignn_model_id,
        "--feedback-snapshot-id",
        normalized_feedback_snapshot_id,
        "--pipeline-run-id",
        screening_run_id,
    ]
    if publish_dry_run:
        publish_command.append("--dry-run")
    _run_checked(publish_command, cwd=PROJECT_ROOT, runner=command_runner)

    publish_manifest = _load_json(publish_manifest_path)
    publishable_rows = int(publish_manifest.get("counts", {}).get("publishable_rows", 0) or 0)

    batch_result = None
    if publishable_rows > 0 and not skip_downstream_run and not publish_dry_run:
        batch_result = launch_batch(
            two_d_mobility_root=str(normalized_two_d_root),
            base_env=two_d_env,
            run_id=None,
            source_code=normalized_round_id,
            mongo_db=str(mongo_db),
            mongo_collection=str(mongo_collection),
            batch_tag=normalized_batch_tag,
            force=True,
            dry_run_batch=bool(downstream_dry_run),
            runs_root_base=str(downstream_runs_root),
            claim_filter={"loop_metadata.round_id": normalized_round_id},
        )

    loop_manifest = {
        "round_id": normalized_round_id,
        "round_index": normalized_round_index,
        "parent_round_id": normalized_parent_round_id,
        "feedback_source_round_id": normalized_feedback_source_round_id,
        "feedback_snapshot_id": normalized_feedback_snapshot_id,
        "feedback_batch_root": str(normalized_feedback_batch_root),
        "feedback_archive_root": str(feedback_archive_root),
        "mongo": {
            "db": str(mongo_db),
            "collection": str(mongo_collection),
            "publish_dry_run": bool(publish_dry_run),
        },
        "reference_manifest": reference_manifest,
        "generator_model": {
            "model_id": normalized_generator_model_id,
            "model_dir": str(layout["generator_round_dir"]),
            "warm_start_ckpt": str(normalized_generator_warm_start),
        },
        "alignn_model": {
            "model_id": normalized_alignn_model_id,
            "model_dir": str(layout["alignn_round_dir"]),
            "restart_model_path": str(normalized_alignn_restart),
        },
        "generation": {
            "run_id": generation_run_id,
            "run_root": str(generation_run_root),
            "generated_cif_dir": str(generation_cif_dir),
            "total_samples": int(total_samples),
            "samples_per_job": int(samples_per_job),
            "num_batches_to_samples": int(num_batches_to_samples),
            "gpu_list": str(gpu_list),
        },
        "screening": {
            "run_id": screening_run_id,
            "run_root": str(screening_run_root),
            "strict90_csv": str(strict90_csv),
            "top_k": int(top_k),
            "target_crystal_system": str(target_crystal_system),
            "bandgap_threshold": float(bandgap_threshold),
            "formation_threshold": float(formation_threshold),
            "phonon_imag_threshold": float(phonon_imag_threshold),
        },
        "manifests": {
            "feedback_summary": _load_json(feedback_dir / "feedback_summary.json"),
            "alignn_dataset_manifest": _load_json(alignn_dataset_manifest),
            "diffcsp_dataset_manifest": _load_json(diffcsp_dataset_manifest),
            "generation_manifest": _load_json(generation_run_root / "manifest.json"),
            "screening_manifest": _load_json(screening_run_root / "manifest.json"),
            "publish_manifest": publish_manifest,
        },
    }
    if batch_result is not None:
        loop_manifest["downstream_batch"] = {
            "batch_tag": batch_result.batch_tag,
            "runs_root": batch_result.runs_root,
            "command": list(batch_result.command),
            "returncode": int(batch_result.returncode),
            "final_state": dict(batch_result.final_state or {}),
            "dry_run": bool(downstream_dry_run),
        }
    else:
        loop_manifest["downstream_batch"] = {
            "batch_tag": normalized_batch_tag,
            "runs_root": str(downstream_runs_root),
            "skipped": bool(skip_downstream_run or publishable_rows == 0 or publish_dry_run),
            "dry_run": bool(downstream_dry_run),
        }

    _write_json(loop_manifest_path, loop_manifest)
    return loop_manifest
