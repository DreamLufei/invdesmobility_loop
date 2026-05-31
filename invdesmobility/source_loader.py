from __future__ import annotations

import csv
import json
import os
import re

from pathlib import Path
from typing import Any

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from .models import RunSource, SourceItem


_RANKED_CIF_RE = re.compile(r"^rank_(\d+)__(.+\.cif)$")
_DATE_RE = re.compile(r"(\d{8})")
_PART_SAMPLE_RE = re.compile(r"part0*(\d+)__sample_(\d+)", re.IGNORECASE)


def _safe_float(raw: str | None) -> float | None:
    text = str(raw or "").strip()
    if not text:
        return None
    return float(text)


def _contains_ranked_cifs(path: Path) -> bool:
    return path.is_dir() and any(path.glob("rank_*.cif"))


def _find_run_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "manifest.json").exists():
            return candidate
    raise FileNotFoundError(f"Could not locate manifest.json above {start}")


def _find_ranked_cif_dir(run_root: Path) -> Path:
    candidates = (
        run_root / "10_top10_strict90" / "strict90_cif",
        run_root / "07_top10_cif" / "strict90_cif",
        run_root / "07_top10_cif",
        run_root / "06_top10_cif",
    )
    for candidate in candidates:
        if _contains_ranked_cifs(candidate):
            return candidate
    raise FileNotFoundError(f"Could not locate a ranked CIF directory under {run_root}")


def _resolve_run_paths(source_run_link: str | os.PathLike[str]) -> tuple[str, Path, Path]:
    raw_path = Path(source_run_link).expanduser()
    raw_abs = Path(os.path.abspath(str(raw_path)))
    resolved = raw_abs.resolve()

    if _contains_ranked_cifs(resolved):
        top10_dir = resolved
        run_root = _find_run_root(top10_dir)
        return str(raw_abs), run_root, top10_dir

    run_root = resolved
    top10_dir = _find_ranked_cif_dir(run_root)
    return str(raw_abs), run_root, top10_dir


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    with manifest_path.open("r", encoding="utf-8") as handle:
        return dict(json.load(handle) or {})


def _load_csv_rows(top10_csv_path: Path) -> dict[tuple[int, str], dict[str, str]]:
    with top10_csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    indexed: dict[tuple[int, str], dict[str, str]] = {}
    for row in rows:
        rank = int(str(row.get("rank") or "").strip())
        cif_name = str(row.get("cif_name") or "").strip()
        indexed[(rank, cif_name)] = dict(row)
    return indexed


def _find_top10_csv(top10_dir: Path, run_root: Path) -> Path:
    candidates = (
        top10_dir / "top10_candidates.csv",
        top10_dir / "top10_candidates_strict90.csv",
        top10_dir.parent / "top10_candidates_strict90.csv",
        top10_dir.parent / "top10_candidates.csv",
        run_root / "10_top10_strict90" / "top10_candidates_strict90.csv",
        run_root / "top10_candidates.csv",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing top10_candidates.csv near {top10_dir}")


def _date_tag_for(run_id: str) -> str:
    match = _DATE_RE.search(run_id)
    if match:
        return match.group(1)[2:]
    return "nodate"


def _selection_tag_for(top10_dir: Path) -> str:
    joined = "/".join((top10_dir.parent.name.lower(), top10_dir.name.lower()))
    if "strict90" in joined:
        return "s90"
    if top10_dir.name.lower() in {"06_top10_cif", "07_top10_cif"}:
        return "t10"
    return "src"


def _run_tag_for(run_id: str) -> str:
    lowered = run_id.lower()
    if "orthorhombic" in lowered:
        return "orth"
    if "semiconductor" in lowered:
        return "semi"
    return "run"


def _source_code_for(run_id: str, top10_dir: Path) -> str:
    return f"{_date_tag_for(run_id)}_{_run_tag_for(run_id)}_{_selection_tag_for(top10_dir)}"


def _cif_suffix_for(cif_name: str) -> str:
    match = _PART_SAMPLE_RE.search(cif_name)
    if match:
        return f"p{int(match.group(1)):02d}_s{int(match.group(2)):04d}"
    stem = Path(cif_name).stem.lower()
    sanitized = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return sanitized[-16:] or "sample"


def _build_source_item(
    *,
    run_id: str,
    source_code: str,
    cif_path: Path,
    metadata_row: dict[str, str] | None,
) -> SourceItem:
    match = _RANKED_CIF_RE.match(cif_path.name)
    if not match:
        raise ValueError(f"Unsupported ranked CIF filename: {cif_path.name}")
    rank_token = match.group(1)
    cif_name = match.group(2)
    rank = int(rank_token)

    structure = Structure.from_file(cif_path)
    analyzer = SpacegroupAnalyzer(structure, symprec=0.1)
    composition = structure.composition
    hall_getter = getattr(analyzer, "get_hall", None)
    hall_symbol = str(hall_getter() if callable(hall_getter) else "" or "")
    material_id = f"inv_{source_code}_r{rank_token}_{_cif_suffix_for(cif_name)}"
    source_key = f"{run_id}::rank_{rank_token}::{cif_name}"
    row = dict(metadata_row or {})
    return SourceItem(
        rank=rank,
        rank_token=rank_token,
        cif_name=cif_name,
        cif_path=str(cif_path),
        material_id=material_id,
        source_key=source_key,
        formula_pretty=composition.reduced_formula,
        formula_reduced=composition.reduced_formula,
        formula_reduced_abc=composition.reduced_composition.formula,
        spacegroup_symbol=analyzer.get_space_group_symbol(),
        spacegroup_number=int(analyzer.get_space_group_number()),
        spacegroup_point_group=analyzer.get_point_group_symbol(),
        spacegroup_crystal_system=analyzer.get_crystal_system(),
        spacegroup_hall=hall_symbol,
        structure_dict=structure.as_dict(),
        bandgap_ev=_safe_float(row.get("bandgap")),
        formation_energy_ev_per_atom=_safe_float(row.get("formation_energy")),
        alignn_mobility_score=_safe_float(row.get("mobility_score")),
    )


def resolve_run_source(source_run_link: str | os.PathLike[str]) -> RunSource:
    run_link, run_root, top10_dir = _resolve_run_paths(source_run_link)
    if not run_root.exists():
        raise FileNotFoundError(f"Resolved run root does not exist: {run_root}")
    if not top10_dir.exists():
        raise FileNotFoundError(f"Top10 CIF directory does not exist: {top10_dir}")

    manifest_path = run_root / "manifest.json"
    top10_csv_path = _find_top10_csv(top10_dir, run_root)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest.json: {manifest_path}")

    manifest = _load_manifest(manifest_path)
    csv_rows = _load_csv_rows(top10_csv_path)
    run_id = run_root.name
    source_code = _source_code_for(run_id, top10_dir)
    items: list[SourceItem] = []
    for cif_path in sorted(top10_dir.glob("rank_*.cif")):
        match = _RANKED_CIF_RE.match(cif_path.name)
        if not match:
            continue
        rank = int(match.group(1))
        cif_name = match.group(2)
        row = csv_rows.get((rank, cif_name))
        items.append(
            _build_source_item(
                run_id=run_id,
                source_code=source_code,
                cif_path=cif_path,
                metadata_row=row,
            )
        )

    if not items:
        raise RuntimeError(f"No ranked CIF files found under {top10_dir}")

    return RunSource(
        run_link=run_link,
        run_root=str(run_root),
        run_id=run_id,
        source_code=source_code,
        top10_dir=str(top10_dir),
        top10_csv_path=str(top10_csv_path),
        manifest_path=str(manifest_path),
        manifest=manifest,
        items=tuple(items),
    )
