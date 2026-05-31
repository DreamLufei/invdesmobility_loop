from __future__ import annotations

import copy
import csv
import json

from pathlib import Path
from typing import Any

from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter


def build_sample_run(
    base_dir: Path,
    *,
    item_count: int = 2,
    top10_dir_rel: str = "06_top10_cif",
) -> tuple[Path, Path]:
    run_root = base_dir / "20260409__from_example__bg_gt_0p4eV__eform_lt_0p0eV_atom__mobility_rank"
    top10_dir = run_root / top10_dir_rel
    top10_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = top10_dir if top10_dir.name in {"06_top10_cif", "07_top10_cif"} else top10_dir.parent
    csv_name = "top10_candidates.csv"
    if top10_dir.name == "strict90_cif" and top10_dir.parent.name == "10_top10_strict90":
        csv_name = "top10_candidates_strict90.csv"

    manifest = {
        "run_id": run_root.name,
        "top_k": item_count,
    }
    (run_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    structure = Structure(
        lattice=Lattice.tetragonal(3.0, 20.0),
        species=["Si", "Si"],
        coords=[[0.0, 0.0, 0.5], [0.5, 0.5, 0.5]],
    )

    csv_rows: list[dict[str, Any]] = []
    for idx in range(1, item_count + 1):
        rank_token = f"{idx:02d}"
        cif_name = f"sample_{idx:04d}.cif"
        ranked_name = f"rank_{rank_token}__{cif_name}"
        ranked_path = top10_dir / ranked_name
        CifWriter(structure).write_file(str(ranked_path))
        csv_rows.append(
            {
                "rank": idx,
                "cif_name": cif_name,
                "cif_path": str(run_root / "01_input_generated_cif" / cif_name),
                "bandgap": f"{1.0 + 0.1 * idx:.4f}",
                "is_nonmetal": "True",
                "formation_energy": f"{-0.1 * idx:.4f}",
                "passes_formation_filter": "True",
                "mobility_score": f"{3.0 + 0.5 * idx:.4f}",
                "copied_cif": str(csv_dir / ranked_name),
            }
        )

    with (csv_dir / csv_name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "cif_name",
                "cif_path",
                "bandgap",
                "is_nonmetal",
                "formation_energy",
                "passes_formation_filter",
                "mobility_score",
                "copied_cif",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    link_path = base_dir / "current__latest_default_semiconductor_pipeline"
    link_path.symlink_to(run_root, target_is_directory=True)
    return link_path, run_root


def _set_dotted(document: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    current = document
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def _unset_dotted(document: dict[str, Any], dotted_key: str) -> None:
    parts = dotted_key.split(".")
    current = document
    for part in parts[:-1]:
        next_value = current.get(part)
        if not isinstance(next_value, dict):
            return
        current = next_value
    current.pop(parts[-1], None)


class FakeCollection:
    def __init__(self) -> None:
        self.docs: dict[str, dict[str, Any]] = {}
        self.index_calls: list[tuple[Any, dict[str, Any]]] = []

    def create_index(self, keys: Any, **kwargs: Any) -> None:
        self.index_calls.append((keys, dict(kwargs)))

    def find_one(self, query: dict[str, Any]) -> dict[str, Any] | None:
        doc = self.docs.get(str(query.get("source_key")))
        return copy.deepcopy(doc) if doc is not None else None

    def update_one(self, query: dict[str, Any], update: dict[str, Any], upsert: bool = False) -> None:
        key = str(query.get("source_key"))
        if not key and not upsert:
            return
        current = copy.deepcopy(self.docs.get(key, {"source_key": key}))
        for dotted_key, value in dict(update.get("$set") or {}).items():
            _set_dotted(current, dotted_key, value)
        for dotted_key in dict(update.get("$unset") or {}).keys():
            _unset_dotted(current, dotted_key)
        self.docs[key] = current


class FakeCollectionContext:
    def __init__(self, collection: FakeCollection) -> None:
        self.collection = collection

    def __enter__(self) -> FakeCollection:
        return self.collection

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False
