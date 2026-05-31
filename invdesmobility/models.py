from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SourceItem:
    rank: int
    rank_token: str
    cif_name: str
    cif_path: str
    material_id: str
    source_key: str
    formula_pretty: str
    formula_reduced: str
    formula_reduced_abc: str
    spacegroup_symbol: str
    spacegroup_number: int
    spacegroup_point_group: str
    spacegroup_crystal_system: str
    spacegroup_hall: str
    structure_dict: dict[str, Any]
    bandgap_ev: float | None = None
    formation_energy_ev_per_atom: float | None = None
    alignn_mobility_score: float | None = None


@dataclass(frozen=True)
class RunSource:
    run_link: str
    run_root: str
    run_id: str
    source_code: str
    top10_dir: str
    top10_csv_path: str
    manifest_path: str
    manifest: dict[str, Any]
    items: tuple[SourceItem, ...]


@dataclass(frozen=True)
class PreflightResult:
    two_d_mobility_root: str
    effective_env: dict[str, str]


@dataclass(frozen=True)
class SyncSummary:
    upserted_count: int
    skipped_count: int
    force_reset_count: int
    selected_material_ids: tuple[str, ...]


@dataclass(frozen=True)
class BatchLaunchResult:
    batch_tag: str
    runs_root: str
    command: tuple[str, ...]
    returncode: int
    final_state: dict[str, Any] | None
    stdout: str
    stderr: str
