from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from pymongo import MongoClient

from .models import RunSource, SourceItem, SyncSummary

MONGO_DB_NAME = "materials_database"
MONGO_COLLECTION_NAME = "invdesmobility"

_MOBILITY_CALC_RESET_FIELDS = (
    "mobility_calc.batch_tag",
    "mobility_calc.started_at",
    "mobility_calc.completed_at",
    "mobility_calc.failed_at",
    "mobility_calc.run_dir",
    "mobility_calc.results",
    "mobility_calc.quality_label",
    "mobility_calc.error",
    "mobility_calc.potcar_used",
)


@contextmanager
def open_collection(
    mongo_uri: str,
    db_name: str = MONGO_DB_NAME,
    collection_name: str = MONGO_COLLECTION_NAME,
) -> Iterator[Any]:
    client = MongoClient(
        mongo_uri,
        serverSelectionTimeoutMS=30000,
        connectTimeoutMS=30000,
        socketTimeoutMS=30000,
    )
    try:
        yield client[db_name][collection_name]
    finally:
        try:
            client.close()
        except Exception:
            pass


def ensure_indexes(collection: Any) -> None:
    collection.create_index([("source_key", 1)], unique=True, name="source_key_unique")
    collection.create_index([("material_id", 1)], name="material_id_idx")
    collection.create_index([("invdes_source.run_id", 1)], name="invdes_source_run_id_idx")
    collection.create_index([("mobility_calc.status", 1)], name="mobility_calc_status_idx")


def _base_set_fields(item: SourceItem, source: RunSource) -> dict[str, Any]:
    return {
        "material_id": item.material_id,
        "source_key": item.source_key,
        "formula_pretty": item.formula_pretty,
        "formula_reduced": item.formula_reduced,
        "formula_reduced_abc": item.formula_reduced_abc,
        "spacegroup": {
            "symbol": item.spacegroup_symbol,
            "number": item.spacegroup_number,
            "point_group": item.spacegroup_point_group,
            "source": "spglib",
            "crystal_system": item.spacegroup_crystal_system,
            "hall": item.spacegroup_hall,
        },
        "structure": item.structure_dict,
        "invdes_source": {
            "run_link": source.run_link,
            "run_id": source.run_id,
            "rank": item.rank,
            "cif_name": item.cif_name,
            "cif_path": item.cif_path,
            "manifest_path": source.manifest_path,
            "top10_csv_path": source.top10_csv_path,
            "bandgap_ev": item.bandgap_ev,
            "formation_energy_ev_per_atom": item.formation_energy_ev_per_atom,
            "alignn_mobility_score": item.alignn_mobility_score,
        },
    }


def _reset_unset_fields() -> dict[str, str]:
    unset_fields = {key: "" for key in _MOBILITY_CALC_RESET_FIELDS}
    unset_fields["mobility_agent"] = ""
    return unset_fields


def _status_of(doc: dict[str, Any] | None) -> str:
    if not doc:
        return ""
    mobility_calc = dict(doc.get("mobility_calc") or {})
    return str(mobility_calc.get("status") or "").strip().lower()


def sync_source_items(collection: Any, source: RunSource, *, force: bool = False) -> SyncSummary:
    ensure_indexes(collection)
    upserted_count = 0
    skipped_count = 0
    force_reset_count = 0
    selected_material_ids: list[str] = []

    for item in source.items:
        existing = collection.find_one({"source_key": item.source_key})
        status = _status_of(existing)
        set_fields = _base_set_fields(item, source)

        if status == "completed" and not force:
            skipped_count += 1
            continue

        if force:
            set_fields["mobility_calc.status"] = "pending"
            collection.update_one(
                {"source_key": item.source_key},
                {"$set": set_fields, "$unset": _reset_unset_fields()},
                upsert=True,
            )
            upserted_count += 1
            selected_material_ids.append(item.material_id)
            if existing:
                force_reset_count += 1
            continue

        if status == "running":
            collection.update_one(
                {"source_key": item.source_key},
                {"$set": set_fields},
                upsert=True,
            )
            upserted_count += 1
            continue

        update: dict[str, Any] = {
            "$set": {
                **set_fields,
                "mobility_calc.status": "pending",
            }
        }
        if existing:
            update["$unset"] = _reset_unset_fields()
        collection.update_one({"source_key": item.source_key}, update, upsert=True)
        upserted_count += 1
        selected_material_ids.append(item.material_id)

    return SyncSummary(
        upserted_count=upserted_count,
        skipped_count=skipped_count,
        force_reset_count=force_reset_count,
        selected_material_ids=tuple(selected_material_ids),
    )
