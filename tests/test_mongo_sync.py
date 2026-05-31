from __future__ import annotations

import tempfile
import unittest

from pathlib import Path

from pymatgen.core import Structure

from invdesmobility.mongo_sync import sync_source_items
from invdesmobility.source_loader import resolve_run_source
from tests.test_support import FakeCollection, build_sample_run


class MongoSyncTests(unittest.TestCase):
    def test_sync_source_items_writes_roundtrippable_structure_and_source_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            link_path, _ = build_sample_run(Path(tmpdir), item_count=1)
            source = resolve_run_source(link_path)
            collection = FakeCollection()
            summary = sync_source_items(collection, source, force=False)

        item = source.items[0]
        stored = collection.docs[item.source_key]
        self.assertEqual(summary.upserted_count, 1)
        self.assertEqual(stored["source_key"], item.source_key)
        self.assertEqual(stored["material_id"], item.material_id)
        self.assertEqual(stored["formula_pretty"], item.formula_pretty)
        self.assertEqual(stored["formula_reduced_abc"], item.formula_reduced_abc)
        self.assertEqual(stored["spacegroup"]["point_group"], item.spacegroup_point_group)
        self.assertEqual(stored["spacegroup"]["crystal_system"], item.spacegroup_crystal_system)
        self.assertEqual(stored["spacegroup"]["source"], "spglib")
        self.assertEqual(stored["mobility_calc"]["status"], "pending")
        restored = Structure.from_dict(stored["structure"])
        self.assertEqual(restored.composition.reduced_formula, item.formula_reduced)

    def test_sync_source_items_skips_completed_documents_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            link_path, _ = build_sample_run(Path(tmpdir), item_count=1)
            source = resolve_run_source(link_path)
            collection = FakeCollection()
            item = source.items[0]
            collection.docs[item.source_key] = {
                "source_key": item.source_key,
                "material_id": item.material_id,
                "mobility_calc": {"status": "completed", "results": {"ok": True}},
                "mobility_agent": {"status": "completed"},
            }
            summary = sync_source_items(collection, source, force=False)

        self.assertEqual(summary.skipped_count, 1)
        self.assertEqual(summary.selected_material_ids, ())
        self.assertEqual(collection.docs[item.source_key]["mobility_calc"]["status"], "completed")
        self.assertIn("mobility_agent", collection.docs[item.source_key])

    def test_sync_source_items_force_resets_terminal_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            link_path, _ = build_sample_run(Path(tmpdir), item_count=1)
            source = resolve_run_source(link_path)
            collection = FakeCollection()
            item = source.items[0]
            collection.docs[item.source_key] = {
                "source_key": item.source_key,
                "material_id": item.material_id,
                "mobility_calc": {
                    "status": "completed",
                    "results": {"old": True},
                    "run_dir": "/tmp/old-run",
                    "completed_at": "2026-04-11T00:00:00Z",
                },
                "mobility_agent": {
                    "status": "completed",
                    "summary": {"old": True},
                },
            }
            summary = sync_source_items(collection, source, force=True)

        stored = collection.docs[item.source_key]
        self.assertEqual(summary.force_reset_count, 1)
        self.assertEqual(stored["mobility_calc"]["status"], "pending")
        self.assertNotIn("results", stored["mobility_calc"])
        self.assertNotIn("run_dir", stored["mobility_calc"])
        self.assertNotIn("mobility_agent", stored)
        self.assertEqual(summary.selected_material_ids, (item.material_id,))


if __name__ == "__main__":
    unittest.main()
