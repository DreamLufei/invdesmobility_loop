from __future__ import annotations

import tempfile
import unittest

from pathlib import Path

from invdesmobility.source_loader import resolve_run_source
from tests.test_support import build_sample_run


class SourceLoaderTests(unittest.TestCase):
    def test_resolves_run_symlink_and_uses_ranked_cif_directory_as_truth(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            link_path, run_root = build_sample_run(Path(tmpdir), item_count=2)
            source = resolve_run_source(link_path)

        self.assertEqual(source.run_id, run_root.name)
        self.assertEqual(source.run_link, str(link_path))
        self.assertEqual(source.source_code, "260409_run_t10")
        self.assertEqual(len(source.items), 2)
        first = source.items[0]
        self.assertEqual(first.rank, 1)
        self.assertEqual(first.material_id, "inv_260409_run_t10_r01_sample_0001")
        self.assertTrue(first.cif_path.endswith("06_top10_cif/rank_01__sample_0001.cif"))
        self.assertEqual(first.formula_pretty, "Si")
        self.assertEqual(first.formula_reduced_abc, "Si1")
        self.assertTrue(first.spacegroup_symbol)
        self.assertTrue(first.spacegroup_point_group)
        self.assertTrue(first.spacegroup_crystal_system)
        self.assertAlmostEqual(first.bandgap_ev or 0.0, 1.1)
        self.assertAlmostEqual(first.formation_energy_ev_per_atom or 0.0, -0.1)
        self.assertAlmostEqual(first.alignn_mobility_score or 0.0, 3.5)
        self.assertTrue(first.source_key.endswith("::rank_01::sample_0001.cif"))

    def test_resolves_nested_ranked_cif_directory_and_uses_parent_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, run_root = build_sample_run(Path(tmpdir), item_count=2, top10_dir_rel="07_top10_cif/strict90_cif")
            source = resolve_run_source(run_root / "07_top10_cif" / "strict90_cif")

        self.assertEqual(source.source_code, "260409_run_s90")
        self.assertTrue(source.top10_dir.endswith("07_top10_cif/strict90_cif"))
        self.assertTrue(source.top10_csv_path.endswith("07_top10_cif/top10_candidates.csv"))
        self.assertTrue(source.items[0].material_id.startswith("inv_260409_run_s90_r01_"))

    def test_resolves_new_10_top10_strict90_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _, run_root = build_sample_run(Path(tmpdir), item_count=3, top10_dir_rel="10_top10_strict90/strict90_cif")
            source = resolve_run_source(run_root)

        self.assertEqual(source.source_code, "260409_run_s90")
        self.assertTrue(source.top10_dir.endswith("10_top10_strict90/strict90_cif"))
        self.assertTrue(source.top10_csv_path.endswith("10_top10_strict90/top10_candidates_strict90.csv"))
        self.assertEqual(len(source.items), 3)
        self.assertTrue(source.items[0].material_id.startswith("inv_260409_run_s90_r01_"))


if __name__ == "__main__":
    unittest.main()
