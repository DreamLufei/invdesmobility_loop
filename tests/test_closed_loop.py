from __future__ import annotations

import json
import tempfile
import unittest

from pathlib import Path
from unittest.mock import patch

from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter

import invdesmobility.closed_loop as closed_loop


class ClosedLoopRoundTests(unittest.TestCase):
    def _write_cif(self, path: Path, *, a: float, b: float, c: float) -> None:
        structure = Structure(
            lattice=Lattice.orthorhombic(a, b, c),
            species=["Si", "Si"],
            coords=[[0.1, 0.2, 0.3], [0.6, 0.7, 0.3]],
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        CifWriter(structure).write_file(str(path))

    def _write_text(self, path: Path, text: str = "ok\n") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def _prepare_fake_invdes_root(self, root: Path) -> None:
        for rel in (
            "05_steps/09_closed_loop_feedback/extract_trusted_feedback.py",
            "05_steps/09_closed_loop_feedback/build_alignn_round_dataset.py",
            "05_steps/09_closed_loop_feedback/build_diffcsp_round_dataset.py",
            "05_steps/09_closed_loop_feedback/publish_round_candidates_to_mongo.py",
            "05_steps/06_alignn_mobility_rank/train_best.sh",
            "05_steps/02_finetune_generator/run.sh",
            "05_steps/03_generate_structures/run_multigpu.sh",
            "05_steps/08_orchestration/run_dedup_orthorhombic_semiconductor_pipeline.sh",
            "04_models/04_alignn_mobility/mobility_reg_v1_bs8_lr5e4_wu200_nw4/best_model.pt",
            "04_models/04_alignn_mobility/mobility_reg_v1_bs8_lr5e4_wu200_nw4/config.json",
            "04_models/01_diffcsp_generator/finetuned/mobility2d_highquality280_ft_v1/best.ckpt",
            "03_datasets/01_source_cif/high_quality_280/id_prop.csv",
        ):
            self._write_text(root / rel)
        self._write_cif(root / "03_datasets/01_source_cif/base_ref_a.cif", a=3.0, b=4.0, c=18.0)
        self._write_cif(root / "03_datasets/01_source_cif/base_ref_b.cif", a=3.2, b=4.2, c=18.2)
        self._write_cif(root / "03_datasets/01_source_cif/high_quality_280/base_seed_001.cif", a=3.1, b=4.1, c=18.1)

    def test_closed_loop_round_uses_feedback_augmented_dedup_reference_pool(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            project_root = tmp / "project"
            project_root.mkdir()
            invdes_root = tmp / "invDesMobility"
            two_d_root = tmp / "2d-mobility"
            feedback_batch_root = tmp / "feedback_batch"
            feedback_batch_root.mkdir()
            (two_d_root / ".env.local").parent.mkdir(parents=True, exist_ok=True)
            (two_d_root / ".env.local").write_text("MONGO_URI=mongodb://example\n", encoding="utf-8")
            self._prepare_fake_invdes_root(invdes_root)

            seen_screening_envs: list[dict[str, str]] = []

            def fake_runner(command, cwd, env):  # type: ignore[no-untyped-def]
                args = list(command)
                if args[0].endswith("python") and args[1].endswith("extract_trusted_feedback.py"):
                    output_dir = Path(args[args.index("--output-dir") + 1])
                    trusted_cif_dir = output_dir / "trusted_relaxed_cif"
                    trusted_cif_dir.mkdir(parents=True, exist_ok=True)
                    relaxed_cif = trusted_cif_dir / "loop_00__trusted_001.cif"
                    self._write_cif(relaxed_cif, a=3.4, b=4.4, c=18.4)
                    (output_dir / "trusted_channels.csv").write_text(
                        "round_id,round_index,batch_id,material_id,channel,relaxed_cif_path,structure_hash\n"
                        f"round_00_bootstrap,0,batch_a,loop_00__trusted_001,electron_x,{relaxed_cif},hash001\n",
                        encoding="utf-8",
                    )
                    (output_dir / "trusted_materials.csv").write_text(
                        "round_id,round_index,batch_id,material_id,usable_channel_count,best_channel,best_mobility_cm2_vs,best_target,best_direction,best_carrier,relaxed_cif_path,relaxed_contcar_path,input_poscar_path,structure_hash,source_workdir\n"
                        f"round_00_bootstrap,0,batch_a,loop_00__trusted_001,1,electron_x,1600.0,3.204120,x,electron,{relaxed_cif},/tmp/CONTCAR,/tmp/POSCAR,hash001,/tmp/workdir\n",
                        encoding="utf-8",
                    )
                    (output_dir / "rejected_feedback.csv").write_text("round_id,batch_id,material_id,reason\n", encoding="utf-8")
                    (output_dir / "feedback_summary.json").write_text(
                        json.dumps({"counts": {"trusted_material_count": 1, "trusted_channel_count": 1}}),
                        encoding="utf-8",
                    )
                elif args[0].endswith("python") and args[1].endswith("build_alignn_round_dataset.py"):
                    output_dir = Path(args[args.index("--output-dir") + 1])
                    output_dir.mkdir(parents=True, exist_ok=True)
                    (output_dir / "id_prop.csv").write_text("base_seed_001.cif,1.234000\n", encoding="utf-8")
                    manifest_path = Path(args[args.index("--manifest-path") + 1])
                    manifest_path.write_text(json.dumps({"counts": {"feedback_samples": 1, "total_samples": 2}}), encoding="utf-8")
                elif args[0] == "bash" and args[1].endswith("train_best.sh"):
                    output_dir = Path(env["ALIGNN_MOBILITY_OUTPUT_DIR"])
                    output_dir.mkdir(parents=True, exist_ok=True)
                    (output_dir / "best_model.pt").write_text("pt\n", encoding="utf-8")
                    (output_dir / "config.json").write_text("{}\n", encoding="utf-8")
                elif args[0].endswith("python") and args[1].endswith("build_diffcsp_round_dataset.py"):
                    manifest_path = Path(args[args.index("--manifest-path") + 1])
                    manifest_path.write_text(json.dumps({"counts": {"feedback_unique_structures": 1, "train_rows": 10000}}), encoding="utf-8")
                elif args[0] == "bash" and args[1].endswith("05_steps/02_finetune_generator/run.sh"):
                    output_dir = Path(env["FINETUNED_DIR"])
                    output_dir.mkdir(parents=True, exist_ok=True)
                    (output_dir / "best.ckpt").write_text("ckpt\n", encoding="utf-8")
                elif args[0] == "bash" and args[1].endswith("05_steps/03_generate_structures/run_multigpu.sh"):
                    run_id = env["RUN_ID"]
                    run_root = invdes_root / "06_runs" / run_id
                    generated_dir = run_root / "03_generate_structures" / "generated_cif"
                    generated_dir.mkdir(parents=True, exist_ok=True)
                    self._write_cif(generated_dir / "gen_001.cif", a=3.8, b=4.8, c=18.8)
                    (run_root / "manifest.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
                elif args[0] == "bash" and args[1].endswith("05_steps/08_orchestration/run_dedup_orthorhombic_semiconductor_pipeline.sh"):
                    seen_screening_envs.append(dict(env))
                    reference_dir = Path(env["DEDUP_REFERENCE_DIR"])
                    self.assertEqual(len(list(reference_dir.glob("*.cif"))), 4)
                    run_id = env["RUN_NAME"]
                    run_root = invdes_root / "06_runs" / run_id
                    strict90_dir = run_root / "10_top10_strict90"
                    strict90_cif_dir = strict90_dir / "strict90_cif"
                    strict90_cif_dir.mkdir(parents=True, exist_ok=True)
                    strict90_cif = strict90_cif_dir / "rank_01__candidate_a.cif"
                    self._write_cif(strict90_cif, a=4.0, b=5.0, c=19.0)
                    merged_csv = strict90_dir / "top10_candidates_strict90.csv"
                    merged_csv.write_text(
                        "rank,cif_name,cif_path,mobility_score,strict90_cif_path,strict90_written\n"
                        f"1,candidate_a.cif,{strict90_cif},3.2,{strict90_cif},True\n",
                        encoding="utf-8",
                    )
                    (run_root / "manifest.json").write_text(json.dumps({"run_id": run_id, "reference_dedup_dir": env["DEDUP_REFERENCE_DIR"]}), encoding="utf-8")
                elif args[0].endswith("python") and args[1].endswith("publish_round_candidates_to_mongo.py"):
                    manifest_path = Path(args[args.index("--output-manifest") + 1])
                    manifest_path.write_text(json.dumps({"counts": {"publishable_rows": 1}}), encoding="utf-8")
                else:
                    raise AssertionError(f"Unexpected command: {args}")
                return None

            with patch.object(closed_loop, "PROJECT_ROOT", project_root):
                payload = closed_loop.run_closed_loop_round(
                    round_index=1,
                    feedback_batch_root=feedback_batch_root,
                    invdes_root=invdes_root,
                    two_d_mobility_root=two_d_root,
                    mongo_collection="invdesmobility_loop",
                    publish_dry_run=True,
                    command_runner=fake_runner,
                )

        self.assertEqual(payload["reference_manifest"]["counts"]["base_reference_count"], 3)
        self.assertEqual(payload["reference_manifest"]["counts"]["feedback_reference_count"], 1)
        self.assertTrue(seen_screening_envs)
        self.assertEqual(
            Path(seen_screening_envs[0]["DEDUP_REFERENCE_DIR"]),
            project_root / "runs" / "loop_01__closed_loop_round" / "02_reference_pool_for_dedup",
        )
        self.assertEqual(seen_screening_envs[0]["DEDUP_MODE"], "formula")
        self.assertEqual(seen_screening_envs[0]["DEDUP_BANDGAP_THRESHOLD"], "0.2")
        self.assertTrue(payload["downstream_batch"]["skipped"])
        self.assertEqual(payload["screening"]["top_k"], 10)


if __name__ == "__main__":
    unittest.main()
