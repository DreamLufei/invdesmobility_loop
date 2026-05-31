from __future__ import annotations

import argparse
import json
import tempfile
import unittest

from pathlib import Path
from unittest.mock import patch

from invdesmobility.cli import run_command
from invdesmobility.models import BatchLaunchResult, PreflightResult
from tests.test_support import FakeCollection, FakeCollectionContext, build_sample_run


class CliIntegrationTests(unittest.TestCase):
    def test_run_command_writes_ingestion_summary_under_batch_runs_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            link_path, run_root = build_sample_run(base_dir, item_count=2)
            collection = FakeCollection()
            runs_root = base_dir / "runs" / "invdesmobility__batch"
            batch_result = BatchLaunchResult(
                batch_tag="invdesmobility__batch",
                runs_root=str(runs_root),
                command=("python", "run_mongo_batch.py", "--json", "--dry-run"),
                returncode=0,
                final_state={"batch": {"summary_path": str(runs_root / "batch_summary.json")}},
                stdout="{}",
                stderr="",
            )
            preflight = PreflightResult(
                two_d_mobility_root="/fake/2d-mobility",
                effective_env={"MONGO_URI": "mongodb://example"},
            )
            args = argparse.Namespace(
                source_run_link=str(link_path),
                force=False,
                dry_run_batch=True,
            )

            with patch("invdesmobility.cli.run_preflight", return_value=preflight), patch(
                "invdesmobility.cli.open_collection",
                return_value=FakeCollectionContext(collection),
            ), patch(
                "invdesmobility.cli.launch_batch",
                return_value=batch_result,
            ):
                rc = run_command(args)

            summary_path = runs_root / "ingestion_summary.json"
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(rc, 0)
            self.assertEqual(payload["run_id"], run_root.name)
            self.assertEqual(payload["cif_count"], 2)
            self.assertEqual(payload["ingestion"]["upserted_count"], 2)
            self.assertEqual(payload["batch"]["batch_tag"], batch_result.batch_tag)


if __name__ == "__main__":
    unittest.main()
