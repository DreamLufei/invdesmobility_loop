from __future__ import annotations

import json
import tempfile
import unittest

from pathlib import Path
from unittest.mock import patch

from invdesmobility.batch_launcher import build_claim_filter, launch_batch


class _FakeCompleted:
    def __init__(self) -> None:
        self.returncode = 0
        self.stdout = '{"batch": {"summary_path": "/tmp/batch_summary.json"}}'
        self.stderr = ""


class BatchLauncherTests(unittest.TestCase):
    def test_build_claim_filter_defaults_to_source_run_id(self) -> None:
        self.assertEqual(build_claim_filter(run_id="demo_run"), {"invdes_source.run_id": "demo_run"})

    def test_launch_batch_accepts_explicit_claim_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_root_base = Path(tmpdir) / "runs"
            with patch("invdesmobility.batch_launcher.subprocess.run", return_value=_FakeCompleted()) as mock_run:
                result = launch_batch(
                    two_d_mobility_root=tmpdir,
                    base_env={"MONGO_URI": "mongodb://example"},
                    run_id=None,
                    source_code="loop_01",
                    mongo_db="materials_database",
                    mongo_collection="invdesmobility",
                    batch_tag="loop_01__mobility_batch",
                    dry_run_batch=True,
                    runs_root_base=runs_root_base,
                    claim_filter={"loop_metadata.round_id": "loop_01"},
                )
                self.assertTrue((runs_root_base / "loop_01__mobility_batch").exists())

        self.assertEqual(result.batch_tag, "loop_01__mobility_batch")
        call = mock_run.call_args
        self.assertIsNotNone(call)
        env = dict(call.kwargs["env"])
        self.assertEqual(json.loads(env["MONGO_CLAIM_FILTER_JSON"]), {"loop_metadata.round_id": "loop_01"})


if __name__ == "__main__":
    unittest.main()
