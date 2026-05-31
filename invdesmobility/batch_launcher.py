from __future__ import annotations

import json
import os
import subprocess
import sys

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import BatchLaunchResult


def build_batch_tag(source_code: str, *, now: datetime | None = None) -> str:
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return f"inv_{source_code}_{stamp}"


def build_claim_filter(*, run_id: str | None = None, claim_filter: dict[str, Any] | None = None) -> dict[str, Any]:
    if claim_filter is not None:
        return dict(claim_filter)
    text = str(run_id or "").strip()
    if not text:
        raise ValueError("run_id is required when claim_filter is not provided")
    return {"invdes_source.run_id": text}


def _extract_json_payload(stdout: str) -> dict[str, Any] | None:
    text = str(stdout or "").strip()
    if not text:
        return None
    try:
        return dict(json.loads(text) or {})
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return dict(json.loads(text[start : end + 1]) or {})
    except json.JSONDecodeError:
        return None


def launch_batch(
    *,
    two_d_mobility_root: str | os.PathLike[str],
    base_env: dict[str, str],
    run_id: str | None,
    source_code: str,
    mongo_db: str,
    mongo_collection: str,
    batch_tag: str | None = None,
    force: bool = False,
    dry_run_batch: bool = False,
    runs_root_base: str | os.PathLike[str],
    claim_filter: dict[str, Any] | None = None,
) -> BatchLaunchResult:
    batch_tag = str(batch_tag or build_batch_tag(source_code)).strip()
    runs_root = Path(runs_root_base) / batch_tag
    runs_root.mkdir(parents=True, exist_ok=True)
    claim_filter_payload = build_claim_filter(run_id=run_id, claim_filter=claim_filter)

    env = dict(base_env)
    env.update(
        {
            "MONGO_DB": str(mongo_db),
            "MONGO_COLLECTION": str(mongo_collection),
            "MONGO_CLAIM_FILTER_JSON": json.dumps(
                claim_filter_payload,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "BATCH_TAG": batch_tag,
            "RUNS_ROOT": str(runs_root),
        }
    )

    command = [sys.executable, "run_mongo_batch.py", "--json"]
    if dry_run_batch:
        command.append("--dry-run")
    if force:
        command.append("--fresh-materials")

    completed = subprocess.run(
        command,
        cwd=str(two_d_mobility_root),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    return BatchLaunchResult(
        batch_tag=batch_tag,
        runs_root=str(runs_root),
        command=tuple(command),
        returncode=int(completed.returncode),
        final_state=_extract_json_payload(completed.stdout),
        stdout=str(completed.stdout or ""),
        stderr=str(completed.stderr or ""),
    )
