from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INVDES_ROOT = os.environ.get("INVDES_ROOT", str(PROJECT_ROOT.parent / "invDesMobility"))
DEFAULT_SOURCE_RUN_LINK = os.environ.get(
    "INVDES_SOURCE_RUN_LINK",
    str(Path(DEFAULT_INVDES_ROOT) / "06_runs" / "current__latest_default_semiconductor_pipeline"),
)
DEFAULT_2D_MOBILITY_ROOT = os.environ.get("TWO_D_MOBILITY_ROOT", str(PROJECT_ROOT.parent / "2d-mobility"))
DEFAULT_EXPECTED_TOPK = 10
