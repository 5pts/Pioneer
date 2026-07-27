"""Freeze code and input hashes immediately before formal prediction.

The lock refuses to run after predictions exist. This shows that the protocol,
prompts, evaluation census, and analysis code were not changed after model
outputs became available.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    """Return an uppercase SHA-256 digest for a local file."""
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    """Create the pre-run execution lock after the data audit passes."""
    raw = PROJECT / "results" / "raw" / "formal_predictions.jsonl"
    if raw.exists() and raw.stat().st_size:
        raise RuntimeError("Predictions already exist; refusing to create a pre-run lock.")
    audit = json.loads(
        (PROJECT / "data" / "processed" / "audit_report.json").read_text("utf-8")
    )
    if audit.get("status") != "PASS":
        raise RuntimeError("Data audit has not passed.")

    # Freeze every artifact that can influence predictions or evaluation.
    files = [
        "config/protocol.json",
        "config/prompts.json",
        "config/frozen_manifest.json",
        "data/processed/level1_few_shot.csv",
        "data/processed/level2_few_shot.csv",
        "data/processed/evaluation_census.csv",
        "data/processed/preparation_report.json",
        "data/processed/audit_report.json",
        "src/prepare.py",
        "src/audit.py",
        "src/run.py",
        "src/evaluate.py",
        "src/lock.py",
        "tests/test_run.py",
        "tests/test_metrics.py",
    ]
    lock = {
        "status": "LOCKED_BEFORE_PREDICTIONS",
        "locked_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_file_absent_or_empty": True,
        "sha256": {relative: sha256(PROJECT / relative) for relative in files},
    }
    path = PROJECT / "config" / "execution_lock.json"
    path.write_text(json.dumps(lock, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(lock, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

