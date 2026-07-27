"""Validate the frozen CAPC-CG evaluation inputs before any API calls.

The audit checks class balance, time boundaries, unique identifiers, prompt/test
separation, and the hashes recorded during data preparation. It writes a small
machine-readable PASS report that ``run.py`` requires before prediction.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data" / "processed"


def read_csv(name: str) -> list[dict]:
    """Read one UTF-8 CSV from the private processed-data directory."""
    with (DATA / name).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    """Return an uppercase SHA-256 digest for a local file."""
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    """Run every pre-execution integrity check and write the audit report."""
    protocol = json.loads((PROJECT / "config" / "protocol.json").read_text("utf-8"))
    frozen = json.loads(
        (PROJECT / "config" / "frozen_manifest.json").read_text("utf-8")
    )
    level1 = read_csv("level1_few_shot.csv")
    level2 = read_csv("level2_few_shot.csv")
    evaluation = read_csv("evaluation_census.csv")

    assert protocol["frozen_before_predictions"] is True
    assert frozen["protocol_frozen_before_predictions"] is True
    for relative, expected in frozen["sha256"].items():
        assert sha256(PROJECT / relative) == expected, relative

    # Enforce the preregistered prompt balance before any model is called.
    assert Counter(row["gold_label"] for row in level1) == {
        label: protocol["level1_few_shot_per_label"]
        for label in ("W", "R", "N")
    }
    assert Counter(row["gold_label"] for row in level2) == {
        label: protocol["level2_few_shot_per_label"]
        for label in ("B", "Y", "C", "G")
    }
    examples = level1 + level2
    assert all(row["period"] == "recent" for row in examples)
    assert all(row["source_split"] == "train" for row in examples)
    assert len({row["content_id"] for row in examples}) == len(examples)
    assert len({row["document_id"] for row in examples}) == len(examples)

    assert len(evaluation) == len({row["sample_id"] for row in evaluation})
    assert [row["sample_id"] for row in evaluation] == frozen["evaluation_ids"]
    # Exclude both exact text overlap and shared-document leakage.
    assert not (
        {row["content_id"] for row in examples}
        & {row["content_id"] for row in evaluation}
    )
    assert not (
        {row["document_id"] for row in examples}
        & {row["document_id"] for row in evaluation}
    )
    # Validate time bins and the official two-stage label relationships.
    for row in evaluation:
        year = int(row["year"])
        if row["period"] == "recent":
            assert 2013 <= year <= 2023
        elif row["period"] == "earlier":
            assert 1979 <= year <= 2012
        else:
            raise AssertionError(row["period"])
        if row["level2_eval"] == "1":
            assert row["gold_level1"] == "W"
            assert row["gold_final"] in {"B", "Y", "C", "G"}
        if row["final_eval"] == "1":
            assert row["gold_final"] in {"B", "Y", "C", "G", "R"}
        if row["gold_final"] == "R":
            assert row["gold_level1"] == "R"

    report = {
        "status": "PASS",
        "evaluation_rows": len(evaluation),
        "evaluation_documents": len({row["document_id"] for row in evaluation}),
        "prompt_rows": len(examples),
        "prompt_documents": len({row["document_id"] for row in examples}),
        "prompt_evaluation_content_overlap": 0,
        "prompt_evaluation_document_overlap": 0,
        "final_counts": {
            f"{period}|{label}": count
            for (period, label), count in sorted(
                Counter(
                    (row["period"], row["gold_final"])
                    for row in evaluation
                    if row["final_eval"] == "1"
                ).items()
            )
        },
        "level1_counts": {
            f"{period}|{label}": count
            for (period, label), count in sorted(
                Counter(
                    (row["period"], row["gold_level1"])
                    for row in evaluation
                    if row["level1_eval"] == "1"
                ).items()
            )
        },
    }
    (DATA / "audit_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

