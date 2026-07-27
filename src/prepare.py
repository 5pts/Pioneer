"""Build a leakage-free temporal evaluation census from authorized CAPC-CG data.

The script reads the gated human-label files, removes conflicting annotations,
matches paragraphs to document metadata, samples recent few-shot examples, and
excludes all prompt texts and prompt documents from formal evaluation.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import duckdb
import pyarrow.parquet as pq


PROJECT = Path(__file__).resolve().parents[1]
CORPUS = PROJECT.parent / "external_corpora" / "capc-cg-v1.0"
FULL_CORPUS = CORPUS / "data" / "CN_DOCS_FINAL_complete.parquet"
DATA = PROJECT / "data" / "processed"
LEVEL1_LABELS = ("W", "R", "N")
LEVEL2_LABELS = ("B", "Y", "C", "G")
FINAL_LABELS = ("B", "Y", "C", "G", "R")
USER_PREFIX = "待判断政策段落："


def normalize_text(value: str) -> str:
    """Normalize Unicode and whitespace for conflicts and stable identifiers."""
    value = unicodedata.normalize("NFKC", str(value or ""))
    value = value.replace("\u3000", " ")
    return re.sub(r"\s+", " ", value).strip()


def match_key(value: str) -> str:
    """Create the conservative whitespace-only key used for corpus matching."""
    return re.sub(
        r"\s+", " ", str(value or "").replace("\u3000", " ")
    ).strip()


def extract_text(value: str) -> str:
    """Remove the dataset's user-message prefix from a policy paragraph."""
    value = str(value or "").strip()
    if value.startswith(USER_PREFIX):
        value = value[len(USER_PREFIX) :]
    return value.strip()


def load_task(folder: str, allowed: tuple[str, ...]) -> tuple[dict[str, dict], dict]:
    """Load one human-label task and exclude texts with conflicting labels."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    raw_rows = 0
    for split in ("train", "validation"):
        path = CORPUS / folder / f"{split}.parquet"
        for row in pq.read_table(path).to_pylist():
            raw_rows += 1
            messages = row["messages"]
            user = next(item["content"] for item in messages if item["role"] == "user")
            assistant = next(
                item["content"]
                for item in reversed(messages)
                if item["role"] == "assistant"
            )
            label = str(assistant).strip().upper()
            if label not in allowed:
                continue
            text = extract_text(user)
            norm = normalize_text(text)
            if norm:
                grouped[norm].append(
                    {
                        "text": text,
                        "label": label,
                        "source_split": split,
                    }
                )

    clean: dict[str, dict] = {}
    conflict_patterns: Counter = Counter()
    duplicate_rows = 0
    for norm, records in grouped.items():
        labels = sorted({record["label"] for record in records})
        if len(labels) != 1:
            conflict_patterns["/".join(labels)] += 1
            continue
        duplicate_rows += max(0, len(records) - 1)
        splits = sorted({record["source_split"] for record in records})
        clean[norm] = {
            "text": records[0]["text"],
            "norm": norm,
            "gold_label": labels[0],
            "source_split": "+".join(splits),
        }
    return clean, {
        "raw_rows": raw_rows,
        "unique_conflict_free_texts": len(clean),
        "conflicts_excluded": sum(conflict_patterns.values()),
        "conflict_patterns": dict(conflict_patterns),
        "same_label_duplicate_rows_collapsed": duplicate_rows,
    }


def parse_year(value: str) -> int | None:
    """Extract a four-digit leading year from issue-date metadata."""
    found = re.match(r"^\s*(\d{4})", str(value or ""))
    return int(found.group(1)) if found else None


def period_for_year(year: int) -> str | None:
    """Map an eligible year to the preregistered recent or earlier period."""
    if 2013 <= year <= 2023:
        return "recent"
    if 1979 <= year <= 2012:
        return "earlier"
    return None


def attach_metadata(rows: dict[str, dict]) -> tuple[dict[str, dict], dict]:
    """Join human-labeled text to full-corpus metadata with DuckDB."""
    connection = duckdb.connect()
    connection.execute("SET enable_progress_bar=false")
    connection.execute("CREATE TEMP TABLE target_keys(match_key VARCHAR PRIMARY KEY)")
    keys = sorted({match_key(row["text"]) for row in rows.values()})
    connection.executemany("INSERT INTO target_keys VALUES (?)", [(key,) for key in keys])
    query = """
        SELECT
            c.DocumentID,
            c.ID,
            c.Paragraph_content,
            c.Title,
            c.IssueDate,
            c.EffectivenessLevel,
            c.Category,
            c.IssuingDepartment
        FROM read_parquet(?) AS c
        INNER JOIN target_keys AS t
          ON trim(
               regexp_replace(
                 replace(c.Paragraph_content, '　', ' '),
                 '\\s+',
                 ' ',
                 'g'
               )
             ) = t.match_key
    """
    result = connection.execute(query, [str(FULL_CORPUS)]).fetchall()
    connection.close()

    # Multiple matches are accepted only when all belong to the same period.
    # Cross-period matches are excluded because their time label is ambiguous.
    candidates: dict[str, list[dict]] = defaultdict(list)
    for (
        document_id,
        paragraph_id,
        text,
        title,
        issue_date,
        effectiveness_level,
        category,
        issuing_department,
    ) in result:
        norm = normalize_text(text)
        if norm not in rows:
            continue
        year = parse_year(issue_date)
        period = period_for_year(year) if year is not None else None
        if period is None:
            continue
        candidates[norm].append(
            {
                "year": year,
                "period": period,
                "issue_date": str(issue_date or ""),
                "document_id": str(document_id or ""),
                "paragraph_id": str(paragraph_id or ""),
                "title": str(title or ""),
                "effectiveness_level": str(effectiveness_level or ""),
                "category": str(category or ""),
                "issuing_department": str(issuing_department or ""),
            }
        )

    attached: dict[str, dict] = {}
    report: Counter = Counter()
    for norm, row in rows.items():
        found = candidates.get(norm, [])
        if not found:
            report["unmatched"] += 1
            continue
        periods = {candidate["period"] for candidate in found}
        if len(periods) != 1:
            report["cross_period_match_excluded"] += 1
            continue
        found.sort(
            key=lambda item: (
                item["year"],
                item["issue_date"],
                item["document_id"],
                item["paragraph_id"],
            )
        )
        selected = found[0]
        attached[norm] = {
            **row,
            **selected,
            "content_id": hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16],
            "match_count": len(found),
            "match_document_count": len(
                {candidate["document_id"] for candidate in found}
            ),
            "text_length": len(row["text"]),
        }
        report["matched"] += 1
        if len(found) > 1:
            report["same_period_multiple_matches"] += 1
    return attached, dict(report)


def choose_random_examples(
    rows: list[dict],
    *,
    labels: tuple[str, ...],
    per_label: int,
    used_content: set[str],
    used_documents: set[str],
    rng: random.Random,
) -> list[dict]:
    """Sample recent train examples without reusing texts or documents."""
    selected: list[dict] = []
    for label in labels:
        candidates = [
            row
            for row in rows
            if row["gold_label"] == label
            and row["period"] == "recent"
            and row["source_split"] == "train"
            and row["content_id"] not in used_content
            and row["document_id"] not in used_documents
        ]
        candidates.sort(key=lambda row: row["content_id"])
        rng.shuffle(candidates)
        picked: list[dict] = []
        for row in candidates:
            if row["document_id"] in used_documents:
                continue
            picked.append(row)
            used_content.add(row["content_id"])
            used_documents.add(row["document_id"])
            if len(picked) == per_label:
                break
        if len(picked) != per_label:
            raise RuntimeError(
                f"Not enough recent train examples for {label}: "
                f"needed {per_label}, found {len(picked)}"
            )
        selected.extend(picked)
    return selected


FIELDS = [
    "sample_id",
    "content_id",
    "gold_label",
    "period",
    "year",
    "issue_date",
    "text",
    "text_length",
    "title",
    "document_id",
    "paragraph_id",
    "effectiveness_level",
    "category",
    "issuing_department",
    "match_count",
    "match_document_count",
    "gold_level1",
    "gold_final",
    "level1_eval",
    "level2_eval",
    "final_eval",
    "source_split",
    "task1_split",
    "task2_split",
]


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write dictionaries as UTF-8 CSV using a stable public schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    """Return an uppercase SHA-256 digest for a local file."""
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    """Create prompt examples, evaluation rows, reports, and frozen hashes."""
    protocol = json.loads((PROJECT / "config" / "protocol.json").read_text("utf-8"))
    rng = random.Random(protocol["seed"])

    task1, task1_report = load_task("task1_level1", LEVEL1_LABELS)
    task2, task2_report = load_task("task2_level2", LEVEL2_LABELS)
    cross_task_conflicts = {
        norm
        for norm in task1.keys() & task2.keys()
        if task1[norm]["gold_label"] != "W"
    }
    for norm in cross_task_conflicts:
        task1.pop(norm, None)
        task2.pop(norm, None)

    metadata_input = {
        norm: task2.get(norm, task1.get(norm))
        for norm in task1.keys() | task2.keys()
    }
    attached, metadata_report = attach_metadata(metadata_input)

    level1_pool = [
        {**attached[norm], **row}
        for norm, row in task1.items()
        if norm in attached
    ]
    level2_pool = [
        {**attached[norm], **row}
        for norm, row in task2.items()
        if norm in attached
    ]

    used_content: set[str] = set()
    used_documents: set[str] = set()
    level2_examples = choose_random_examples(
        level2_pool,
        labels=LEVEL2_LABELS,
        per_label=protocol["level2_few_shot_per_label"],
        used_content=used_content,
        used_documents=used_documents,
        rng=rng,
    )
    level1_examples = choose_random_examples(
        level1_pool,
        labels=LEVEL1_LABELS,
        per_label=protocol["level1_few_shot_per_label"],
        used_content=used_content,
        used_documents=used_documents,
        rng=rng,
    )

    evaluation: list[dict] = []
    excluded_prompt_document_rows = 0
    # Build the census only after removing every prompt text and every row from
    # a source document represented in the prompt.
    for norm in sorted(attached):
        meta = attached[norm]
        if meta["content_id"] in used_content or meta["document_id"] in used_documents:
            excluded_prompt_document_rows += 1
            continue
        row1 = task1.get(norm)
        row2 = task2.get(norm)
        level1_eval = row1 is not None
        level2_eval = row2 is not None
        final_eval = row2 is not None or (
            row1 is not None and row1["gold_label"] == "R"
        )
        gold_level1 = row1["gold_label"] if row1 is not None else "W"
        if row2 is not None:
            if row1 is not None and row1["gold_label"] != "W":
                raise AssertionError("Cross-task conflict survived exclusion.")
            gold_level1 = "W"
            gold_final = row2["gold_label"]
        elif row1 is not None and row1["gold_label"] == "R":
            gold_final = "R"
        else:
            gold_final = ""
        evaluation.append(
            {
                **meta,
                "sample_id": hashlib.sha256(
                    f"formal|{norm}".encode("utf-8")
                ).hexdigest()[:16],
                "gold_level1": gold_level1,
                "gold_final": gold_final,
                "level1_eval": int(level1_eval),
                "level2_eval": int(level2_eval),
                "final_eval": int(final_eval),
                "task1_split": row1["source_split"] if row1 else "",
                "task2_split": row2["source_split"] if row2 else "",
            }
        )

    evaluation.sort(key=lambda row: row["sample_id"])
    for rows in (level1_examples, level2_examples):
        for row in rows:
            row["sample_id"] = hashlib.sha256(
                f"example|{row['norm']}".encode("utf-8")
            ).hexdigest()[:16]
            row["gold_level1"] = (
                row["gold_label"] if row in level1_examples else "W"
            )
            row["gold_final"] = (
                row["gold_label"] if row in level2_examples else ""
            )
            row["level1_eval"] = 0
            row["level2_eval"] = 0
            row["final_eval"] = 0
            row["task1_split"] = (
                row["source_split"] if row in level1_examples else ""
            )
            row["task2_split"] = (
                row["source_split"] if row in level2_examples else ""
            )

    DATA.mkdir(parents=True, exist_ok=True)
    write_csv(DATA / "level1_few_shot.csv", level1_examples)
    write_csv(DATA / "level2_few_shot.csv", level2_examples)
    write_csv(DATA / "evaluation_census.csv", evaluation)

    def period_label_counts(rows: list[dict], label_key: str, scope_key: str) -> dict:
        return {
            f"{period}|{label}": count
            for (period, label), count in sorted(
                Counter(
                    (row["period"], row[label_key])
                    for row in rows
                    if row[scope_key]
                ).items()
            )
        }

    report = {
        "task1": task1_report,
        "task2": task2_report,
        "cross_task_conflicts_excluded": len(cross_task_conflicts),
        "metadata_join": metadata_report,
        "prompt_examples": {
            "level1": Counter(row["gold_label"] for row in level1_examples),
            "level2": Counter(row["gold_label"] for row in level2_examples),
            "all_recent": all(
                row["period"] == "recent"
                for row in level1_examples + level2_examples
            ),
            "all_train_only": all(
                row["source_split"] == "train"
                for row in level1_examples + level2_examples
            ),
            "unique_documents": len(used_documents),
        },
        "evaluation_rows": len(evaluation),
        "excluded_due_to_prompt_text_or_document": excluded_prompt_document_rows,
        "level1_counts": period_label_counts(
            evaluation, "gold_level1", "level1_eval"
        ),
        "level2_counts": period_label_counts(
            evaluation, "gold_final", "level2_eval"
        ),
        "final_counts": period_label_counts(
            evaluation, "gold_final", "final_eval"
        ),
        "unique_evaluation_documents": len(
            {row["document_id"] for row in evaluation}
        ),
        "ambiguous_same_period_metadata_rows": sum(
            int(row["match_document_count"]) > 1 for row in evaluation
        ),
        "year_ranges": {
            period: [
                min(row["year"] for row in evaluation if row["period"] == period),
                max(row["year"] for row in evaluation if row["period"] == period),
            ]
            for period in ("recent", "earlier")
        },
    }
    (DATA / "preparation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=dict),
        encoding="utf-8",
    )

    files_to_freeze = [
        "config/protocol.json",
        "config/prompts.json",
        "data/processed/level1_few_shot.csv",
        "data/processed/level2_few_shot.csv",
        "data/processed/evaluation_census.csv",
        "data/processed/preparation_report.json",
    ]
    frozen = {
        "protocol_frozen_before_predictions": True,
        "level1_example_ids": [row["sample_id"] for row in level1_examples],
        "level2_example_ids": [row["sample_id"] for row in level2_examples],
        "evaluation_ids": [row["sample_id"] for row in evaluation],
        "sha256": {
            relative: sha256(PROJECT / relative) for relative in files_to_freeze
        },
    }
    (PROJECT / "config" / "frozen_manifest.json").write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=dict))


if __name__ == "__main__":
    main()

