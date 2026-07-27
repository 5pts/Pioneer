from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
FINAL_LABELS = ("B", "Y", "C", "G", "R")
LEVEL2_LABELS = ("B", "Y", "C", "G")
LEVEL1_LABELS = ("W", "R", "N")


def load_latest(path: Path) -> list[dict]:
    latest: dict[str, dict] = {}
    for line in path.read_text("utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            latest[row["sample_id"]] = row
    return list(latest.values())


def nested_prediction(row: dict, key: str) -> str | None:
    if key == "level1_prediction":
        return (row.get("level1") or {}).get("prediction")
    return row.get(key)


def classification_metrics(
    rows: list[dict],
    *,
    labels: tuple[str, ...],
    gold_key: str,
    prediction_key: str,
) -> dict:
    other = "__OTHER__"
    prediction_labels = labels + (other,)
    confusion = {
        gold: {predicted: 0 for predicted in prediction_labels}
        for gold in labels
    }
    evaluated = []
    valid_predictions = 0
    for row in rows:
        gold = row.get(gold_key)
        if gold not in labels:
            continue
        predicted = nested_prediction(row, prediction_key)
        evaluated.append(row)
        if predicted in labels:
            valid_predictions += 1
            confusion[gold][predicted] += 1
        else:
            confusion[gold][other] += 1

    per_label = {}
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[gold][label] for gold in labels if gold != label)
        fn = sum(
            confusion[label][predicted]
            for predicted in prediction_labels
            if predicted != label
        )
        support = sum(confusion[label].values())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    total = len(evaluated)
    correct = sum(confusion[label][label] for label in labels)
    overall_accuracy = correct / total if total else 0.0
    standardized_accuracy = sum(
        per_label[label]["recall"] for label in labels
    ) / len(labels)
    macro_f1 = sum(per_label[label]["f1"] for label in labels) / len(labels)

    predicted_marginals = {
        label: sum(confusion[gold][label] for gold in labels)
        for label in prediction_labels
    }
    gold_marginals = {
        label: sum(confusion[label].values()) for label in labels
    }
    expected = (
        sum(
            gold_marginals[label] * predicted_marginals[label]
            for label in labels
        )
        / (total * total)
        if total
        else 0.0
    )
    kappa = (
        (overall_accuracy - expected) / (1 - expected)
        if total and expected < 1
        else 0.0
    )
    return {
        "n": total,
        "n_valid_in_scope_predictions": valid_predictions,
        "n_out_of_scope_or_missing_predictions": total - valid_predictions,
        "overall_accuracy": overall_accuracy,
        "label_standardized_accuracy": standardized_accuracy,
        "macro_f1": macro_f1,
        "cohens_kappa": kappa,
        "per_label": per_label,
        "confusion_matrix": confusion,
    }


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def stratified_document_sample(
    rows: list[dict],
    labels: tuple[str, ...],
    rng: random.Random,
) -> list[dict]:
    by_label_document: dict[str, dict[str, list[dict]]] = {
        label: defaultdict(list) for label in labels
    }
    for row in rows:
        by_label_document[row["gold_final"]][row["document_id"]].append(row)
    sampled: list[dict] = []
    for label in labels:
        clusters = list(by_label_document[label].values())
        if not clusters:
            raise RuntimeError(f"No document clusters for label {label}.")
        for _ in range(len(clusters)):
            sampled.extend(rng.choice(clusters))
    return sampled


def cluster_bootstrap_gap(
    recent: list[dict],
    earlier: list[dict],
    *,
    repetitions: int,
    seed: int,
) -> dict:
    rng = random.Random(seed)
    gaps = {
        "label_standardized_accuracy": [],
        "overall_accuracy": [],
        "macro_f1": [],
    }
    for _ in range(repetitions):
        recent_sample = stratified_document_sample(recent, FINAL_LABELS, rng)
        earlier_sample = stratified_document_sample(earlier, FINAL_LABELS, rng)
        recent_metrics = classification_metrics(
            recent_sample,
            labels=FINAL_LABELS,
            gold_key="gold_final",
            prediction_key="pipeline_prediction",
        )
        earlier_metrics = classification_metrics(
            earlier_sample,
            labels=FINAL_LABELS,
            gold_key="gold_final",
            prediction_key="pipeline_prediction",
        )
        for key in gaps:
            gaps[key].append(recent_metrics[key] - earlier_metrics[key])

    output = {
        "method": (
            "Label-stratified document-cluster bootstrap; documents are "
            "resampled with replacement within each period and gold label."
        ),
        "repetitions": repetitions,
    }
    for key, values in gaps.items():
        nonpositive = sum(value <= 0 for value in values) / len(values)
        nonnegative = sum(value >= 0 for value in values) / len(values)
        output[key] = {
            "ci_95": [percentile(values, 0.025), percentile(values, 0.975)],
            "bootstrap_two_sided_sign_p": min(
                1.0, 2 * min(nonpositive, nonnegative)
            ),
        }
    return output


def error_stage(row: dict) -> str:
    level1 = (row.get("level1") or {}).get("prediction")
    if level1 != row.get("gold_level1"):
        return "level1"
    if row.get("final_eval") == "1" and (
        row.get("pipeline_prediction") != row.get("gold_final")
    ):
        return "level2"
    return "correct"


def token_usage(rows: list[dict]) -> dict:
    totals: Counter = Counter()
    for row in rows:
        for stage in ("level1", "level2"):
            usage = (row.get(stage) or {}).get("usage") or {}
            for key in (
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "prompt_cache_hit_tokens",
                "prompt_cache_miss_tokens",
            ):
                totals[key] += usage.get(key, 0)
    return dict(totals)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    protocol = json.loads((PROJECT / "config" / "protocol.json").read_text("utf-8"))
    frozen = json.loads(
        (PROJECT / "config" / "frozen_manifest.json").read_text("utf-8")
    )
    raw_path = PROJECT / "results" / "raw" / "formal_predictions.jsonl"
    rows = load_latest(raw_path)
    expected_ids = set(frozen["evaluation_ids"])
    by_id = {row["sample_id"]: row for row in rows}
    missing = sorted(expected_ids - set(by_id))
    extra = sorted(set(by_id) - expected_ids)
    errors = [row for row in rows if row.get("error")]
    if missing or extra or errors:
        raise RuntimeError(
            f"Formal run incomplete: missing={len(missing)} "
            f"extra={len(extra)} errors={len(errors)}"
        )
    rows = [by_id[sample_id] for sample_id in frozen["evaluation_ids"]]

    summary = {
        "protocol": protocol["protocol_name"],
        "model_requested": sorted({row["requested_model"] for row in rows}),
        "models_returned": sorted(
            {
                (row.get(stage) or {}).get("returned_model")
                for row in rows
                for stage in ("level1", "level2")
                if (row.get(stage) or {}).get("returned_model")
            }
        ),
        "system_fingerprints": sorted(
            {
                (row.get(stage) or {}).get("system_fingerprint")
                for row in rows
                for stage in ("level1", "level2")
                if (row.get(stage) or {}).get("system_fingerprint")
            }
        ),
        "evaluation_rows": len(rows),
        "evaluation_documents": len({row["document_id"] for row in rows}),
        "api_error_count": 0,
        "token_usage": token_usage(rows),
        "periods": {},
    }

    for period in ("recent", "earlier"):
        period_rows = [row for row in rows if row["period"] == period]
        final_rows = [row for row in period_rows if row["final_eval"] == "1"]
        level2_rows = [row for row in period_rows if row["level2_eval"] == "1"]
        level1_rows = [row for row in period_rows if row["level1_eval"] == "1"]
        summary["periods"][period] = {
            "year_range": [
                min(int(row["year"]) for row in period_rows),
                max(int(row["year"]) for row in period_rows),
            ],
            "end_to_end_five_class": classification_metrics(
                final_rows,
                labels=FINAL_LABELS,
                gold_key="gold_final",
                prediction_key="pipeline_prediction",
            ),
            "oracle_routed_level2": classification_metrics(
                level2_rows,
                labels=LEVEL2_LABELS,
                gold_key="gold_final",
                prediction_key="oracle_level2_prediction",
            ),
            "task1_level1": classification_metrics(
                level1_rows,
                labels=LEVEL1_LABELS,
                gold_key="gold_level1",
                prediction_key="level1_prediction",
            ),
            "end_to_end_error_stages": dict(
                Counter(error_stage(row) for row in final_rows)
            ),
            "documents": len({row["document_id"] for row in period_rows}),
        }

    recent_final = [
        row
        for row in rows
        if row["period"] == "recent" and row["final_eval"] == "1"
    ]
    earlier_final = [
        row
        for row in rows
        if row["period"] == "earlier" and row["final_eval"] == "1"
    ]
    recent_metrics = summary["periods"]["recent"]["end_to_end_five_class"]
    earlier_metrics = summary["periods"]["earlier"]["end_to_end_five_class"]
    summary["recent_minus_earlier"] = {
        key: recent_metrics[key] - earlier_metrics[key]
        for key in (
            "label_standardized_accuracy",
            "overall_accuracy",
            "macro_f1",
        )
    }
    summary["uncertainty"] = cluster_bootstrap_gap(
        recent_final,
        earlier_final,
        repetitions=10_000,
        seed=protocol["seed"],
    )

    output = PROJECT / "results" / "summary"
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "formal_metrics.json"
    metrics_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    error_rows = [
        {**row, "error_stage": error_stage(row)}
        for row in rows
        if row["final_eval"] == "1"
        and row.get("pipeline_prediction") != row.get("gold_final")
    ]
    fields = [
        "sample_id",
        "period",
        "year",
        "gold_level1",
        "gold_final",
        "pipeline_prediction",
        "oracle_level2_prediction",
        "error_stage",
        "text",
        "title",
        "document_id",
        "category",
        "effectiveness_level",
        "issuing_department",
    ]
    with (output / "formal_errors.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(error_rows)

    lock_path = PROJECT / "config" / "execution_lock.json"
    manifest = {
        "status": "COMPLETE",
        "formal_rows": len(rows),
        "raw_sha256": sha256(raw_path),
        "metrics_sha256": sha256(metrics_path),
        "execution_lock_sha256": sha256(lock_path),
        "frozen_manifest_sha256": sha256(
            PROJECT / "config" / "frozen_manifest.json"
        ),
    }
    (output / "result_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

