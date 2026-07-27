"""Run the frozen two-stage DeepSeek policy-signal classification.

The runner verifies the pre-run lock, loads private local inputs, resumes from
an append-only JSONL file, and applies the official Level-1 then Level-2
routing. API keys are read from the environment and are never written here.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
LEVEL1_LABELS = {"W", "R", "N"}
LEVEL2_LABELS = {"B", "Y", "C", "G"}
PRINT_LOCK = threading.Lock()


def load_dotenv(path: Path) -> None:
    """Load missing variables from a local, git-ignored ``.env`` file."""
    if not path.exists():
        return
    for line in path.read_text("utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def sha256(path: Path) -> str:
    """Return an uppercase SHA-256 digest for a local file."""
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_csv(path: Path) -> list[dict]:
    """Read a UTF-8 CSV into dictionaries."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def build_messages(prompt: str, examples: list[dict], text: str) -> list[dict]:
    """Construct one chat request from a prompt and few-shot examples."""
    messages = [{"role": "system", "content": prompt}]
    for row in examples:
        messages.append(
            {"role": "user", "content": f"待判断政策段落：\n{row['text']}"}
        )
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    {"label": row["gold_label"]}, ensure_ascii=False
                ),
            }
        )
    messages.append({"role": "user", "content": f"待判断政策段落：\n{text}"})
    return messages


def parse_label(content: str | None, allowed: set[str]) -> str | None:
    """Parse a JSON label, returning ``None`` for out-of-scope responses."""
    try:
        parsed = json.loads(content or "")
        label = str(parsed.get("label", "")).strip().upper()
    except (json.JSONDecodeError, AttributeError):
        label = str(content or "").strip().upper()
    return label if label in allowed else None


def call_api(
    *,
    api_key: str,
    protocol: dict,
    messages: list[dict],
    allowed: set[str],
    request_seed: int,
    retries: int = 7,
) -> dict:
    """Call the API with bounded exponential backoff and deterministic jitter."""
    payload = {
        "model": protocol["model"],
        "messages": messages,
        "temperature": protocol["temperature"],
        "max_tokens": protocol["max_tokens"],
        "thinking": {"type": protocol["thinking"]},
        "response_format": {"type": "json_object"},
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    for attempt in range(retries):
        request = urllib.request.Request(
            protocol["base_url"],
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                data = json.loads(response.read().decode("utf-8"))
            choice = data["choices"][0]
            content = choice["message"].get("content")
            return {
                "prediction": parse_label(content, allowed),
                "raw_content": content,
                "finish_reason": choice.get("finish_reason"),
                "returned_model": data.get("model"),
                "system_fingerprint": data.get("system_fingerprint"),
                "usage": data.get("usage") or {},
            }
        except urllib.error.HTTPError as error:
            # Retry only transient failures. Permanent client errors fail fast.
            retryable = error.code in {408, 409, 429, 500, 502, 503, 504}
            if not retryable or attempt + 1 == retries:
                raise RuntimeError(f"HTTP {error.code}") from error
            retry_after = error.headers.get("Retry-After")
            delay = float(retry_after) if retry_after else 2**attempt
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt + 1 == retries:
                raise RuntimeError(type(error).__name__) from error
            delay = 2**attempt
        jitter = random.Random(request_seed + attempt).uniform(0.0, 0.35)
        time.sleep(min(30.0, delay + jitter))
    raise AssertionError("Unreachable")


def route_final(level1: str | None, level2: str | None) -> str | None:
    """Convert two-stage predictions into the final policy-signal label."""
    if level1 == "W":
        return level2 if level2 in LEVEL2_LABELS else None
    if level1 in {"R", "N"}:
        return level1
    return None


def latest_successful_ids(path: Path) -> set[str]:
    """Return completed sample IDs from an append-only prediction file."""
    if not path.exists():
        return set()
    latest: dict[str, dict] = {}
    for line in path.read_text("utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            latest[row["sample_id"]] = row
    return {
        sample_id
        for sample_id, row in latest.items()
        if not row.get("error")
    }


def run_one(
    row: dict,
    *,
    api_key: str,
    protocol: dict,
    prompts: dict,
    level1_examples: list[dict],
    level2_examples: list[dict],
) -> dict:
    """Classify one row and retain both stages for later decomposition."""
    result = {
        **row,
        "requested_model": protocol["model"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "level1": None,
        "level2": None,
        "pipeline_prediction": None,
        "oracle_level2_prediction": None,
        "error": None,
    }
    try:
        seed_base = int(row["sample_id"], 16) % 2_000_000_000
        level1 = call_api(
            api_key=api_key,
            protocol=protocol,
            messages=build_messages(
                prompts["level1"], level1_examples, row["text"]
            ),
            allowed=LEVEL1_LABELS,
            request_seed=seed_base,
        )
        result["level1"] = level1
        # All gold-W rows receive Level 2 for oracle analysis. Other rows
        # receive it only when the model itself routes them through W.
        needs_level2 = row["level2_eval"] == "1" or (
            row["final_eval"] == "1" and level1["prediction"] == "W"
        )
        if needs_level2:
            level2 = call_api(
                api_key=api_key,
                protocol=protocol,
                messages=build_messages(
                    prompts["level2"], level2_examples, row["text"]
                ),
                allowed=LEVEL2_LABELS,
                request_seed=seed_base + 1,
            )
            result["level2"] = level2
            if row["level2_eval"] == "1":
                result["oracle_level2_prediction"] = level2["prediction"]
        if row["final_eval"] == "1":
            result["pipeline_prediction"] = route_final(
                level1["prediction"],
                (result["level2"] or {}).get("prediction"),
            )
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def main() -> None:
    """Verify the lock, resume pending rows, and stream predictions to JSONL."""
    audit = json.loads(
        (PROJECT / "data" / "processed" / "audit_report.json").read_text("utf-8")
    )
    if audit.get("status") != "PASS":
        raise RuntimeError("Frozen data audit has not passed.")
    lock = json.loads(
        (PROJECT / "config" / "execution_lock.json").read_text("utf-8")
    )
    if lock.get("status") != "LOCKED_BEFORE_PREDICTIONS":
        raise RuntimeError("Execution lock is missing.")
    for relative, expected in lock["sha256"].items():
        if sha256(PROJECT / relative) != expected:
            raise RuntimeError(f"Locked file changed: {relative}")

    load_dotenv(PROJECT / ".env")
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is missing.")

    protocol = json.loads((PROJECT / "config" / "protocol.json").read_text("utf-8"))
    prompts = json.loads((PROJECT / "config" / "prompts.json").read_text("utf-8"))
    data = PROJECT / "data" / "processed"
    level1_examples = read_csv(data / "level1_few_shot.csv")
    level2_examples = read_csv(data / "level2_few_shot.csv")
    rows = read_csv(data / "evaluation_census.csv")

    output = PROJECT / "results" / "raw" / "formal_predictions.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    # Successful rows are skipped on restart; failed rows remain retryable.
    done = latest_successful_ids(output)
    pending = [row for row in rows if row["sample_id"] not in done]
    print(
        f"formal_run total={len(rows)} complete={len(done)} "
        f"remaining={len(pending)} workers={protocol['concurrency']}",
        flush=True,
    )
    if not pending:
        return

    finished = 0
    errors = 0
    started = time.time()
    with output.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=protocol["concurrency"]) as executor:
            futures = {
                executor.submit(
                    run_one,
                    row,
                    api_key=api_key,
                    protocol=protocol,
                    prompts=prompts,
                    level1_examples=level1_examples,
                    level2_examples=level2_examples,
                ): row["sample_id"]
                for row in pending
            }
            for future in as_completed(futures):
                result = future.result()
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                finished += 1
                errors += bool(result.get("error"))
                if finished % 25 == 0 or finished == len(pending):
                    elapsed = max(time.time() - started, 0.001)
                    rate = finished / elapsed
                    remaining_seconds = (len(pending) - finished) / max(rate, 0.001)
                    with PRINT_LOCK:
                        print(
                            f"progress={finished}/{len(pending)} "
                            f"errors={errors} rate={rate:.2f}/s "
                            f"eta_min={remaining_seconds / 60:.1f}",
                            flush=True,
                        )


if __name__ == "__main__":
    main()

