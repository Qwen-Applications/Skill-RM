from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from skillrm.common.io import read_jsonl


def response_text(response: dict[str, Any]) -> str:
    for key in ("content", "response", "text", "completion", "output"):
        value = response.get(key)
        if isinstance(value, str):
            return value
    messages = response.get("messages")
    if isinstance(messages, list):
        return "\n".join(str(m.get("content", "")) for m in messages if isinstance(m, dict))
    return json.dumps(response, ensure_ascii=False)


def response_score(response: dict[str, Any]) -> float | None:
    metadata = response.get("metadata")
    if isinstance(metadata, dict):
        value = metadata.get("score")
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    for key in ("score", "pass", "correct"):
        value = response.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def get_prompt(row: dict[str, Any]) -> str:
    for key in ("query", "prompt", "question", "instruction"):
        value = row.get(key)
        if isinstance(value, str):
            return value
    messages = row.get("messages")
    if isinstance(messages, list):
        return "\n".join(str(m.get("content", "")) for m in messages if isinstance(m, dict))
    return json.dumps({k: v for k, v in row.items() if k != "responses"}, ensure_ascii=False)


def load_dataset_records(data_dir: Path, dataset: str, generator: str) -> list[dict[str, Any]]:
    path = data_dir / f"{dataset}_{generator}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing JETTS data file: {path}")
    records = []
    for idx, row in enumerate(read_jsonl(path)):
        responses = row.get("responses")
        if not isinstance(responses, list) or not responses:
            continue
        sample_id = str(row.get("id") or row.get("sample_id") or row.get("task_id") or idx)
        records.append(
            {
                "sample_id": sample_id,
                "dataset": dataset,
                "generator": generator,
                "prompt": get_prompt(row),
                "responses": responses,
                "source_index": idx,
                "raw_metadata": {k: v for k, v in row.items() if k not in {"responses", "query", "prompt", "question", "instruction"}},
            }
        )
    return records
