from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any


PAIR_LABELS = ["aa", "ab", "ac", "ba", "bb", "bc", "ca", "cb", "cc"]
MODE_BY_PAIR = {
    "aa": "normal",
    "bb": "normal",
    "cc": "normal",
    "ab": "hard",
    "ac": "hard",
    "bc": "hard",
    "ca": "easy",
    "cb": "easy",
    "ba": "easy",
}
VARIANT_MAP = {"a": 0, "b": 1, "c": 2}


@dataclass(frozen=True)
class OpenRSPairTask:
    task_id: str
    benchmark: str
    sample_id: str
    prompt: str
    responses: dict[str, str]
    gold_label: str
    query_type: str | None = None
    domain: str | None = None
    pair: str | None = None
    order: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    sample_resources: dict[str, Any] = field(default_factory=dict)


def is_judgebench(benchmark: str) -> bool:
    return str(benchmark or "").lower().startswith("judgebench")


def is_rmbench(benchmark: str) -> bool:
    return str(benchmark or "").lower() in {"rmbench", "rm-bench", "rm_bench"}


def load_openrs_tasks(config: dict[str, Any]) -> list[OpenRSPairTask]:
    benchmark = str(config.get("benchmark") or "").lower()
    raw_limit = config.get("limit")
    limit = None if raw_limit in (None, 0, "0") else int(raw_limit)
    data_sources = configured_openrs_data_sources(config)
    if len(data_sources) > 1:
        tasks: list[OpenRSPairTask] = []
        remaining = limit
        for source_path, source_name in data_sources:
            source_limit = remaining
            loaded = load_openrs_tasks_from_source(config, benchmark=benchmark, data_source=source_path, limit=source_limit)
            loaded = [prefix_openrs_task_source(task, source_name) for task in loaded]
            tasks.extend(loaded)
            if remaining is not None:
                remaining -= len({task.sample_id for task in loaded})
                if remaining <= 0:
                    break
        return tasks
    data_source = data_sources[0][0]
    return load_openrs_tasks_from_source(config, benchmark=benchmark, data_source=data_source, limit=limit)


def configured_openrs_data_sources(config: dict[str, Any]) -> list[tuple[str, str]]:
    raw = config.get("data_sources")
    sources: list[tuple[str, str]] = []
    if raw:
        raw_items = raw if isinstance(raw, list) else [item.strip() for item in str(raw).split(",") if item.strip()]
        for index, item in enumerate(raw_items):
            if isinstance(item, dict):
                path = str(item.get("path") or item.get("data_source") or "").strip()
                name = str(item.get("source") or item.get("name") or Path(path).stem or index).strip()
            else:
                path = str(item).strip()
                name = Path(path).stem or str(index)
            if not path:
                continue
            sources.append((path, sanitize_source_prefix(name)))
    if sources:
        return sources
    return [(str(config["data_source"]), sanitize_source_prefix(Path(str(config["data_source"])).stem))]


def sanitize_source_prefix(value: str) -> str:
    text = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value or "source")).strip("_")
    return text or "source"


def prefix_openrs_task_source(task: OpenRSPairTask, source: str) -> OpenRSPairTask:
    prefix = sanitize_source_prefix(source)
    sample_id = str(task.sample_id)
    task_id = str(task.task_id)
    if sample_id.startswith(f"{prefix}::"):
        return task
    metadata = dict(task.metadata or {})
    metadata["data_source"] = prefix
    resources = dict(task.sample_resources or {})
    resources.setdefault("source", prefix)
    return replace(
        task,
        task_id=f"{prefix}::{task_id}",
        sample_id=f"{prefix}::{sample_id}",
        metadata=metadata,
        sample_resources=resources,
    )


def load_openrs_tasks_from_source(
    config: dict[str, Any],
    *,
    benchmark: str,
    data_source: str,
    limit: int | None,
) -> list[OpenRSPairTask]:
    if benchmark in {"judgebench_gpt", "judgebench_claude", "judgebench"} or benchmark.startswith("judgebench_"):
        return load_judgebench_tasks(data_source, benchmark=benchmark, limit=limit)
    if benchmark in {"rmbench", "rm-bench", "rm_bench"}:
        return load_rmbench_tasks(data_source, benchmark="rmbench", limit=limit)
    raise ValueError(f"Unsupported OpenRS benchmark: {benchmark}")


def nonempty_resource_value(value: Any) -> bool:
    return value not in (None, "", [], {})


def openrs_visible_sample_resources(record: dict[str, Any]) -> dict[str, Any]:
    """Keep only OpenRS resource fields that are allowed in operational mode."""
    resources: dict[str, Any] = {}
    for key in (
        "ground_truth",
        "reference",
        "answer",
        "expected_answer",
        "correct_answer",
        "gt",
        "constraints",
        "check_list",
        "checklist",
        "criteria",
        "rubric",
        "verifier_signal",
        "additional_metadata",
        "gt_explanation",
        "gt_question_type",
    ):
        value = record.get(key)
        if nonempty_resource_value(value):
            resources[key] = value
    return resources


def load_judgebench_tasks(
    path: str,
    *,
    benchmark: str,
    limit: int | None,
) -> list[OpenRSPairTask]:
    tasks: list[OpenRSPairTask] = []
    samples_seen = 0
    for record in iter_jsonl(path):
        if record.get("label_error") is True:
            continue
        if limit is not None and samples_seen >= int(limit):
            break
        prompt = str(record.get("prompt") or record.get("input") or record.get("instruction") or "")
        sample_id = str(record.get("question_id") or record.get("id") or samples_seen)
        sample_resources = openrs_visible_sample_resources(record)

        if is_judgebench(benchmark) and ("response_A" in record or "response_B" in record):
            response_a = str(record.get("response_A") or "")
            response_b = str(record.get("response_B") or "")
            gold_original = normalize_judgebench_label(record.get("label"))
            if not response_a or not response_b or gold_original is None:
                continue
            samples_seen += 1
            for order, responses, gold in (
                (1, {"A": response_a, "B": response_b}, gold_original),
                (2, {"A": response_b, "B": response_a}, "B" if gold_original == "A" else "A"),
            ):
                tasks.append(
                    OpenRSPairTask(
                        task_id=f"{sample_id}::order{order}",
                        benchmark=benchmark,
                        sample_id=sample_id,
                        prompt=prompt or str(record.get("question") or ""),
                        responses=responses,
                        gold_label=gold,
                        query_type=record.get("query_type") or record.get("source"),
                        domain=record.get("query_type") or record.get("source"),
                        order=order,
                        metadata={"official_schema": "judgebench", "official_metric": "judgebench_reverse_order"},
                        sample_resources=sample_resources,
                    )
                )
            continue

        chosen = str(record.get("chosen") or "")
        rejected = str(record.get("rejected") or "")
        if not chosen or not rejected:
            continue
        samples_seen += 1
        for order, responses, gold in (
            (1, {"A": chosen, "B": rejected}, "A"),
            (2, {"A": rejected, "B": chosen}, "B"),
        ):
            tasks.append(
                OpenRSPairTask(
                    task_id=f"{sample_id}::order{order}",
                    benchmark=benchmark,
                    sample_id=sample_id,
                    prompt=prompt,
                    responses=responses,
                    gold_label=gold,
                    query_type=record.get("query_type"),
                    domain=record.get("query_type"),
                    order=order,
                    metadata={"official_metric": "judgebench_reverse_order"},
                    sample_resources=sample_resources,
                )
            )
        continue
    return tasks


def normalize_judgebench_label(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    compact = re.sub(r"\s+", "", value.strip().upper())
    if compact in {"A>B", "A>>B"}:
        return "A"
    if compact in {"B>A", "B>>A"}:
        return "B"
    return None


def load_rmbench_tasks(path: str, *, benchmark: str, limit: int | None) -> list[OpenRSPairTask]:
    tasks: list[OpenRSPairTask] = []
    samples_seen = 0
    for record in iter_jsonl(path):
        if limit is not None and samples_seen >= int(limit):
            break
        sample_id = str(record.get("id") or samples_seen)
        prompt = str(record.get("prompt") or "")
        domain = str(record.get("domain") or record.get("query_type") or "general")
        sample_resources = openrs_visible_sample_resources(record)
        chosen = record.get("chosen") or []
        rejected = record.get("rejected") or []
        if isinstance(chosen, str):
            chosen = [chosen]
        if isinstance(rejected, str):
            rejected = [rejected]
        if not chosen or not rejected:
            continue
        chosen = [str(item) for item in chosen]
        rejected = [str(item) for item in rejected]
        while len(chosen) < 3:
            chosen.append(chosen[-1])
        while len(rejected) < 3:
            rejected.append(rejected[-1])
        samples_seen += 1
        for pair in PAIR_LABELS:
            ci = VARIANT_MAP[pair[0]]
            ri = VARIANT_MAP[pair[1]]
            tasks.append(
                OpenRSPairTask(
                    task_id=f"{sample_id}::{pair}",
                    benchmark=benchmark,
                    sample_id=sample_id,
                    prompt=prompt,
                    responses={"A": chosen[ci], "B": rejected[ri]},
                    gold_label="A",
                    query_type=domain,
                    domain=domain,
                    pair=pair,
                    order=1,
                    metadata={"official_metric": "rmbench_response_matrix"},
                    sample_resources=sample_resources,
                )
            )
    return tasks


def iter_jsonl(path: str):
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def summarize_tasks(tasks: list[OpenRSPairTask]) -> dict[str, Any]:
    by_group: dict[str, int] = defaultdict(int)
    for task in tasks:
        by_group[str(task.query_type or task.domain or "unknown")] += 1
    return {"n": len(tasks), "by_group": dict(sorted(by_group.items()))}
