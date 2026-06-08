from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


DEFAULT_HF_DATASET = "allenai/reward-bench-2"
DEFAULT_HF_SPLIT = "test"
LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


@dataclass(frozen=True)
class RB2Example:
    sample_id: str
    prompt: str
    responses: dict[str, str]
    chosen_label: str
    subset: str | None = None
    visible_metadata: dict[str, Any] = field(default_factory=dict)
    hidden_metadata: dict[str, Any] = field(default_factory=dict)
    source_record: dict[str, Any] = field(default_factory=dict)

    def labels(self) -> list[str]:
        return list(self.responses.keys())

    def to_agent_payload(self) -> dict[str, Any]:
        """Return only fields the agent may see."""
        return {
            "id": self.sample_id,
            "prompt": self.prompt,
            "responses": self.responses,
            "visible_metadata": self.visible_metadata,
        }


class RB2FormatError(ValueError):
    pass


def load_rb2_examples(
    data_source: str,
    *,
    limit: int | None = None,
    seed: int = 0,
    expose_subset: bool = False,
    include_ties: bool = False,
) -> list[RB2Example]:
    records = iter_rb2_records(data_source)
    examples: list[RB2Example] = []
    for record in records:
        example = normalize_rb2_record(
            record,
            seed=seed,
            expose_subset=expose_subset,
            include_ties=include_ties,
        )
        if example is None:
            continue
        examples.append(example)
        if limit is not None and len(examples) >= limit:
            break
    return examples


def iter_rb2_records(data_source: str) -> Iterable[dict[str, Any]]:
    if data_source.startswith("hf://"):
        dataset_ref = data_source.removeprefix("hf://")
        dataset_name, split = _parse_hf_ref(dataset_ref)
        yield from _iter_hf_dataset(dataset_name, split)
        return

    path = Path(data_source)
    if path.exists():
        yield from _iter_jsonl(path)
        return

    if "/" in data_source and not data_source.endswith(".jsonl"):
        dataset_name, split = _parse_hf_ref(data_source)
        yield from _iter_hf_dataset(dataset_name, split)
        return

    raise RB2FormatError(f"Unsupported RBv2 data source: {data_source}")


def normalize_rb2_record(
    record: dict[str, Any],
    *,
    seed: int = 0,
    expose_subset: bool = False,
    include_ties: bool = False,
) -> RB2Example | None:
    subset = record.get("subset")
    if _is_ties_subset(subset) and not include_ties:
        return None

    chosen = _as_list(record.get("chosen"))
    rejected = _as_list(record.get("rejected"))
    if len(chosen) != 1:
        if include_ties:
            raise RB2FormatError("Ties scoring is not supported by this RB2 example loader.")
        raise RB2FormatError(f"Expected exactly one chosen response, got {len(chosen)}.")

    responses = chosen + rejected
    if len(responses) != 4:
        raise RB2FormatError(
            f"Expected 4 responses for non-Ties RBv2 sample, got {len(responses)}."
        )

    sample_id = str(record.get("id", record.get("sample_id", "")))
    if not sample_id:
        sample_id = f"row-{abs(hash(json.dumps(record, sort_keys=True, default=str)))}"

    order = list(range(4))
    random.Random(f"{seed}:{sample_id}").shuffle(order)
    labels = LABELS[:4]
    shuffled = {labels[new_idx]: responses[old_idx] for new_idx, old_idx in enumerate(order)}
    chosen_label = labels[order.index(0)]

    visible_metadata: dict[str, Any] = {}
    if expose_subset and subset is not None:
        visible_metadata["subset"] = subset

    hidden_metadata = {
        "subset": subset,
        "chosen_original_index": 0,
        "shuffle_order": order,
    }

    return RB2Example(
        sample_id=sample_id,
        prompt=str(record.get("prompt", "")),
        responses=shuffled,
        chosen_label=chosen_label,
        subset=subset,
        visible_metadata=visible_metadata,
        hidden_metadata=hidden_metadata,
        source_record=record,
    )


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise RB2FormatError(f"{path}:{line_number}: invalid JSONL") from exc


def _iter_hf_dataset(dataset_name: str, split: str) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RB2FormatError(
            "Loading RBv2 from Hugging Face requires the `datasets` package."
        ) from exc

    dataset = load_dataset(dataset_name, split=split)
    for row in dataset:
        yield dict(row)


def _parse_hf_ref(ref: str) -> tuple[str, str]:
    if ":" in ref:
        dataset_name, split = ref.rsplit(":", 1)
        return dataset_name, split
    if "/" in ref and ref.count("/") >= 2:
        parts = ref.split("/")
        dataset_name = "/".join(parts[:2])
        split = "/".join(parts[2:]) or DEFAULT_HF_SPLIT
        return dataset_name, split
    return ref or DEFAULT_HF_DATASET, DEFAULT_HF_SPLIT


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    return [str(value)]


def _is_ties_subset(subset: Any) -> bool:
    return str(subset or "").strip().lower() in {"tie", "ties"}
