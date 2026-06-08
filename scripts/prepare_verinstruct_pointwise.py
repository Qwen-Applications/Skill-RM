#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


DEFAULT_DATASET = "THU-KEG/VerInstruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare VerInstruct pointwise data for the Skill-RM verl recipe.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Hugging Face dataset id or local dataset path.")
    parser.add_argument("--input-jsonl", type=Path, help="Optional local JSONL input for offline conversion/smoke tests.")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--validation-size", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train_examples, val_examples, split_source = load_examples(args)
    if args.max_train_samples > 0:
        train_examples = train_examples[: args.max_train_samples]
    if args.max_val_samples > 0:
        val_examples = val_examples[: args.max_val_samples]

    train_rows = [build_verl_row(example, "train", idx, args.dataset) for idx, example in enumerate(train_examples)]
    val_rows = [build_verl_row(example, "validation", idx, args.dataset) for idx, example in enumerate(val_examples)]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(train_rows, args.output_dir / "train.parquet")
    write_parquet(val_rows, args.output_dir / "validation.parquet")
    metadata = {
        "dataset": args.dataset,
        "input_jsonl": str(args.input_jsonl) if args.input_jsonl else "",
        "split_source": split_source,
        "train_rows": len(train_rows),
        "validation_rows": len(val_rows),
        "validation_size": args.validation_size,
        "seed": args.seed,
        "reference_response_generated": False,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Wrote {len(train_rows)} train rows and {len(val_rows)} validation rows to {args.output_dir}")
    return 0


def load_examples(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    if args.input_jsonl:
        rows = read_jsonl(args.input_jsonl)
        train_rows, val_rows = split_rows(rows, validation_size=args.validation_size, seed=args.seed)
        return train_rows, val_rows, "input_jsonl_deterministic_split"

    try:
        from datasets import DatasetDict, load_dataset
    except ImportError as exc:  # pragma: no cover - dependency guidance.
        raise SystemExit("Install the optional RL dependencies first: pip install -e '.[rl]'") from exc

    dataset = load_dataset(args.dataset)
    if not isinstance(dataset, DatasetDict):
        raise SystemExit(f"Expected DatasetDict from {args.dataset}, got {type(dataset).__name__}.")
    if args.train_split not in dataset:
        raise SystemExit(f"Train split {args.train_split!r} not found. Available splits: {list(dataset.keys())}")
    train = [dict(item) for item in dataset[args.train_split]]
    if args.validation_split in dataset:
        validation = [dict(item) for item in dataset[args.validation_split]]
        return train, validation, "dataset_validation_split"
    if "test" in dataset and args.validation_split == "validation":
        validation = [dict(item) for item in dataset["test"]]
        return train, validation, "dataset_test_split"
    train_rows, val_rows = split_rows(train, validation_size=args.validation_size, seed=args.seed)
    return train_rows, val_rows, "train_deterministic_split"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def split_rows(rows: list[dict[str, Any]], *, validation_size: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not rows:
        return [], []
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    val_count = max(1, int(round(len(rows) * validation_size))) if len(rows) > 1 else 0
    val_indices = set(indices[:val_count])
    train = [row for index, row in enumerate(rows) if index not in val_indices]
    val = [row for index, row in enumerate(rows) if index in val_indices]
    return train, val


def build_verl_row(example: dict[str, Any], split_name: str, idx: int, dataset_name: str) -> dict[str, Any]:
    prompt_value = first_present(example, "prompt", "query", "instruction", "user_prompt")
    if prompt_value is None:
        raise ValueError(f"Example {split_name}-{idx} has no prompt/query/instruction field.")
    prompt_messages = build_messages(prompt_value)
    query_text = build_query_text(prompt_value)
    sample_id = str(first_present(example, "id", "sample_id", "uid") or f"{split_name}-{idx}")
    return {
        "data_source": dataset_name,
        "prompt": prompt_messages,
        "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": {
            "split": split_name,
            "sample_id": sample_id,
            "query": query_text,
            "system_prompt": str(example.get("system_prompt") or ""),
            "history": stringify_history(example.get("history")),
            "original_id": example.get("id"),
            "checkers": example.get("checkers"),
            "functions": example.get("functions"),
        },
    }


def first_present(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return None


def build_messages(prompt_value: Any) -> list[dict[str, str]]:
    if isinstance(prompt_value, list):
        messages = []
        for item in prompt_value:
            if isinstance(item, dict):
                messages.append({"role": str(item.get("role") or "user"), "content": str(item.get("content") or "")})
            else:
                messages.append({"role": "user", "content": str(item)})
        return messages
    return [{"role": "user", "content": str(prompt_value)}]


def build_query_text(prompt_value: Any) -> str:
    if isinstance(prompt_value, str):
        return prompt_value
    return "\n".join(f"[{msg['role']}] {msg['content']}" for msg in build_messages(prompt_value))


def stringify_history(value: Any) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    try:
        from datasets import Dataset
    except ImportError as exc:  # pragma: no cover - dependency guidance.
        raise SystemExit("Install the optional RL dependencies first: pip install -e '.[rl]'") from exc
    Dataset.from_list(rows).to_parquet(str(path))


if __name__ == "__main__":
    raise SystemExit(main())
