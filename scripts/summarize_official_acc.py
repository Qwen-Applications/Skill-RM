from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


BENCHES = ("rewardbench2", "rmbench", "judgebench")
MAIN_METHODS = ("baseline", "skill_fair", "skill_operational")
ABLATION_METHODS = ("fair_flat_prompt", "flat_prompt", "tool_only")


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def rewardbench2_acc(run_dir: Path) -> dict[str, Any]:
    metrics = load_json(run_dir / "metrics.json") or {}
    return {
        "acc": metrics.get("official_leaderboard_average"),
        "n": metrics.get("n"),
        "counts": f"n={metrics.get('n', 'NA')} invalid_non_ties={metrics.get('invalid_rate_non_ties', 'NA')}",
    }


def pairwise_full_acc(run_dir: Path) -> dict[str, Any]:
    rows = load_rows(run_dir / "predictions.jsonl")
    acc = err = same = invalid = 0
    for row in rows:
        if not row.get("valid"):
            invalid += 1
            continue
        pred = row.get("predicted_label")
        gold = row.get("gold_label")
        if pred == gold:
            acc += 1
        elif pred in {"A", "B"}:
            err += 1
        elif pred in {"Tie", "Abstain", "A=B", "tie", "same"}:
            same += 1
        else:
            invalid += 1
    n = len(rows)
    return {
        "acc": acc / n if n else None,
        "n": n,
        "counts": f"acc={acc} err={err} same={same} invalid={invalid} all={n}",
    }


def judgebench_official_acc(run_dir: Path) -> dict[str, Any]:
    rows = load_rows(run_dir / "predictions.jsonl")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        sample_id = row.get("original_sample_id") or str(row.get("sample_id", "")).split("::")[0]
        grouped.setdefault(str(sample_id), []).append(row)

    acc = err = same = invalid = 0
    for sample_rows in grouped.values():
        score = 0
        has_valid_vote = False
        for row in sample_rows:
            if not row.get("valid"):
                continue
            pred = row.get("predicted_label")
            gold = row.get("gold_label")
            if pred == gold:
                score += 1
                has_valid_vote = True
            elif pred in {"A", "B"}:
                score -= 1
                has_valid_vote = True
            elif pred in {"Tie", "Abstain", "A=B", "tie", "same"}:
                has_valid_vote = True
        if not has_valid_vote:
            invalid += 1
        elif score > 0:
            acc += 1
        elif score < 0:
            err += 1
        else:
            same += 1
    n = len(grouped)
    return {
        "acc": acc / n if n else None,
        "n": n,
        "counts": f"acc={acc} err={err} same={same} invalid={invalid} all={n}",
    }


def metric_for(root: Path, method: str, bench: str) -> dict[str, Any]:
    run_dir = root / method / bench
    if bench == "rewardbench2":
        return rewardbench2_acc(run_dir)
    if bench == "judgebench":
        return judgebench_official_acc(run_dir)
    return pairwise_full_acc(run_dir)


def infer_methods(root: Path) -> tuple[str, ...]:
    main = [method for method in MAIN_METHODS if (root / method).is_dir()]
    ablations = [method for method in ABLATION_METHODS if (root / method).is_dir()]
    if main or ablations:
        return tuple(main + ablations)
    if not root.exists():
        return MAIN_METHODS
    discovered = []
    for path in sorted(root.iterdir()):
        if path.is_dir() and any((path / bench).exists() for bench in BENCHES):
            discovered.append(path.name)
    return tuple(discovered) if discovered else MAIN_METHODS


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs")
    methods = infer_methods(root)
    print("# Skill-RM Official / Full-Set Accuracy Summary")
    print()
    print(f"Run root: `{root}`")
    print()
    print("Metric convention:")
    print("- RewardBench2: `official_leaderboard_average`.")
    print("- RM-Bench: `win / total`; tie and error are counted as not correct.")
    print("- JudgeBench: reverse-order aggregation; a sample is correct if correct votes outnumber wrong votes, so win+tie counts as correct.")
    print()
    print("```text")
    print(f"{'Method':<18} {'RewardBench2':>12} {'RM-Bench':>10} {'JudgeBench':>12} {'Avg3':>10}")
    for method in methods:
        vals = [metric_for(root, method, bench)["acc"] for bench in BENCHES]
        present = [value for value in vals if value is not None]
        avg = sum(present) / len(present) if present else None
        print(f"{method:<18} {fmt(vals[0]):>12} {fmt(vals[1]):>10} {fmt(vals[2]):>12} {fmt(avg):>10}")
    print("```")
    print()
    print("## Counts")
    print()
    print("```text")
    for method in methods:
        for bench in BENCHES:
            info = metric_for(root, method, bench)
            print(f"{method:<18} {bench:<12} {info['counts']}")
    print("```")


if __name__ == "__main__":
    main()
