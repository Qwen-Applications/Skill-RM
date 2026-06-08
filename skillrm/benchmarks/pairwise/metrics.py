from __future__ import annotations

from collections import defaultdict
from typing import Any

from ...common.stats import mean, safe_div
from .data import MODE_BY_PAIR, PAIR_LABELS, is_judgebench, is_rmbench


def compute_openrs_metrics(
    tasks: list[Any],
    rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    task_metrics = compute_pairwise_task_metrics(tasks, rows)
    metrics: dict[str, Any] = {
        "benchmark": config.get("benchmark"),
        "evaluation_mode": config.get("evaluation_mode"),
        "n": len(tasks),
        "completed": len(rows),
        "missing": len(tasks) - len({row["sample_id"] for row in rows}),
        "task_level": task_metrics,
        "skill_usage": skill_usage_from_openrs_rows(rows),
        "mean_latency_sec": mean([float(row["latency_sec"]) for row in rows if row.get("latency_sec")]),
        "endpoints": sorted({row.get("endpoint") for row in rows if row.get("endpoint")}),
    }
    benchmark = str(config.get("benchmark", "")).lower()
    if is_rmbench(benchmark):
        metrics["rmbench"] = compute_rmbench_metrics(tasks, rows)
        metrics["overall"] = metrics["rmbench"]["global"]["overall"]
    elif is_judgebench(benchmark):
        metrics["judgebench"] = compute_judgebench_metrics(tasks, rows)
        metrics["overall"] = metrics["judgebench"]["overall"]
        metrics["by_query_type"] = metrics["judgebench"]["by_query_type"]
    else:
        metrics["overall"] = task_metrics["overall"]
        metrics["by_query_type"] = task_metrics["by_group"]
    return metrics


def compute_pairwise_task_metrics(tasks: list[Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {row["sample_id"]: row for row in rows}
    overall = new_counter()
    by_group: dict[str, dict[str, int]] = defaultdict(new_counter)
    for task in tasks:
        row = by_id.get(task.task_id)
        group = task.query_type or task.domain or "unknown"
        add_pairwise_outcome(overall, task, row)
        add_pairwise_outcome(by_group[str(group)], task, row)
    return {
        "overall": finalize_counter(overall),
        "by_group": {group: finalize_counter(counter) for group, counter in sorted(by_group.items())},
    }


def new_counter() -> dict[str, int]:
    return {"acc_num": 0, "err_num": 0, "same_num": 0, "invalid_num": 0, "all_num": 0}


def add_pairwise_outcome(counter: dict[str, int], task: Any, row: dict[str, Any] | None) -> None:
    counter["all_num"] += 1
    if row is None or not row.get("valid"):
        counter["invalid_num"] += 1
        return
    winner = row.get("predicted_label")
    if winner == task.gold_label:
        counter["acc_num"] += 1
    elif winner in {"A", "B"}:
        counter["err_num"] += 1
    elif winner in {"Tie", "Abstain"}:
        counter["same_num"] += 1
    else:
        counter["invalid_num"] += 1


def finalize_counter(counter: dict[str, int]) -> dict[str, Any]:
    valid_decisive = counter["acc_num"] + counter["err_num"]
    all_num = counter["all_num"]
    return {
        **counter,
        "acc_rate": safe_div(counter["acc_num"], valid_decisive),
        "raw_accuracy": safe_div(counter["acc_num"], all_num),
        "same_rate": safe_div(counter["same_num"], all_num),
        "invalid_rate": safe_div(counter["invalid_num"], all_num),
    }


def compute_judgebench_metrics(tasks: list[Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_task = {row["sample_id"]: row for row in rows}
    grouped_tasks: dict[str, list[Any]] = defaultdict(list)
    sample_group: dict[str, str] = {}
    for task in tasks:
        grouped_tasks[task.sample_id].append(task)
        sample_group[task.sample_id] = str(task.query_type or task.domain or "unknown")

    overall = judgebench_counter()
    by_group: dict[str, dict[str, int]] = defaultdict(judgebench_counter)
    sample_results: dict[str, dict[str, Any]] = {}
    for sample_id, sample_tasks in grouped_tasks.items():
        score = 0
        valid_vote_count = 0
        raw: dict[str, str] = {}
        for task in sorted(sample_tasks, key=lambda item: item.order or 0):
            row = by_task.get(task.task_id)
            if row is None or not row.get("valid"):
                raw[f"order{task.order}"] = "invalid"
                continue
            pred = str(row.get("predicted_label"))
            raw[f"order{task.order}"] = pred
            if pred == task.gold_label:
                score += 1
                valid_vote_count += 1
            elif pred in {"A", "B"}:
                score -= 1
                valid_vote_count += 1
            elif pred in {"Tie", "Abstain", "A=B", "tie", "same"}:
                valid_vote_count += 1
        if valid_vote_count == 0:
            result = "invalid"
        elif score > 0:
            result = "correct"
        elif score < 0:
            result = "wrong"
        else:
            result = "same"
        group = sample_group.get(sample_id, "unknown")
        judgebench_add(overall, result)
        judgebench_add(by_group[group], result)
        sample_results[sample_id] = {
            "result": result,
            "group": group,
            "orders": raw,
            "vote_score": score,
            "valid_vote_count": valid_vote_count,
        }
    return {
        "overall": finalize_judgebench_counter(overall),
        "by_query_type": {group: finalize_judgebench_counter(counter) for group, counter in sorted(by_group.items())},
        "sample_results": sample_results,
    }


def judgebench_counter() -> dict[str, int]:
    return {"acc_num": 0, "err_num": 0, "same_num": 0, "invalid_num": 0, "all_num": 0}


def judgebench_add(counter: dict[str, int], result: str) -> None:
    counter["all_num"] += 1
    if result == "correct":
        counter["acc_num"] += 1
    elif result == "wrong":
        counter["err_num"] += 1
    elif result == "same":
        counter["same_num"] += 1
    else:
        counter["invalid_num"] += 1


def finalize_judgebench_counter(counter: dict[str, int]) -> dict[str, Any]:
    all_num = counter["all_num"]
    return {
        **counter,
        "acc_rate": safe_div(counter["acc_num"], all_num),
        "err_rate": safe_div(counter["err_num"], all_num),
        "same_rate": safe_div(counter["same_num"], all_num),
        "invalid_rate": safe_div(counter["invalid_num"], all_num),
    }


def compute_rmbench_metrics(tasks: list[Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_task = {row["sample_id"]: row for row in rows}
    sample_domain: dict[str, str] = {}
    for task in tasks:
        sample_domain[task.sample_id] = task.domain or "unknown"

    global_pair = {pair: rm_counter() for pair in PAIR_LABELS}
    global_mode = {mode: rm_counter() for mode in ["easy", "normal", "hard"]}
    overall = rm_counter()
    domain_pair: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: {pair: rm_counter() for pair in PAIR_LABELS})
    domain_mode: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: {mode: rm_counter() for mode in ["easy", "normal", "hard"]})

    pair_results: dict[str, dict[str, Any]] = {}
    for task in tasks:
        if not task.pair:
            continue
        sample_id = task.sample_id
        pair = task.pair
        row = by_task.get(task.task_id)
        result = aggregate_rm_pair(row)
        domain = sample_domain.get(sample_id, "unknown")
        mode = MODE_BY_PAIR[pair]
        pair_results[f"{sample_id}::{pair}"] = {
            "sample_id": sample_id,
            "domain": domain,
            "pair": pair,
            "mode": mode,
            "winner": row.get("predicted_label") if row else None,
            "result": result,
        }
        rm_add(overall, result)
        rm_add(global_pair[pair], result)
        rm_add(global_mode[mode], result)
        rm_add(domain_pair[domain][pair], result)
        rm_add(domain_mode[domain][mode], result)

    return {
        "global": {
            "overall": finalize_rm_counter(overall),
            "by_pair": {pair: finalize_rm_counter(global_pair[pair]) for pair in PAIR_LABELS},
            "by_mode": {mode: finalize_rm_counter(global_mode[mode]) for mode in ["easy", "normal", "hard"]},
        },
        "by_domain": {
            domain: {
                "by_pair": {pair: finalize_rm_counter(counters[pair]) for pair in PAIR_LABELS},
                "by_mode": {mode: finalize_rm_counter(domain_mode[domain][mode]) for mode in ["easy", "normal", "hard"]},
            }
            for domain, counters in sorted(domain_pair.items())
        },
    }


def aggregate_rm_pair(row: dict[str, Any] | None) -> str:
    if not row or not row.get("valid"):
        return "error"
    winner = row.get("predicted_label")
    if winner == "A":
        return "win"
    if winner == "B":
        return "lose"
    return "tie"


def rm_counter() -> dict[str, int]:
    return {"win": 0, "tie": 0, "lose": 0, "error": 0}


def rm_add(counter: dict[str, int], result: str) -> None:
    if result == "win":
        counter["win"] += 1
    elif result == "lose":
        counter["lose"] += 1
    elif result == "tie":
        counter["tie"] += 1
    else:
        counter["error"] += 1


def finalize_rm_counter(counter: dict[str, int]) -> dict[str, Any]:
    total = sum(counter.values())
    decisive = counter["win"] + counter["lose"]
    return {
        **counter,
        "total": total,
        "win_rate": safe_div(counter["win"], total),
        "tie_rate": safe_div(counter["tie"], total),
        "lose_rate": safe_div(counter["lose"], total),
        "error_rate": safe_div(counter["error"], total),
        "net_win_rate": safe_div(counter["win"], decisive),
    }


def skill_usage_from_openrs_rows(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    skill_rows = [row for row in rows if row.get("mode") == "self_select_skill_pairwise"]
    if not skill_rows:
        return None
    resource_counts: dict[str, int] = defaultdict(int)
    for row in skill_rows:
        for resource in row.get("resources_viewed") or row.get("resources_loaded") or []:
            resource_counts[str(resource)] += 1
    return {
        "n_skill_rows": len(skill_rows),
        "skill_path": skill_rows[0].get("skill_path"),
        "skill_package_sha256": skill_rows[0].get("skill_package_sha256"),
        "skill_loading_mode": skill_rows[0].get("skill_loading_mode"),
        "resources_loaded": dict(sorted(resource_counts.items())),
        "skill_trigger_rate": safe_div(sum(1 for row in skill_rows if row.get("skill_triggered")), len(skill_rows)),
        "mean_agent_step_count": mean([float(row.get("agent_step_count", 0)) for row in skill_rows]),
        "mean_tool_call_count": mean([float(row.get("tool_call_count", 0)) for row in skill_rows]),
        "mean_python_sandbox_call_count": mean([float(row.get("python_sandbox_call_count", 0)) for row in skill_rows]),
        "mean_resource_view_count": mean([float(row.get("resource_view_count", 0)) for row in skill_rows]),
        "position_swap_audit_rate": safe_div(sum(1 for row in skill_rows if row.get("position_swap_audit")), len(skill_rows)),
        "position_swap_disagreement_rate": safe_div(
            sum(1 for row in skill_rows if row.get("position_swap_audit") and row.get("position_swap_consistent") is False),
            len(skill_rows),
        ),
        "position_swap_override_rate": safe_div(sum(1 for row in skill_rows if row.get("position_swap_override")), len(skill_rows)),
    }
