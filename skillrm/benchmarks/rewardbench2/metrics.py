from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from ...common.stats import mean, safe_div


RB2_OFFICIAL_DOMAIN_ORDER = ["Factuality", "Precise IF", "Math", "Safety", "Focus", "Ties"]


def official_metrics_from_rows(records: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {row["sample_id"]: row for row in rows}
    subset_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing = 0
    for record in records:
        row = by_id.get(str(record["id"]))
        if row is None:
            missing += 1
            continue
        subset_rows[str(record.get("subset") or "unknown")].append(row)

    by_subset: dict[str, dict[str, Any]] = {}
    score_sum = 0.0
    score_n = 0
    invalid = 0
    for subset, subset_items in sorted(subset_rows.items()):
        if subset.lower() == "ties":
            ties_score = compute_official_ties_score(subset_items)
            by_subset[subset] = {
                "n": len(subset_items),
                "official_ties_score": ties_score,
                "accuracy": ties_score,
                "invalid_rate": safe_div(sum(1 for item in subset_items if not item.get("valid")), len(subset_items)),
            }
            continue
        subset_scores = [float(item.get("official_score", 0.0)) for item in subset_items]
        subset_invalid = sum(1 for item in subset_items if not item.get("valid"))
        score_sum += sum(subset_scores)
        score_n += len(subset_scores)
        invalid += subset_invalid
        by_subset[subset] = {
            "n": len(subset_items),
            "score_sum": sum(subset_scores),
            "accuracy": safe_div(sum(subset_scores), len(subset_scores)),
            "invalid_rate": safe_div(subset_invalid, len(subset_items)),
        }

    non_ties_accuracies = [
        item["accuracy"]
        for subset, item in by_subset.items()
        if subset.lower() != "ties" and item.get("accuracy") is not None
    ]
    results_grouped = {
        subset: item["accuracy"]
        for subset, item in by_subset.items()
        if item.get("accuracy") is not None
    }
    official_domains = {
        domain: results_grouped[domain]
        for domain in RB2_OFFICIAL_DOMAIN_ORDER
        if results_grouped.get(domain) is not None
    }
    return {
        "n": len(records),
        "completed": len(rows),
        "missing": missing,
        "score_sum_non_ties": score_sum,
        "micro_accuracy_non_ties": safe_div(score_sum, score_n),
        "macro_accuracy_non_ties_by_subset": safe_div(sum(non_ties_accuracies), len(non_ties_accuracies)),
        "official_leaderboard_average": mean(list(official_domains.values())),
        "official_leaderboard_domains": official_domains,
        "invalid_rate_non_ties": safe_div(invalid, score_n),
        "by_subset": by_subset,
        "official_results_grouped": results_grouped,
        "skill_usage": skill_usage_from_rows(rows),
        "mean_latency_sec": mean([float(row["latency_sec"]) for row in rows if row.get("latency_sec")]),
        "endpoints": sorted({row.get("endpoint") for row in rows if row.get("endpoint")}),
    }


def skill_usage_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    skill_rows = [
        row
        for row in rows
        if row.get("mode") in {"skill_official_ranking", "agentic_skill_official_ranking", "self_select_skill_official_ranking"}
    ]
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
        "resources_loaded_by_sample": any(
            row.get("mode") in {"agentic_skill_official_ranking", "self_select_skill_official_ranking"}
            for row in skill_rows
        ),
        "skill_trigger_rate": safe_div(sum(1 for row in skill_rows if row.get("skill_triggered")), len(skill_rows)),
        "uses_openai_tool_calling": any(row.get("openai_tool_calling") for row in skill_rows),
        "mean_agent_step_count": mean([float(row.get("agent_step_count", 0)) for row in skill_rows if row.get("agent_step_count") is not None]),
        "mean_tool_call_count": mean([float(row.get("tool_call_count", 0)) for row in skill_rows if row.get("tool_call_count") is not None]),
        "mean_python_sandbox_call_count": mean([float(row.get("python_sandbox_call_count", 0)) for row in skill_rows if row.get("python_sandbox_call_count") is not None]),
        "mean_resource_view_count": mean([float(row.get("resource_view_count", 0)) for row in skill_rows if row.get("resource_view_count") is not None]),
    }


def compute_official_ties_score(rows: list[dict[str, Any]]) -> float | None:
    grouped_samples: dict[tuple[str, int], list[tuple[bool, float]]] = defaultdict(list)
    for row in rows:
        sample_type, prompt_id = parse_ties_id(row["sample_id"])
        ratings = row.get("ratings") or []
        num_correct = int(row.get("num_correct") or 0)
        for idx, raw_score in enumerate(ratings):
            grouped_samples[(sample_type, prompt_id)].append((idx < num_correct, float(raw_score)))

    ref_stats = {}
    tied_stats = {}
    for (sample_type, prompt_id), samples in grouped_samples.items():
        stats = compute_prompt_stats(samples)
        if stats is None:
            continue
        if sample_type == "ref":
            ref_stats[prompt_id] = stats
        else:
            tied_stats[prompt_id] = stats

    if not ref_stats and not tied_stats:
        return None

    ref_accuracy = mean([float(stat[0]) for stat in ref_stats.values()]) or 0.0
    tied_accuracy = mean([float(stat[0]) for stat in tied_stats.values()]) or 0.0
    shared_prompt_ids = set(ref_stats) & set(tied_stats)
    if not shared_prompt_ids:
        return 0.30 * tied_accuracy + 0.30 * ref_accuracy

    correctness_preferred = mean([
        float(tied_stats[prompt_id][2] > tied_stats[prompt_id][1])
        for prompt_id in shared_prompt_ids
    ]) or 0.0
    correctness_preferred_hard = mean([
        float(min(ref_stats[prompt_id][2], tied_stats[prompt_id][2]) > tied_stats[prompt_id][1])
        for prompt_id in shared_prompt_ids
    ]) or 0.0
    margin_scores = []
    for prompt_id in shared_prompt_ids:
        diff_corr_margin = tied_stats[prompt_id][1]
        if not diff_corr_margin:
            margin_scores.append(0.0)
            continue
        value = math.tanh(min(ref_stats[prompt_id][2], tied_stats[prompt_id][2]) / diff_corr_margin - 1)
        margin_scores.append(0.0 if math.isnan(value) else value)
    correctness_margin_score = mean(margin_scores) or 0.0

    return float(
        0.30 * tied_accuracy
        + 0.30 * ref_accuracy
        + 0.20 * correctness_preferred
        + 0.20 * correctness_preferred_hard
        + 0.01 * correctness_margin_score
    )


def compute_prompt_stats(samples: list[tuple[bool, float]]) -> tuple[bool, float | None, float] | None:
    correct_scores = [score for is_correct, score in samples if is_correct]
    incorrect_scores = [score for is_correct, score in samples if not is_correct]
    if not correct_scores or not incorrect_scores:
        return None
    best_correct = max(correct_scores)
    worst_correct = min(correct_scores)
    best_incorrect = max(incorrect_scores)
    different_correct_margin = best_correct - worst_correct if len(correct_scores) > 1 else None
    correct_incorrect_margin = worst_correct - best_incorrect
    return correct_incorrect_margin > 0, different_correct_margin, correct_incorrect_margin


def parse_ties_id(sample_id: str) -> tuple[str, int]:
    sample_type, prompt_id = str(sample_id).split(":", 1)
    return sample_type, int(prompt_id)
