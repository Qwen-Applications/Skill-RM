from __future__ import annotations

import copy
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .io_utils import read_json
from .paths import DEFAULT_DATA_PATH, DEFAULT_METRICS_DIR, DEFAULT_POSITION_MAP_PATH


def build_metrics(
    rows: list[dict[str, Any]],
    *,
    mode: str,
    data_path: str | Path = DEFAULT_DATA_PATH,
    position_map_path: str | Path = DEFAULT_POSITION_MAP_PATH,
    metrics_dir: str | Path = DEFAULT_METRICS_DIR,
) -> dict[str, Any]:
    mode = mode.strip().lower()
    if mode == "overall":
        return build_overall_metrics(rows, data_path=data_path, position_map_path=position_map_path, metrics_dir=metrics_dir)
    if mode == "constraint":
        return build_constraint_metrics(rows, data_path=data_path, metrics_dir=metrics_dir)
    raise ValueError(f"unsupported IF-RewardBench mode: {mode}")


def build_overall_metrics(
    rows: list[dict[str, Any]],
    *,
    data_path: str | Path,
    position_map_path: str | Path,
    metrics_dir: str | Path,
) -> dict[str, Any]:
    overall_mod = import_metric_module(metrics_dir, "analysis_overall_assessment")
    original_data = read_json(data_path)
    position_maps = read_json(position_map_path)
    overall_mod.position_maps = position_maps
    data_by_id = {str(item["id"]): copy.deepcopy(item) for item in original_data}
    attempted = set()
    pred_by_prompt: dict[str, dict[str, str]] = defaultdict(dict)
    status = Counter()
    malformed = 0
    invalid_prediction_rows = 0
    unknown_prompt_rows = 0
    duplicate_pair_overwrites = 0
    valid_ab_pairs = 0
    for row in rows:
        status[row.get("status", "unknown")] += 1
        sid = str(row.get("original_id") or "")
        attempted.add(sid)
        if sid and sid not in data_by_id:
            unknown_prompt_rows += 1
            continue
        pair = row.get("pair_indices")
        pred = row.get("prediction")
        if not sid or not isinstance(pair, list) or len(pair) != 2:
            malformed += 1
            continue
        u, v = int(pair[0]), int(pair[1])
        key = f"{u}_{v}"
        if pred not in {"A", "B"}:
            invalid_prediction_rows += 1
            continue
        if key in pred_by_prompt[sid]:
            duplicate_pair_overwrites += 1
        pred_by_prompt[sid][key] = f"[[{pred}]]"
        valid_ab_pairs += 1

    collector = overall_mod.StatsCollector()
    scored = 0
    skipped = 0
    expected_pairs_total = 0
    pairs_missing_or_invalid = 0
    scoring_errors = Counter()
    random.seed(42)
    for sid in sorted(attempted):
        if sid not in data_by_id:
            skipped += 1
            continue
        item = data_by_id[sid]
        expected_keys = expected_overall_pair_keys(item, position_maps)
        expected_pairs_total += len(expected_keys)
        item["pairwise_evaluation_results"] = {}
        for key in expected_keys:
            val = pred_by_prompt.get(sid, {}).get(key, "")
            item["pairwise_evaluation_results"][key] = val
            if not val:
                pairs_missing_or_invalid += 1
        try:
            pairwise = overall_mod.calculate_pairwise_metrics(item)
        except Exception as exc:  # noqa: BLE001
            skipped += 1
            scoring_errors[type(exc).__name__] += 1
            continue
        c_bucket = overall_mod.get_constraint_count_bucket(len(item.get("checklist", [])))
        for group, key in [
            ("instruction_type", item.get("instruction_type", "unknown")),
            ("constraint_count", c_bucket),
        ]:
            collector.update(group, key, pairwise)
        scored += 1

    by_type: dict[str, Any] = {}
    acc_values: list[float] = []
    kendall_values: list[float] = []
    for t_type in ["Single_Turn", "Multi_Turn", "System_Prompt"]:
        if t_type in collector.stats.get("instruction_type", {}):
            acc = float(collector.get_raw_means("instruction_type", t_type, "pair_acc"))
            kendall = float(collector.get_raw_means("instruction_type", t_type, "kendall"))
            acc_values.append(acc)
            kendall_values.append(kendall)
            by_type[t_type] = {"pair_acc": acc, "kendall": kendall}
        else:
            by_type[t_type] = {"pair_acc": None, "kendall": None}
    return {
        "mode": "overall",
        "official_metric": "if_rewardbench.overall.kendall",
        "official_value": float(np.mean(kendall_values)) if kendall_values else 0.0,
        "overall": {
            "pair_acc": float(np.mean(acc_values)) if acc_values else 0.0,
            "kendall": float(np.mean(kendall_values)) if kendall_values else 0.0,
        },
        "by_instruction_type": by_type,
        "coverage": {
            "rows": len(rows),
            "prompts_attempted": len(attempted),
            "prompts_scored": scored,
            "prompts_skipped": skipped,
            "malformed_rows": malformed,
            "expected_pair_keys": expected_pairs_total,
            "valid_ab_pairs": valid_ab_pairs,
            "pairs_missing_or_invalid": pairs_missing_or_invalid,
            "invalid_prediction_rows": invalid_prediction_rows,
            "unknown_prompt_rows": unknown_prompt_rows,
            "duplicate_pair_overwrites": duplicate_pair_overwrites,
            "scoring_errors": dict(scoring_errors),
            "status": dict(status),
        },
    }


def expected_overall_pair_keys(item: dict[str, Any], position_maps: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    responses = item.get("responses", [])
    pos_for_id = position_maps.get(str(item.get("id")), {})
    for u in range(len(responses)):
        for v in range(len(responses)):
            if u == v:
                continue
            pair_key = f"{min(u, v)}_{max(u, v)}"
            position = pos_for_id.get(pair_key)
            if position is None:
                continue
            if (position == 0 and u < v) or (position == 1 and u > v):
                keys.append(f"{u}_{v}")
    return sorted(set(keys))


def build_constraint_metrics(rows: list[dict[str, Any]], *, data_path: str | Path, metrics_dir: str | Path) -> dict[str, Any]:
    const_mod = import_metric_module(metrics_dir, "analysis_constraint_assessment")
    original_data = read_json(data_path)
    original_data_by_id = {str(item["id"]): item for item in original_data}
    const_mod.constraint_composition_type_maps = {}
    const_mod.constraint_category_maps = {}
    for item in original_data:
        for constraint in item.get("constraint_type", []):
            text = str(constraint.get("item") or "").strip()
            if text:
                const_mod.constraint_composition_type_maps[text] = constraint.get("constraint_composition_types", [])
                const_mod.constraint_category_maps[text] = constraint.get("constraint_categories", [])

    pred_dict: dict[str, dict[int, str]] = defaultdict(dict)
    attempted = set()
    status = Counter()
    malformed = 0
    for row in rows:
        status[row.get("status", "unknown")] += 1
        sid = str(row.get("original_id") or "")
        attempted.add(sid)
        try:
            resp_idx = int(row.get("response_index"))
        except (TypeError, ValueError):
            malformed += 1
            continue
        if row.get("status") == "success":
            pred_dict[sid][resp_idx] = str(row.get("raw_output") or "")

    collector = const_mod.StatsCollector()
    scored = 0
    skipped = 0
    for sid in sorted(attempted):
        if sid not in original_data_by_id:
            skipped += 1
            continue
        item = copy.deepcopy(original_data_by_id[sid])
        for idx, response in enumerate(item.get("responses", [])):
            response["critique"] = pred_dict.get(sid, {}).get(idx, "")
        try:
            c_bucket = const_mod.get_constraint_count_bucket(len(item.get("checklist", [])))
            scores: dict[Any, float] = {}
            golden_labels: list[int] = []
            pred_labels: list[int] = []
            for response in item.get("responses", []):
                gold = response.get("labels", [])
                pred = const_mod.get_label(response, len(gold))
                scores[response.get("response_id")] = float(np.mean(pred)) if pred else 0.0
                golden_labels.extend(gold)
                pred_labels.extend(pred)
            pointwise = const_mod.calculate_pointwise_metrics(pred_labels, golden_labels)
            pairwise = const_mod.calculate_pairwise_metrics(item.get("preference_graph", []), scores)
            for group, key in [
                ("instruction_type", item.get("instruction_type", "unknown")),
                ("constraint_count", c_bucket),
            ]:
                collector.update(group, key, pointwise, pairwise, golden_labels, pred_labels)
            scored += 1
        except Exception:
            skipped += 1

    by_type: dict[str, Any] = {}
    pf1_values: list[float] = []
    nf1_values: list[float] = []
    acc_values: list[float] = []
    kendall_values: list[float] = []
    for t_type in ["Single_Turn", "Multi_Turn", "System_Prompt"]:
        if t_type in collector.stats.get("instruction_type", {}):
            pf1 = float(collector.get_raw_means("instruction_type", t_type, "pf1"))
            nf1 = float(collector.get_raw_means("instruction_type", t_type, "nf1"))
            acc = float(collector.get_raw_means("instruction_type", t_type, "pair_acc"))
            kendall = float(collector.get_raw_means("instruction_type", t_type, "kendall"))
            pf1_values.append(pf1)
            nf1_values.append(nf1)
            acc_values.append(acc)
            kendall_values.append(kendall)
            by_type[t_type] = {"positive_f1": pf1, "negative_f1": nf1, "pair_acc": acc, "kendall": kendall}
        else:
            by_type[t_type] = {"positive_f1": None, "negative_f1": None, "pair_acc": None, "kendall": None}
    return {
        "mode": "constraint",
        "official_metric": "if_rewardbench.constraint.positive_f1",
        "official_value": float(np.mean(pf1_values)) if pf1_values else 0.0,
        "overall": {
            "positive_f1": float(np.mean(pf1_values)) if pf1_values else 0.0,
            "negative_f1": float(np.mean(nf1_values)) if nf1_values else 0.0,
            "pair_acc": float(np.mean(acc_values)) if acc_values else 0.0,
            "kendall": float(np.mean(kendall_values)) if kendall_values else 0.0,
        },
        "by_instruction_type": by_type,
        "coverage": {
            "rows": len(rows),
            "prompts_attempted": len(attempted),
            "prompts_scored": scored,
            "prompts_skipped": skipped,
            "malformed_rows": malformed,
            "status": dict(status),
        },
    }


def import_metric_module(metrics_dir: str | Path, module_name: str) -> Any:
    metrics_dir = Path(metrics_dir).resolve()
    if str(metrics_dir) not in sys.path:
        sys.path.insert(0, str(metrics_dir))
    return __import__(module_name)


def metrics_summary_markdown(metrics: dict[str, Any]) -> str:
    lines = [
        f"# IF-RewardBench {metrics.get('mode')} Summary",
        "",
        f"- official_metric: `{metrics.get('official_metric')}`",
        f"- official_value: `{metrics.get('official_value')}`",
        "",
        "## Overall",
        "",
    ]
    for key, value in (metrics.get("overall") or {}).items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Coverage", ""])
    for key, value in (metrics.get("coverage") or {}).items():
        lines.append(f"- {key}: `{value}`")
    return "\n".join(lines) + "\n"
