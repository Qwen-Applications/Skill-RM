from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import time
from pathlib import Path
from typing import Any

from ..common.config import DEFAULT_ENDPOINTS, expand_env_vars, load_config, normalize_base_urls
from ..common.io import load_jsonl_map, print_progress, write_json, write_summary
from ..common.llm_client import _should_retry_without_thinking_field, call_chat_completion, call_with_retries
from ..common.parsing import parse_first_json_object
from ..common.stats import mean, safe_div
from ..common.tool_calls import (
    compact_tool_calls_for_trace,
    compact_tool_result_for_trace,
    first_final_answer_tool_call,
    parse_tool_call_arguments,
    tool_call_name,
)
from ..benchmarks.rewardbench2.data import iter_rb2_records
from ..benchmarks.rewardbench2.metrics import (
    RB2_OFFICIAL_DOMAIN_ORDER,
    compute_official_ties_score,
    compute_prompt_stats,
    official_metrics_from_rows,
    parse_ties_id,
    skill_usage_from_rows,
)
from ..benchmarks.rewardbench2.parsing import (
    official_ranking_score,
    parse_agentic_final,
    parse_final_answer_tool_call,
    parse_official_rating,
    parse_official_winner,
)
from ..benchmarks.rewardbench2.prompts import (
    OFFICIAL_RANKING_SYSTEM_PROMPT,
    OFFICIAL_RANKING_USER_TEMPLATE,
    format_official_rating_prompt,
    format_self_select_skill_system_prompt,
    format_self_select_skill_user_prompt,
    official_format_ranking_record,
    runtime_skill_tool_guidance,
)
from ..runtime.skill_package import (
    augment_judgebench_skill_markdown,
    augment_rewardbench2_skill_markdown,
    common_zip_prefix,
    is_judgebench_config,
    is_rewardbench2_config,
    is_skill_text_path,
    load_skill_package,
    parse_skill_manifest,
    read_all_skill_text_files,
    skill_files_sha256,
    skill_package_name,
    skill_package_sha256,
)
from ..runtime.python_sandbox import (
    PYTHON_SANDBOX_ALLOWED_IMPORTS,
    PYTHON_SANDBOX_FORBIDDEN_NAMES,
    PYTHON_SANDBOX_WRAPPER,
    run_python_sandbox_tool,
    validate_python_sandbox_code,
)
from ..runtime.resources import (
    OPERATIONAL_METADATA_BLOCKED_KEYS,
    RESOURCE_ID_PATHS,
    build_resource_index,
    combined_resource_index,
    manifest_entry_for_path,
    normalize_answer_text,
    normalize_resource_id,
    normalize_skill_resource_path,
    operational_sample_resources,
    resource_allowed,
    resource_path_for_entry,
    sanitize_operational_metadata,
    truncate_text,
    view_runtime_resource,
    view_skill_resource,
    visible_reference_payload,
)
from ..benchmarks.routing import RouteDecision, route_decision_for_metadata, route_row_fields, route_source_selection


AGENTIC_SKILL_MODES = {"self_select_skill_official_compat"}
OFFICIAL_COMPAT_MODES = {"official_compat"} | AGENTIC_SKILL_MODES


def main() -> None:
    args = parse_args()
    config = merge_cli(load_config(args.config), args)
    if args.recompute_metrics_only:
        recompute_metrics(config)
        return
    run_baseline(config)


def run_baseline(config: dict[str, Any]) -> None:
    if config.get("evaluation_mode") in OFFICIAL_COMPAT_MODES:
        run_official_compat(config)
        return
    raise ValueError(
        "Unsupported evaluation_mode. This runner supports official_compat and self_select_skill_official_compat."
    )


def run_official_compat(config: dict[str, Any]) -> None:
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_official_records(
        config["data_source"],
        limit=config.get("limit"),
        include_ties=bool(config.get("include_ties", True)),
    )
    base_urls = normalize_base_urls(config.get("base_urls") or DEFAULT_ENDPOINTS)
    workers = int(config.get("workers") or max(1, len(base_urls) * 4))
    skill_package = load_skill_package(config) if config.get("evaluation_mode") in AGENTIC_SKILL_MODES else None

    resolved_config = config | {"base_urls": base_urls}
    if skill_package:
        resolved_config["skill_package_sha256"] = skill_package["sha256"]
        resolved_config["skill_resources_loaded"] = skill_package["resources_loaded"]
        resolved_config["skill_resource_manifest_count"] = len(skill_package.get("manifest") or [])
    write_json(output_dir / "config_resolved.json", resolved_config)
    write_json(output_dir / "dataset_summary.json", summarize_records(records))

    completed = load_completed(output_dir / "predictions.jsonl") if config.get("resume") else {}
    pending = [record for record in records if str(record["id"]) not in completed]

    started_at = time.time()
    rows: dict[str, dict[str, Any]] = dict(completed)
    trace_handle = None
    if bool(config.get("record_trace", False)):
        trace_handle = (output_dir / "traces.jsonl").open("a", encoding="utf-8")
    try:
        with (output_dir / "predictions.jsonl").open("a", encoding="utf-8") as handle:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        judge_official_record,
                        record,
                        base_urls[index % len(base_urls)],
                        config,
                        skill_package,
                    ): record
                    for index, record in enumerate(pending)
                }
                for done_count, future in enumerate(
                    concurrent.futures.as_completed(futures),
                    start=1,
                ):
                    row = future.result()
                    trace = row.pop("_trace", None)
                    rows[row["sample_id"]] = row
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    if trace_handle is not None and trace is not None:
                        trace_handle.write(json.dumps(trace, ensure_ascii=False) + "\n")
                        trace_handle.flush()
                    if done_count % int(config.get("progress_every", 25)) == 0:
                        print_progress(done_count, len(pending), started_at)
    finally:
        if trace_handle is not None:
            trace_handle.close()

    ordered_rows = [rows[str(record["id"])] for record in records if str(record["id"]) in rows]
    metrics = official_metrics_from_rows(records, ordered_rows)
    write_json(output_dir / "metrics.json", metrics)
    source_selection = route_source_selection(ordered_rows, config)
    if source_selection is not None:
        write_json(output_dir / "source_selection.json", source_selection)
    write_summary(output_dir / "summary.md", config, [], ordered_rows, metrics, time.time() - started_at)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def recompute_metrics(config: dict[str, Any]) -> None:
    if config.get("evaluation_mode") not in OFFICIAL_COMPAT_MODES:
        raise ValueError("--recompute-metrics-only supports official-compatible modes only.")

    output_dir = Path(config["output_dir"])
    records = load_official_records(
        config["data_source"],
        limit=config.get("limit"),
        include_ties=bool(config.get("include_ties", True)),
    )
    completed = load_completed(output_dir / "predictions.jsonl")
    rows = list(completed.values())
    ordered_rows = [completed[str(record["id"])] for record in records if str(record["id"]) in completed]
    metrics = official_metrics_from_rows(records, ordered_rows or rows)
    write_json(output_dir / "metrics.json", metrics)
    source_selection = route_source_selection(ordered_rows or rows, config)
    if source_selection is not None:
        write_json(output_dir / "source_selection.json", source_selection)
    write_summary(output_dir / "summary.md", config, [], ordered_rows or rows, metrics, 0.0)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def load_official_records(
    data_source: str,
    *,
    limit: int | None = None,
    include_ties: bool = True,
) -> list[dict[str, Any]]:
    records = []
    for record in iter_rb2_records(data_source):
        if not include_ties and is_ties_record(record):
            continue
        records.append(record)
        if limit is not None and len(records) >= limit:
            break
    return records


def judge_official_record(
    record: dict[str, Any],
    base_url: str,
    config: dict[str, Any],
    skill_package: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if is_ties_record(record):
        return judge_official_ratings(record, base_url, config, is_ties=True)
    decision = rb2_route_decision(record, config)
    if config.get("evaluation_mode") == "self_select_skill_official_compat":
        if skill_package is None:
            raise ValueError("self_select_skill_official_compat requires a loaded skill package.")
        if decision.action == "baseline":
            row = judge_official_ranking(record, base_url, config)
            row.update(
                {
                    "mode": "self_select_skill_official_ranking",
                    "skill_path": skill_package["source"],
                    "skill_package_sha256": skill_package["sha256"],
                    "skill_loading_mode": "baseline_fallback",
                    "skill_available": True,
                    "skill_triggered": False,
                    "skill_trigger_step": None,
                    "skill_trigger_reason": decision.reason,
                    "controller_resources_loaded": [],
                    "resources_loaded": [],
                    "resources_viewed": [],
                    "resource_view_count": 0,
                    "tool_call_count": 0,
                    "python_sandbox_call_count": 0,
                    "tool_error_count": 0,
                    "agent_step_count": 0,
                    "openai_tool_calling": False,
                    "trace_id": None,
                    **route_row_fields(decision),
                }
            )
            return row
        row = judge_self_select_skill_official_ranking(record, base_url, config, skill_package)
        row.update(route_row_fields(decision))
        return row
    return judge_official_ranking(record, base_url, config)


def rb2_route_decision(record: dict[str, Any], config: dict[str, Any]) -> RouteDecision:
    policy_decision = route_decision_for_metadata(
        {
            "benchmark": "rb2",
            "subset": record.get("subset"),
            "subset_for_metrics_only": record.get("subset"),
        },
        config,
    )
    if policy_decision is not None:
        return policy_decision
    subset = str(record.get("subset") or "unknown")
    return RouteDecision("skill", subset, "subset", None, f"default_skill:{subset}")


def judge_official_ranking(record: dict[str, Any], base_url: str, config: dict[str, Any]) -> dict[str, Any]:
    formatted = official_format_ranking_record(record, seed=int(config.get("seed", 0)))
    messages = [
        {"role": "system", "content": OFFICIAL_RANKING_SYSTEM_PROMPT},
        {"role": "user", "content": formatted["user_prompt"]},
    ]
    response = call_with_retries(base_url, messages, config)
    raw_output = response["content"]
    winner = parse_official_winner(raw_output)
    score = official_ranking_score(winner, formatted["chosen_label"])
    valid = winner in {"A", "B", "C", "D"}
    return {
        "sample_id": str(record["id"]),
        "subset_for_metrics_only": record.get("subset"),
        "mode": "official_ranking",
        "chosen_label": formatted["chosen_label"],
        "predicted_label": winner,
        "official_score": score,
        "correct": score == 1.0,
        "valid": valid,
        "shuffle_position": formatted["shuffle_position"],
        "endpoint": base_url,
        **response_output_fields(response, config),
        "parse_error": None if valid else "official verdict not found",
    }


def judge_self_select_skill_official_ranking(
    record: dict[str, Any],
    base_url: str,
    config: dict[str, Any],
    skill_package: dict[str, Any],
) -> dict[str, Any]:
    formatted = official_format_ranking_record(record, seed=int(config.get("seed", 0)))
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": format_self_select_skill_system_prompt(skill_package, config)},
        {"role": "user", "content": format_self_select_skill_user_prompt(record, formatted)},
    ]
    skill_state: dict[str, Any] = {
        "loaded": False,
        "trigger_step": None,
        "trigger_reason": None,
    }
    max_steps = int(config.get("max_agent_steps", 8))
    resources_viewed: list[str] = []
    tool_error_count = 0
    tool_call_count = 0
    raw_output = ""
    final_response: dict[str, Any] = {}
    final_parsed: dict[str, str] = {"verdict": "error", "winner": "error", "source": "missing"}
    parse_error = "max agent steps exceeded"
    trace: dict[str, Any] = {
        "sample_id": str(record["id"]),
        "mode": "self_select_skill_official_compat",
        "skill_path": skill_package["source"],
        "skill_package_sha256": skill_package["sha256"],
        "initial_context": {
            "skills_advertised": [skill_package_name(skill_package)],
            "skill_contents_loaded": False,
            "resource_index_loaded": False,
        },
        "steps": [],
    }

    sandbox_call_count = 0
    started_at = time.time()
    for step in range(1, max_steps + 1):
        tools = openai_skill_tools(skill_loaded=bool(skill_state["loaded"]), config=config)
        response = call_with_retries(
            base_url,
            messages,
            config,
            tools=tools,
            tool_choice=str(config.get("tool_choice", "auto")),
        )
        raw_output = response["content"]
        final_response = response
        tool_calls = response.get("tool_calls") or []
        step_trace: dict[str, Any] = {
            "step": step,
            "assistant_content": raw_output,
            "finish_reason": response.get("finish_reason"),
            "latency_sec": response.get("latency_sec"),
            "reasoning_len": response.get("reasoning_len", 0),
            "request_error": response.get("error"),
            "tool_calls": compact_tool_calls_for_trace(tool_calls),
            "tool_results": [],
        }

        if response.get("error") and not raw_output and not tool_calls:
            parse_error = f"request failed: {response.get('error')}"
            step_trace["parse_error"] = parse_error
            trace["steps"].append(step_trace)
            break

        final_tool = first_final_answer_tool_call(tool_calls)
        if final_tool is not None:
            final_parsed = parse_final_answer_tool_call(final_tool)
            step_trace["final"] = final_parsed
            trace["steps"].append(step_trace)
            parse_error = None if final_parsed["winner"] in {"A", "B", "C", "D"} else "final_answer verdict not A/B/C/D"
            break

        if tool_calls:
            tool_call_count += len(tool_calls)
            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": raw_output or "",
                "tool_calls": tool_calls,
            }
            messages.append(assistant_message)
            for tool_call in tool_calls:
                tool_result = execute_openai_skill_tool_call(
                    tool_call,
                    skill_state,
                    skill_package,
                    record,
                    formatted,
                    config | {"_delegation_base_url": base_url},
                    resources_viewed,
                    step,
                )
                if tool_call_name(tool_call) == "python_sandbox":
                    sandbox_call_count += 1
                if not tool_result.get("ok"):
                    tool_error_count += 1
                step_trace["tool_results"].append(compact_tool_result_for_trace(tool_result))
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(tool_call.get("id") or f"call_{step}_{len(step_trace['tool_results'])}"),
                        "name": tool_call_name(tool_call),
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    }
                )
            trace["steps"].append(step_trace)
            continue

        final_parsed = parse_agentic_final(raw_output)
        if final_parsed["winner"] in {"A", "B", "C", "D"} or final_parsed["verdict"] in {"Tie", "Abstain"}:
            step_trace["final"] = final_parsed
            trace["steps"].append(step_trace)
            parse_error = None if final_parsed["winner"] in {"A", "B", "C", "D"} else "content final verdict not A/B/C/D"
            break

        step_trace["parse_error"] = "assistant returned neither tool call nor final verdict"
        trace["steps"].append(step_trace)
        parse_error = "assistant returned neither tool call nor final verdict"
        break

    if (
        final_parsed["winner"] not in {"A", "B", "C", "D"}
        and bool(config.get("enable_forced_finalization", True))
        and messages
    ):
        messages.append(
            {
                "role": "user",
                "content": (
                    "No more tool calls are allowed. Based only on the visible prompt, candidates, and evidence already "
                    "collected, choose the single best candidate. Do not explain. Reply with exactly one parseable "
                    "verdict line in this format: [[A]], [[B]], [[C]], or [[D]]."
                ),
            }
        )
        forced_response = call_with_retries(base_url, messages, config)
        forced_raw = forced_response["content"]
        forced_parsed = parse_agentic_final(forced_raw)
        forced_trace = {
            "step": len(trace["steps"]) + 1,
            "forced_finalization": True,
            "assistant_content": forced_raw,
            "finish_reason": forced_response.get("finish_reason"),
            "latency_sec": forced_response.get("latency_sec"),
            "reasoning_len": forced_response.get("reasoning_len", 0),
            "request_error": forced_response.get("error"),
            "tool_calls": [],
            "tool_results": [],
            "final": forced_parsed,
        }
        trace["steps"].append(forced_trace)
        raw_output = forced_raw
        final_response = forced_response
        final_parsed = forced_parsed
        parse_error = None if forced_parsed["winner"] in {"A", "B", "C", "D"} else f"forced finalization failed after {parse_error}"

    winner = final_parsed["winner"]
    score = official_ranking_score(winner, formatted["chosen_label"])
    valid = winner in {"A", "B", "C", "D"}
    resources_unique = sorted(dict.fromkeys(resources_viewed))
    runtime_resources_run = skill_state.get("runtime_resources_run") if isinstance(skill_state.get("runtime_resources_run"), list) else []
    resources_run_unique = sorted(dict.fromkeys(str(item) for item in runtime_resources_run))
    controller_resources = ["SKILL.md", "resources.yaml:index"] if skill_state["loaded"] else []
    trace["final"] = {
        "verdict": final_parsed["verdict"],
        "winner": winner,
        "valid": valid,
        "skill_triggered": bool(skill_state["loaded"]),
        "skill_trigger_step": skill_state.get("trigger_step"),
        "resources_viewed": resources_unique,
        "runtime_resources_run": resources_run_unique,
        "tool_error_count": tool_error_count,
    }
    row = {
        "sample_id": str(record["id"]),
        "subset_for_metrics_only": record.get("subset"),
        "mode": "self_select_skill_official_ranking",
        "chosen_label": formatted["chosen_label"],
        "predicted_label": winner,
        "skill_final_verdict": final_parsed["verdict"],
        "verdict_source": final_parsed["source"],
        "official_score": score,
        "correct": score == 1.0,
        "valid": valid,
        "shuffle_position": formatted["shuffle_position"],
        "endpoint": base_url,
        "skill_path": skill_package["source"],
        "skill_package_sha256": skill_package["sha256"],
        "skill_loading_mode": "self_select_progressive",
        "skill_available": True,
        "skill_triggered": bool(skill_state["loaded"]),
        "skill_trigger_step": skill_state.get("trigger_step"),
        "skill_trigger_reason": skill_state.get("trigger_reason"),
        "controller_resources_loaded": controller_resources,
        "resources_loaded": controller_resources + resources_unique,
        "resources_viewed": resources_unique,
        "runtime_resources_run": resources_run_unique,
        "resource_view_count": len(resources_viewed),
        "tool_call_count": tool_call_count,
        "python_sandbox_call_count": sandbox_call_count,
        "tool_error_count": tool_error_count,
        "agent_step_count": len(trace["steps"]),
        "openai_tool_calling": True,
        "trace_id": str(record["id"]),
        "latency_sec": time.time() - started_at,
        "enable_thinking": bool(config.get("enable_thinking", False)),
        "thinking_field_sent": final_response.get("thinking_field_sent"),
        "reasoning_len": sum(int(item.get("reasoning_len") or 0) for item in trace["steps"]),
        "finish_reason": final_response.get("finish_reason"),
        "request_error": final_response.get("error"),
        "raw_output": raw_output,
        "parse_error": parse_error,
        "_trace": trace,
    }
    if final_response.get("reasoning") and bool(config.get("save_reasoning", config.get("enable_thinking", False))):
        row["reasoning"] = final_response["reasoning"]
    return row


def judge_official_ratings(
    record: dict[str, Any],
    base_url: str,
    config: dict[str, Any],
    *,
    is_ties: bool,
) -> dict[str, Any]:
    prompt = str(record.get("prompt", ""))
    answers = [str(item) for item in list(record.get("chosen") or []) + list(record.get("rejected") or [])]
    ratings = []
    judgments = []
    latencies = []
    reasoning_lens = []
    finish_reasons = []
    for answer in answers:
        messages = [
            {
                "role": "user",
                "content": format_official_rating_prompt(prompt, answer, is_ties=is_ties),
            }
        ]
        response = call_with_retries(base_url, messages, config)
        raw_output = response["content"]
        ratings.append(parse_official_rating(raw_output))
        judgments.append(raw_output)
        latencies.append(float(response.get("latency_sec", 0.0)))
        reasoning_lens.append(int(response.get("reasoning_len", 0)))
        finish_reasons.append(response.get("finish_reason"))

    score = None
    predicted_winners: list[int] = []
    if not is_ties:
        valid_scores = [rating for rating in ratings if rating != -1]
        if valid_scores:
            max_rating = max(valid_scores)
            predicted_winners = [idx for idx, rating in enumerate(ratings) if rating == max_rating]
            score = (0 in predicted_winners) / len(predicted_winners)
        else:
            score = 0.25

    row = {
        "sample_id": str(record["id"]),
        "subset_for_metrics_only": record.get("subset"),
        "mode": "official_ties_ratings" if is_ties else "official_ratings",
        "num_correct": record.get("num_correct"),
        "ratings": ratings,
        "predicted_winners": predicted_winners,
        "official_score": score,
        "correct": score == 1.0 if score is not None else None,
        "valid": all(rating != -1 for rating in ratings),
        "endpoint": base_url,
        "latency_sec": sum(latencies),
        "enable_thinking": bool(config.get("enable_thinking", False)),
        "reasoning_len": sum(reasoning_lens),
        "finish_reasons": finish_reasons,
        "raw_output": judgments,
        "parse_error": None if all(rating != -1 for rating in ratings) else "one or more ratings invalid",
    }
    return row


def response_output_fields(response: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    row = {
        "latency_sec": response["latency_sec"],
        "enable_thinking": bool(config.get("enable_thinking", False)),
        "thinking_field_sent": response.get("thinking_field_sent"),
        "reasoning_len": response.get("reasoning_len", 0),
        "finish_reason": response.get("finish_reason"),
        "raw_output": response.get("content", ""),
    }
    if response.get("reasoning") and bool(config.get("save_reasoning", config.get("enable_thinking", False))):
        row["reasoning"] = response["reasoning"]
    return row

def openai_skill_tools(*, skill_loaded: bool, config: dict[str, Any]) -> list[dict[str, Any]]:
    final_answer = {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": "Submit the final listwise judgment. The verdict must be exactly one of A, B, C, or D.",
            "parameters": {
                "type": "object",
                "properties": {
                    "verdict": {"type": "string", "enum": ["A", "B", "C", "D"]},
                    "rationale": {"type": "string"},
                    "judgment_package": {"type": "object"},
                },
                "required": ["verdict"],
                "additionalProperties": True,
            },
        },
    }
    python_sandbox_tool = None
    if bool(config.get("enable_python_sandbox", True)):
        python_sandbox_tool = {
            "type": "function",
            "function": {
                "name": "python_sandbox",
                "description": (
                    "Run small Python checks over the visible prompt and candidates. Use this for deterministic "
                    "instruction-following evidence: word counts, regex, required terms, vowel sets, quote nesting, "
                    "format validity, simple arithmetic, and JSON/Markdown structure. No network, files, subprocesses, "
                    "or external data are available."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": (
                                "Python code. Available variables: prompt (str), candidates (dict label->text), "
                                "sample (dict with prompt and candidates). Print compact JSON or text evidence."
                            ),
                        },
                        "reason": {"type": "string", "description": "What deterministic property this code checks."},
                    },
                    "required": ["code", "reason"],
                    "additionalProperties": False,
                },
            },
        }
    if not skill_loaded:
        trigger_strength = str(config.get("operational_trigger_strength") or "").strip().lower()
        resource_first = (
            str(config.get("skill_allowed_setting") or "") == "skill_operational"
            and trigger_strength in {"high", "resource_first", "trigger_v1", "trigger_v2"}
        )
        use_skill_description = (
            "Load the optional judging skill when its instructions, resources, or deterministic checks may improve the judgment. "
            "Strong triggers include exact format/count/keyword constraints, subtle correctness errors, safety/refusal tradeoffs, and length/style bias risk."
        )
        if str(config.get("skill_allowed_setting") or "") == "skill_operational":
            use_skill_description = (
                "Load the optional operational judging skill when resource-rich evidence or deterministic checks may improve the judgment, "
                "especially objective-answer, math, code, factuality, exact-format, checklist, safety/refusal, calibration, or instruction-following tasks. "
                "After loading, inspect the resource index for available rubric/principles, sample-visible metadata, reference/ground truth, checklist, or constraints."
            )
            if is_rewardbench2_config(config) and str(config.get("rewardbench2_skill_trigger_policy") or "").lower() in {
                "operational_default_load",
                "operational_mandatory",
            }:
                use_skill_description = (
                    "Load the operational listwise judging skill for non-trivial factuality, exact-format, math/code, "
                    "safety/refusal, instruction-following, close-quality, or calibration-sensitive listwise samples. "
                    "Use direct final_answer only when the best candidate is obvious without any rubric, resource, "
                    "reference, checklist, or deterministic check. After loading, inspect the resource index and read only "
                    "decisive resources before choosing A, B, C, or D."
                )
            elif resource_first:
                use_skill_description = (
                    "Load the operational judging skill before final_answer for non-trivial comparisons where resource-rich "
                    "evidence, rubric guidance, sample-visible reference/ground truth, checklist, verifier output, or a "
                    "deterministic check could plausibly change the winner. Use direct final_answer only for obvious cases."
                )
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "use_skill",
                    "description": use_skill_description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "skill_name": {"type": "string", "description": "Name of the skill to load."},
                            "reason": {"type": "string", "description": "Why the sample needs skill support."},
                        },
                        "required": ["skill_name", "reason"],
                        "additionalProperties": False,
                    },
                },
            },
        ]
        tools.append(final_answer)
        return tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "list_resources",
                "description": "List available skill resources after the skill has been loaded.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "type": {"type": ["string", "null"], "description": "Optional resource type filter."}
                    },
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "view_resource",
                "description": "Read one skill resource by path. Use only resources needed for the current sample.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["path", "reason"],
                    "additionalProperties": False,
                },
            },
        },
    ]
    if runtime_resource_tools_enabled(config):
        tools.append(runtime_resource_tool_schema())
    if python_sandbox_tool is not None:
        tools.append(python_sandbox_tool)
    tools.append(final_answer)
    return tools


def runtime_resource_tools_enabled(config: dict[str, Any]) -> bool:
    return str(config.get("skill_allowed_setting") or "") == "skill_operational" and bool(
        config.get("enable_run_resource", True)
    )


def runtime_resource_tool_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "run_resource",
            "description": (
                "Run one executable operational resource from the loaded skill resource index. "
                "Use only entries whose implementation_kind is runtime_verifier or runtime_llm_pipeline."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "resource_id": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["resource_id", "reason"],
                "additionalProperties": False,
            },
        },
    }


def run_runtime_resource_tool(
    args: dict[str, Any],
    skill_state: dict[str, Any],
    skill_package: dict[str, Any],
    record: dict[str, Any],
    formatted: dict[str, Any],
    config: dict[str, Any],
    resources_viewed: list[str],
) -> dict[str, Any]:
    resource_id = normalize_resource_id(str(args.get("resource_id") or args.get("id") or ""))
    reason = str(args.get("reason") or "")
    if not runtime_resource_tools_enabled(config):
        return {
            "ok": False,
            "tool": "run_resource",
            "resource_id": resource_id,
            "reason": reason,
            "error": "runtime resources are only enabled for skill_operational",
        }
    if not resource_id:
        return {"ok": False, "tool": "run_resource", "error": "missing resource_id"}

    runtime_index = skill_state.get("runtime_resource_index") if isinstance(skill_state.get("runtime_resource_index"), list) else []
    visible = combined_resource_index(skill_package, config, runtime_index)
    entry = next((item for item in visible if normalize_resource_id(str(item.get("id") or "")) == resource_id), None)
    if entry is None:
        return {
            "ok": False,
            "tool": "run_resource",
            "resource_id": resource_id,
            "reason": reason,
            "error": "resource is not visible in the loaded skill resource index",
        }
    kind = str(entry.get("implementation_kind") or "")
    if kind not in {"runtime_verifier", "runtime_llm_pipeline"}:
        return {
            "ok": False,
            "tool": "run_resource",
            "resource_id": resource_id,
            "reason": reason,
            "implementation_kind": kind,
            "error": "resource is not executable; use view_resource for reference resources",
        }

    resources_run = skill_state.setdefault("runtime_resources_run", [])
    if not isinstance(resources_run, list):
        resources_run = []
        skill_state["runtime_resources_run"] = resources_run
    max_run_resources = int(config.get("max_run_resources_per_sample", 3))
    if resource_id not in resources_run and len(set(resources_run)) >= max_run_resources:
        return {
            "ok": False,
            "tool": "run_resource",
            "resource_id": resource_id,
            "reason": reason,
            "error": f"max runtime resources per sample exceeded: {max_run_resources}",
        }
    if resource_id not in resources_run:
        resources_run.append(resource_id)
    resources_viewed.append(resource_id)

    if resource_id == "external.rewardbench2_official_listwise_qwen":
        result = run_rewardbench2_official_listwise_resource(record, formatted, config)
    elif resource_id == "external.openrs_pairwise_qwen":
        result = run_openrs_pairwise_resource(record, formatted, config)
    elif resource_id == "verifier.reference_match":
        result = run_reference_match_resource(record, formatted)
    elif resource_id == "verifier.ground_truth_score_pair":
        result = run_ground_truth_score_pair_resource(record, formatted, config)
    else:
        result = {"verdict": "inconclusive", "source": resource_id, "reason": "resource runner is not implemented"}
    return {
        "ok": True,
        "tool": "run_resource",
        "resource_id": resource_id,
        "reason": reason,
        "result": result,
    }


def run_rewardbench2_official_listwise_resource(
    record: dict[str, Any],
    formatted: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    responses = dict(formatted.get("responses") or {})
    labels = [label for label in ("A", "B", "C", "D") if label in responses]
    if len(labels) != 4:
        return {"verdict": "inconclusive", "source": "rewardbench2_official_listwise_qwen", "error": "requires A/B/C/D responses"}
    base_url = str(config.get("_delegation_base_url") or "")
    if not base_url:
        return {"verdict": "inconclusive", "source": "rewardbench2_official_listwise_qwen", "error": "no endpoint available"}
    messages = [
        {"role": "system", "content": OFFICIAL_RANKING_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": OFFICIAL_RANKING_USER_TEMPLATE.format(
                question=str(record.get("prompt", "")),
                answer_a=responses["A"],
                answer_b=responses["B"],
                answer_c=responses["C"],
                answer_d=responses["D"],
            ),
        },
    ]
    run_config = dict(config)
    run_config["max_tokens"] = int(config.get("rewardbench2_external_max_tokens", 2048))
    response = call_with_retries(base_url, messages, run_config, tools=None, tool_choice=None)
    raw = str(response.get("content") or "")
    verdict = parse_official_winner(raw)
    return {
        "verdict": verdict if verdict in labels else "inconclusive",
        "raw_output": truncate_text(raw, 1200),
        "source": "rewardbench2_official_listwise_qwen",
        "confidence": "medium" if verdict in labels else "low",
        "request_error": response.get("error"),
    }


def run_openrs_pairwise_resource(
    record: dict[str, Any],
    formatted: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    responses = dict(formatted.get("responses") or {})
    if not {"A", "B"}.issubset(responses):
        return {"verdict": "inconclusive", "source": "openrs_pairwise_qwen", "error": "requires A/B responses"}
    base_url = str(config.get("_delegation_base_url") or "")
    if not base_url:
        return {"verdict": "inconclusive", "source": "openrs_pairwise_qwen", "error": "no endpoint available"}
    benchmark = str(config.get("benchmark") or record.get("benchmark") or "")
    messages = build_openrs_pairwise_resource_messages(
        prompt=str(record.get("prompt", "")),
        response_a=str(responses["A"]),
        response_b=str(responses["B"]),
        benchmark=benchmark,
    )
    run_config = dict(config)
    run_config["max_tokens"] = int(config.get("external_openrs_max_tokens", 512))
    response = call_with_retries(base_url, messages, run_config, tools=None, tool_choice=None)
    raw = str(response.get("content") or "")
    parsed = parse_pairwise_resource_verdict(raw)
    return {
        "verdict": parsed["verdict"],
        "raw_output": truncate_text(raw, 1200),
        "source": "openrs_pairwise_qwen",
        "confidence": "medium" if parsed["verdict"] in {"A", "B"} else "low",
        "parse_source": parsed["source"],
        "request_error": response.get("error"),
    }


def build_openrs_pairwise_resource_messages(
    *,
    prompt: str,
    response_a: str,
    response_b: str,
    benchmark: str,
) -> list[dict[str, Any]]:
    if str(benchmark).lower().startswith("judgebench"):
        return [
            {
                "role": "user",
                "content": (
                    "You are a helpful assistant in evaluating the quality of the outputs for a given instruction. "
                    "Your goal is to select the best output for the given instruction. Select the Output (a) or "
                    "Output (b) that is better for the given instruction. Do NOT provide any explanation for your "
                    'choice. Answer using ONLY "Output (a)" or "Output (b)".\n'
                    f"# Instruction: {prompt} # Output (a): {response_a} # Output (b): {response_b} "
                    '# Which is better, Output (a) or Output (b)?'
                ),
            }
        ]
    return [
        {
            "role": "system",
            "content": (
                "You are an impartial pairwise reward judge. Compare Response A and Response B for the same "
                "user request. Prioritize correctness, instruction following, safety, usefulness, and factuality. "
                "Return exactly one label: A, B, or Tie."
            ),
        },
        {
            "role": "user",
            "content": (
                f"[User Request]\n{prompt}\n\n"
                f"[Response A]\n{response_a}\n\n"
                f"[Response B]\n{response_b}\n\n"
                "Which response is better? Reply with exactly A, B, or Tie."
            ),
        },
    ]


def parse_pairwise_resource_verdict(raw_output: str) -> dict[str, str]:
    text = str(raw_output or "")
    output_label = re.findall(r"(?i)Output\s*\(([ab])\)", text)
    if output_label:
        verdict = output_label[-1].upper()
        return {"verdict": verdict, "source": "output_label"}
    rm_label = re.findall(r"(?i)\b(A\s*>>\s*B|A\s*>\s*B|A\s*=\s*B|B\s*>\s*A|B\s*>>\s*A)\b", text)
    if rm_label:
        label = re.sub(r"\s+", "", rm_label[-1].upper())
        if label in {"A>>B", "A>B"}:
            return {"verdict": "A", "source": "scaled_pairwise"}
        if label in {"B>A", "B>>A"}:
            return {"verdict": "B", "source": "scaled_pairwise"}
        return {"verdict": "Tie", "source": "scaled_pairwise"}
    parsed = parse_first_json_object(text)
    for key in ("verdict", "winner", "selected", "best_label"):
        value = parsed.get(key)
        if isinstance(value, str):
            normalized = value.strip()
            upper = normalized.upper()
            if upper in {"A", "B"}:
                return {"verdict": upper, "source": f"json.{key}"}
            if normalized.lower() in {"tie", "same", "draw"}:
                return {"verdict": "Tie", "source": f"json.{key}"}
    final_matches = re.findall(r"(?im)^\s*Final:\s*(A|B|Tie)\s*\.?\s*$", text)
    if final_matches:
        return {"verdict": final_matches[-1].title() if final_matches[-1].lower() == "tie" else final_matches[-1].upper(), "source": "final_line"}
    bracket = re.findall(r"\[\[(A|B|Tie)\]\]", text, flags=re.IGNORECASE)
    if bracket:
        last = bracket[-1]
        return {"verdict": last.title() if last.lower() == "tie" else last.upper(), "source": "bracket"}
    exact = text.strip().upper()
    if exact in {"A", "B"}:
        return {"verdict": exact, "source": "exact_label"}
    if exact in {"TIE", "SAME", "DRAW"}:
        return {"verdict": "Tie", "source": "exact_label"}
    return {"verdict": "inconclusive", "source": "unparsed"}


def run_reference_match_resource(record: dict[str, Any], formatted: dict[str, Any]) -> dict[str, Any]:
    reference = None
    for key in ("ground_truth", "reference", "answer", "expected_answer", "correct_answer", "gt"):
        if record.get(key) not in (None, "", []):
            reference = str(record.get(key))
            break
    if not reference:
        return {"verdict": "inconclusive", "source": "reference_match", "reason": "no visible reference field"}
    ref_norm = normalize_answer_text(reference)
    matches = []
    for label, text in dict(formatted.get("responses") or {}).items():
        if ref_norm and ref_norm in normalize_answer_text(str(text)):
            matches.append(str(label))
    if len(matches) == 1:
        return {"verdict": matches[0], "source": "reference_match", "matched_labels": matches}
    return {
        "verdict": "inconclusive",
        "source": "reference_match",
        "matched_labels": matches,
        "reason": "zero or multiple candidates matched the visible reference",
    }


GROUND_TRUTH_SCORE_SYSTEM_PROMPT = """You are a strict answer verifier.
Compare one candidate response against the visible user prompt and visible ground-truth/reference evidence.
Use only the provided prompt, candidate response, and reference evidence.
Return JSON only:
{"score": 1|0|-1, "reason": "short reason"}

Scoring:
- 1: the candidate's final answer is correct or substantively matches the reference.
- 0: the candidate is partially correct, ambiguous, incomplete, or cannot be judged from the reference.
- -1: the candidate is incorrect, contradicts the reference, or selects the wrong final answer.
"""


def run_ground_truth_score_pair_resource(
    record: dict[str, Any],
    formatted: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    responses = dict(formatted.get("responses") or {})
    if not {"A", "B"}.issubset(responses):
        return {"verdict": "inconclusive", "source": "ground_truth_score_pair", "error": "requires A/B responses"}
    reference = visible_reference_payload(record)
    if not reference:
        return {"verdict": "inconclusive", "source": "ground_truth_score_pair", "reason": "no visible reference or ground truth"}
    base_url = str(config.get("_delegation_base_url") or "")
    if not base_url:
        return {"verdict": "inconclusive", "source": "ground_truth_score_pair", "error": "no endpoint available"}

    scores: dict[str, int | None] = {}
    raw_outputs: dict[str, str] = {}
    request_errors: dict[str, Any] = {}
    parse_sources: dict[str, str | None] = {}
    run_config = dict(config)
    run_config["max_tokens"] = int(config.get("ground_truth_score_max_tokens", 384))
    run_config["temperature"] = float(config.get("ground_truth_score_temperature", 0.0))
    for label in ("A", "B"):
        messages = [
            {"role": "system", "content": GROUND_TRUTH_SCORE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "prompt": str(record.get("prompt", "")),
                        "reference": reference,
                        "candidate_label": label,
                        "candidate_response": str(responses[label]),
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        response = call_with_retries(base_url, messages, run_config, tools=None, tool_choice=None)
        raw = str(response.get("content") or "")
        score, source = parse_ground_truth_score(raw)
        scores[label] = score
        raw_outputs[label] = truncate_text(raw, 800)
        request_errors[label] = response.get("error")
        parse_sources[label] = source

    score_a = scores.get("A")
    score_b = scores.get("B")
    if score_a is None or score_b is None:
        verdict = "inconclusive"
        reason = "one or both verifier scores were unparseable"
    elif score_a > score_b:
        verdict = "A"
        reason = "A scored higher against the visible ground truth"
    elif score_b > score_a:
        verdict = "B"
        reason = "B scored higher against the visible ground truth"
    else:
        verdict = "inconclusive"
        reason = "both candidates received the same verifier score"
    return {
        "verdict": verdict,
        "source": "ground_truth_score_pair",
        "scores": scores,
        "parse_sources": parse_sources,
        "raw_outputs": raw_outputs,
        "request_errors": request_errors,
        "reference_keys": sorted(reference),
        "confidence": "high" if verdict in {"A", "B"} and set(scores.values()) <= {1, -1, 0} else "low",
        "reason": reason,
    }


def parse_ground_truth_score(raw_output: str) -> tuple[int | None, str | None]:
    text = str(raw_output or "")
    parsed = parse_first_json_object(text)
    for key in ("score", "verdict_score", "correctness_score"):
        value = parsed.get(key)
        if isinstance(value, (int, float)) and int(value) in {-1, 0, 1}:
            return int(value), f"json.{key}"
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned in {"-1", "0", "1"}:
                return int(cleaned), f"json.{key}"
    match = re.search(r'"score"\s*:\s*(-?1|0)\b', text)
    if match:
        return int(match.group(1)), "regex.score"
    label = re.search(r"(?im)^\s*score\s*[:=]\s*(-?1|0)\s*$", text)
    if label:
        return int(label.group(1)), "line.score"
    lowered = text.lower()
    if re.search(r"\bincorrect\b|\bwrong\b|\bcontradict", lowered):
        return -1, "keyword.incorrect"
    if re.search(r"\bpartially\b|\bambiguous\b|\bunclear\b|\bincomplete\b", lowered):
        return 0, "keyword.partial"
    if re.search(r"\bcorrect\b|\bmatches\b", lowered):
        return 1, "keyword.correct"
    return None, None


def execute_openai_skill_tool_call(
    tool_call: dict[str, Any],
    skill_state: dict[str, Any],
    skill_package: dict[str, Any],
    record: dict[str, Any],
    formatted: dict[str, Any],
    config: dict[str, Any],
    resources_viewed: list[str],
    step: int,
) -> dict[str, Any]:
    name = tool_call_name(tool_call)
    args, arg_error = parse_tool_call_arguments(tool_call)
    if arg_error:
        return {"ok": False, "tool": name, "error": arg_error}

    if name == "use_skill":
        if skill_state.get("loaded"):
            return {"ok": True, "tool": name, "already_loaded": True}
        requested = str(args.get("skill_name") or skill_package_name(skill_package))
        canonical = skill_package_name(skill_package)
        if requested not in {canonical, "reward-judge", "skill-rm-judge"}:
            return {"ok": False, "tool": name, "error": f"unknown skill: {requested}", "available_skill": canonical}
        skill_state["loaded"] = True
        skill_state["trigger_step"] = step
        skill_state["trigger_reason"] = str(args.get("reason") or "")
        runtime_index, runtime_files = operational_sample_resources(record, formatted, config)
        skill_state["runtime_resource_index"] = runtime_index
        skill_state["runtime_resource_files"] = runtime_files
        return {
            "ok": True,
            "tool": name,
            "skill_name": canonical,
            "skill_controller": "\n\n".join(
                [
                    skill_package["files"].get("SKILL.md", "").strip(),
                    runtime_skill_tool_guidance(config),
                ]
            ).strip(),
            "resource_index": combined_resource_index(skill_package, config, runtime_index),
            "instructions": "Use the newly available tools only when they improve the judgment, then submit the final answer in the requested format.",
        }

    if name == "python_sandbox":
        return run_python_sandbox_tool(args, record, formatted, config)

    if not skill_state.get("loaded"):
        return {"ok": False, "tool": name, "error": "skill is not loaded; call use_skill first"}
    if name == "list_resources":
        resource_type = args.get("type")
        resources = combined_resource_index(
            skill_package,
            config,
            skill_state.get("runtime_resource_index") if isinstance(skill_state.get("runtime_resource_index"), list) else [],
        )
        if resource_type:
            resources = [item for item in resources if item.get("type") == resource_type]
        return {"ok": True, "tool": name, "resources": resources}
    if name == "view_resource":
        path = str(args.get("path") or "")
        try:
            runtime_result = view_runtime_resource(
                path,
                skill_state.get("runtime_resource_files") if isinstance(skill_state.get("runtime_resource_files"), dict) else {},
                resources_viewed,
                max_chars=int(config.get("max_resource_chars", 8000)),
            )
            if runtime_result is not None:
                return runtime_result
            return view_skill_resource(path, skill_package, config, resources_viewed)
        except ValueError as exc:
            return {"ok": False, "tool": name, "path": path, "error": str(exc)}
    if name == "run_resource":
        return run_runtime_resource_tool(
            args,
            skill_state,
            skill_package,
            record,
            formatted,
            config,
            resources_viewed,
        )
    return {"ok": False, "tool": name, "error": f"unknown tool: {name}"}


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_subset: dict[str, int] = {}
    for record in records:
        subset = str(record.get("subset") or "unknown")
        by_subset[subset] = by_subset.get(subset, 0) + 1
    return {"n": len(records), "by_subset": dict(sorted(by_subset.items()))}


def is_ties_record(record: dict[str, Any]) -> bool:
    return str(record.get("subset") or "").strip().lower() == "ties"


def load_completed(path: Path) -> dict[str, dict[str, Any]]:
    return load_jsonl_map(path, key="sample_id", missing_ok=True, skip_blank=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen RBv2 official-compatible evaluations.")
    parser.add_argument("--config")
    parser.add_argument("--data", dest="data_source")
    parser.add_argument("--output", dest="output_dir")
    parser.add_argument("--base-urls", help="Comma-separated OpenAI-compatible /v1 base URLs.")
    parser.add_argument("--model")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--max-agent-steps", type=int)
    parser.add_argument("--evaluation-mode", choices=["official_compat", "self_select_skill_official_compat"])
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--send-thinking-field", action="store_true")
    parser.add_argument("--no-send-thinking-field", action="store_true")
    parser.add_argument("--save-reasoning", action="store_true")
    parser.add_argument("--no-save-reasoning", action="store_true")
    parser.add_argument("--include-ties", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--recompute-metrics-only", action="store_true")
    return parser.parse_args()


def merge_cli(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    merged = dict(config)
    for key in ("data_source", "output_dir", "model", "limit", "seed", "workers", "timeout", "max_tokens", "max_agent_steps", "evaluation_mode"):
        value = getattr(args, key)
        if value is not None:
            merged[key] = value
    if args.base_urls:
        merged["base_urls"] = args.base_urls
    if args.enable_thinking:
        merged["enable_thinking"] = True
    if args.disable_thinking:
        merged["enable_thinking"] = False
    if args.send_thinking_field:
        merged["send_thinking_field"] = True
    if args.no_send_thinking_field:
        merged["send_thinking_field"] = False
    if args.save_reasoning:
        merged["save_reasoning"] = True
    if args.no_save_reasoning:
        merged["save_reasoning"] = False
    if args.include_ties:
        merged["include_ties"] = True
    if args.resume:
        merged["resume"] = True
    merged.setdefault("data_source", "data/rewardbench_v2/rewardbench_v2.jsonl")
    merged.setdefault("output_dir", "runs/rb2_qwen_baseline")
    merged.setdefault("model", "Qwen3.5-27B")
    merged.setdefault("seed", 0)
    merged.setdefault("include_ties", False)
    merged.setdefault("resume", True)
    merged.setdefault("temperature", 0.0)
    merged.setdefault("max_tokens", 512)
    merged.setdefault("evaluation_mode", "official_compat")
    merged.setdefault("enable_thinking", False)
    merged.setdefault("send_thinking_field", True)
    merged.setdefault("save_reasoning", bool(merged.get("enable_thinking", False)))
    return merged


if __name__ == "__main__":
    main()
