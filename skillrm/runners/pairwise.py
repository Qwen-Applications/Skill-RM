from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..common.config import DEFAULT_ENDPOINTS, load_config, normalize_base_urls
from ..common.io import load_jsonl_map, print_progress, write_json
from ..common.tool_calls import (
    compact_tool_calls_for_trace,
    compact_tool_result_for_trace,
    first_final_answer_tool_call,
    parse_tool_call_arguments,
    tool_call_name,
)
from ..common.llm_client import call_with_retries
from ..common.stats import mean, safe_div
from ..benchmarks.pairwise.data import (
    MODE_BY_PAIR,
    PAIR_LABELS,
    VARIANT_MAP,
    OpenRSPairTask,
    configured_openrs_data_sources,
    is_judgebench,
    is_rmbench,
    iter_jsonl,
    load_judgebench_tasks,
    load_openrs_tasks,
    load_openrs_tasks_from_source,
    load_rmbench_tasks,
    nonempty_resource_value,
    normalize_judgebench_label,
    openrs_visible_sample_resources,
    prefix_openrs_task_source,
    sanitize_source_prefix,
    summarize_tasks,
)
from ..benchmarks.pairwise.metrics import (
    add_pairwise_outcome,
    aggregate_rm_pair,
    compute_judgebench_metrics,
    compute_openrs_metrics,
    compute_pairwise_task_metrics,
    compute_rmbench_metrics,
    finalize_counter,
    finalize_judgebench_counter,
    finalize_rm_counter,
    judgebench_add,
    judgebench_counter,
    new_counter,
    rm_add,
    rm_counter,
    skill_usage_from_openrs_rows,
)
from ..benchmarks.pairwise.parsing import (
    normalize_pairwise_verdict,
    normalize_pairwise_winner,
    pairwise_final_answer_tool,
    pairwise_final_valid_for_benchmark,
    parse_pairwise_final,
    parse_pairwise_final_answer_tool_call,
)
from ..benchmarks.pairwise.prompts import (
    BASELINE_PAIRWISE_SYSTEM_PROMPT,
    JUDGEBENCH_VANILLA_PROMPT,
    RMBENCH_PAIRWISE_SYSTEM_PROMPT,
    format_baseline_messages,
    format_judgebench_prompt,
    format_pairwise_user_prompt,
    format_rmbench_user_prompt,
)
from .rewardbench2 import (
    build_resource_index,
    execute_openai_skill_tool_call,
    format_self_select_skill_system_prompt,
    load_skill_package,
    runtime_skill_tool_guidance,
    runtime_resource_tool_schema,
    runtime_resource_tools_enabled,
    skill_package_name,
)
from ..benchmarks.routing import (
    RouteDecision,
    config_string_set,
    route_decision_for_metadata,
    route_row_fields,
    route_source_selection,
)


def main() -> None:
    args = parse_args()
    config = merge_cli(load_config(args.config), args)
    if args.recompute_metrics_only:
        recompute_metrics(config)
        return
    run_openrs_benchmark(config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Skill-RM/OpenRS pairwise benchmark experiments.")
    parser.add_argument("--config")
    parser.add_argument("--data", dest="data_source")
    parser.add_argument("--output", dest="output_dir")
    parser.add_argument("--benchmark")
    parser.add_argument("--evaluation-mode", choices=["pairwise_baseline", "self_select_skill_pairwise"])
    parser.add_argument("--base-urls")
    parser.add_argument("--model")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--max-agent-steps", type=int)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--send-thinking-field", action="store_true")
    parser.add_argument("--no-send-thinking-field", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--recompute-metrics-only", action="store_true")
    return parser.parse_args()


def merge_cli(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    merged = dict(config)
    for key in (
        "data_source",
        "output_dir",
        "benchmark",
        "evaluation_mode",
        "model",
        "limit",
        "seed",
        "workers",
        "timeout",
        "max_tokens",
        "max_agent_steps",
    ):
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
    if args.resume:
        merged["resume"] = True
    merged.setdefault("benchmark", "judgebench_gpt")
    merged.setdefault("evaluation_mode", "pairwise_baseline")
    merged.setdefault("model", "Qwen3.5-27B")
    merged.setdefault("seed", 0)
    merged.setdefault("resume", True)
    merged.setdefault("temperature", 0.0)
    merged.setdefault("top_p", 1.0)
    merged.setdefault("max_tokens", 2048)
    merged.setdefault("enable_thinking", False)
    merged.setdefault("send_thinking_field", True)
    merged.setdefault("timeout", 300)
    merged.setdefault("retries", 2)
    merged.setdefault("progress_every", 25)
    return merged


def run_openrs_benchmark(config: dict[str, Any]) -> None:
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_openrs_tasks(config)
    base_urls = resolve_base_urls(config)
    workers = int(config.get("workers") or max(1, len(base_urls)))
    skill_package = load_skill_package(config) if config.get("evaluation_mode") == "self_select_skill_pairwise" else None

    resolved = config | {"base_urls": base_urls}
    if skill_package:
        resolved["skill_package_sha256"] = skill_package["sha256"]
        resolved["skill_resources_loaded"] = skill_package["resources_loaded"]
        resolved["skill_resource_manifest_count"] = len(skill_package.get("manifest") or [])
    write_json(output_dir / "config_resolved.json", resolved)
    write_json(output_dir / "dataset_summary.json", summarize_tasks(tasks))

    completed = load_completed_rows(output_dir / "predictions.jsonl") if config.get("resume") else {}
    pending = [task for task in tasks if task.task_id not in completed]
    rows: dict[str, dict[str, Any]] = dict(completed)

    started_at = time.time()
    trace_handle = None
    if bool(config.get("record_trace", config.get("evaluation_mode") == "self_select_skill_pairwise")):
        trace_handle = (output_dir / "traces.jsonl").open("a", encoding="utf-8")
    try:
        with (output_dir / "predictions.jsonl").open("a", encoding="utf-8") as handle:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        judge_task_with_endpoint_failover,
                        task,
                        base_urls,
                        index,
                        config,
                        skill_package,
                    ): task
                    for index, task in enumerate(pending)
                }
                for done_count, future in enumerate(concurrent.futures.as_completed(futures), start=1):
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

    ordered_rows = [rows[task.task_id] for task in tasks if task.task_id in rows]
    metrics = compute_openrs_metrics(tasks, ordered_rows, config)
    write_json(output_dir / "metrics.json", metrics)
    source_selection = route_source_selection(ordered_rows, config)
    if source_selection is not None:
        write_json(output_dir / "source_selection.json", source_selection)
    write_openrs_summary(output_dir / "summary.md", config, metrics, len(tasks), len(ordered_rows), time.time() - started_at)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def recompute_metrics(config: dict[str, Any]) -> None:
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = load_openrs_tasks(config)
    rows = list(load_completed_rows(output_dir / "predictions.jsonl").values())
    metrics = compute_openrs_metrics(tasks, rows, config)
    write_json(output_dir / "metrics.json", metrics)
    source_selection = route_source_selection(rows, config)
    if source_selection is not None:
        write_json(output_dir / "source_selection.json", source_selection)
    write_openrs_summary(output_dir / "summary.md", config, metrics, len(tasks), len(rows), 0.0)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def resolve_base_urls(config: dict[str, Any]) -> list[str]:
    if config.get("base_urls"):
        return normalize_base_urls(config["base_urls"])
    hosts = config.get("endpoint_hosts")
    if hosts:
        ports = config.get("endpoint_ports") or list(range(8000, 8008))
        return [
            f"http://{str(host).strip()}:{int(port)}/v1"
            for port in ports
            for host in hosts
            if str(host).strip()
        ]
    return normalize_base_urls(config.get("base_urls") or DEFAULT_ENDPOINTS)


def judge_task(
    task: OpenRSPairTask,
    base_url: str,
    config: dict[str, Any],
    skill_package: dict[str, Any] | None,
) -> dict[str, Any]:
    if config.get("evaluation_mode") == "self_select_skill_pairwise":
        if skill_package is None:
            raise ValueError("self_select_skill_pairwise requires skill_path.")
        decision = openrs_route_decision(task, config)
        if decision.action == "baseline":
            fallback_config = config | {
                "send_thinking_field": bool(config.get("baseline_fallback_send_thinking_field", True))
            }
            row = judge_pairwise_baseline(task, base_url, fallback_config)
            row.update(
                {
                    "mode": "self_select_skill_pairwise",
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
        row = judge_pairwise_self_select_skill(task, base_url, config, skill_package)
        row.update(route_row_fields(decision))
        return row
    return judge_pairwise_baseline(task, base_url, config)


def openrs_route_decision(task: OpenRSPairTask, config: dict[str, Any]) -> RouteDecision:
    policy_decision = route_decision_for_metadata(
        {
            "benchmark": task.benchmark,
            "query_type": task.query_type,
            "domain": task.domain,
            "pair": task.pair,
        },
        config,
    )
    if policy_decision is not None:
        return policy_decision

    if is_judgebench(task.benchmark):
        configured = config.get("baseline_fallback_query_types")
        if configured is False:
            return RouteDecision("skill", str(task.query_type or "unknown"), "query_type", None, "configured_judgebench_skill_only")
        if configured is None:
            key = str(task.query_type or "unknown")
            return RouteDecision("skill", key, "query_type", None, f"default_skill:{key}")
        groups = config_string_set(configured)
        key = str(task.query_type or "unknown")
        if key in groups:
            return RouteDecision("baseline", key, "query_type", None, f"configured_judgebench_baseline:{key}")
        return RouteDecision("skill", key, "query_type", None, f"configured_judgebench_skill:{key}")

    if is_rmbench(task.benchmark):
        skill_pairs = config_string_set(config.get("rmbench_skill_domain_pairs"))
        if skill_pairs:
            key = f"{task.domain}:{task.pair}"
            if key not in skill_pairs:
                return RouteDecision("baseline", key, "domain_pair", None, f"configured_rmbench_baseline:{key}")
            return RouteDecision("skill", key, "domain_pair", None, f"configured_rmbench_skill:{key}")
    key = str(task.query_type or task.domain or "unknown")
    return RouteDecision("skill", key, "query_type", None, f"default_skill:{key}")


def pairwise_task_runtime_record(task: OpenRSPairTask) -> dict[str, Any]:
    """Build the sample record exposed to operational runtime resources.

    This intentionally keeps metric labels and chosen/rejected origin out of the
    record while preserving OpenRS-style visible resources such as ground truth,
    checklists, constraints, and sanitized metadata.
    """
    record: dict[str, Any] = {
        "id": task.task_id,
        "prompt": task.prompt,
        "benchmark": task.benchmark,
        "query_type": task.query_type,
        "domain": task.domain,
        "pair": task.pair,
        "order": task.order,
    }
    for key, value in task.sample_resources.items():
        if key not in record:
            record[key] = value
    return record


def judge_task_with_endpoint_failover(
    task: OpenRSPairTask,
    base_urls: list[str],
    start_index: int,
    config: dict[str, Any],
    skill_package: dict[str, Any] | None,
) -> dict[str, Any]:
    max_attempts = min(len(base_urls), int(config.get("endpoint_failover_attempts", 4)))
    last_row: dict[str, Any] | None = None
    for offset in range(max_attempts):
        base_url = base_urls[(start_index + offset) % len(base_urls)]
        row = judge_task(task, base_url, config, skill_package)
        if not row.get("request_error"):
            return row
        last_row = row
    assert last_row is not None
    last_row["endpoint_failover_exhausted"] = True
    last_row["endpoint_failover_attempts"] = max_attempts
    return last_row


def judge_pairwise_baseline(task: OpenRSPairTask, base_url: str, config: dict[str, Any]) -> dict[str, Any]:
    messages = format_baseline_messages(task)
    started_at = time.time()
    tools = [pairwise_final_answer_tool(task.benchmark)] if bool(config.get("baseline_use_final_answer_tool", False)) else None
    baseline_tool_choice = config.get("baseline_tool_choice", config.get("tool_choice", "auto"))
    response = call_with_retries(
        base_url,
        messages,
        config,
        tools=tools,
        tool_choice=baseline_tool_choice if tools else None,
    )
    final_tool = first_final_answer_tool_call(response.get("tool_calls") or [])
    parsed = parse_pairwise_final_answer_tool_call(final_tool) if final_tool is not None else parse_pairwise_final(
        response.get("content", ""),
        finish_reason=response.get("finish_reason"),
    )
    if not pairwise_final_valid_for_benchmark(task.benchmark, parsed) and bool(config.get("enable_forced_finalization", True)):
        if is_rmbench(task.benchmark):
            forced_instruction = (
                "No more reasoning. Based only on the user request, Response A, Response B, and any prior analysis "
                "above, reply with exactly one label and nothing else: A>>B, A>B, A=B, B>A, or B>>A."
            )
        elif is_judgebench(task.benchmark):
            forced_instruction = (
                "No more reasoning. Based only on the instruction, Output (a), and Output (b), reply with exactly "
                "one label and nothing else: Output (a) or Output (b)."
            )
        else:
            forced_instruction = (
                "No more reasoning. Based only on the user request and the two responses, reply with exactly one "
                "line and nothing else: Final: A, Final: B, or Final: Tie."
            )
        forced_messages = list(messages)
        if response.get("content") and not is_rmbench(task.benchmark):
            forced_messages.append({"role": "assistant", "content": response.get("content", "")})
        forced_messages.append({"role": "user", "content": forced_instruction})
        forced_config = dict(config)
        forced_config["max_tokens"] = int(config.get("forced_finalization_max_tokens", 16))
        forced_tools = None
        forced_tool_choice = None
        if bool(config.get("forced_finalization_use_final_answer_tool", False)):
            forced_tools = [pairwise_final_answer_tool(task.benchmark)]
            forced_tool_choice = {"type": "function", "function": {"name": "final_answer"}}
        forced = call_with_retries(
            base_url,
            forced_messages,
            forced_config,
            tools=forced_tools,
            tool_choice=forced_tool_choice,
        )
        forced_final_tool = first_final_answer_tool_call(forced.get("tool_calls") or [])
        forced_parsed = (
            parse_pairwise_final_answer_tool_call(forced_final_tool)
            if forced_final_tool is not None
            else parse_pairwise_final(forced.get("content", ""), finish_reason=forced.get("finish_reason"))
        )
        if pairwise_final_valid_for_benchmark(task.benchmark, forced_parsed):
            response = forced
            parsed = forced_parsed
    winner = parsed["winner"]
    valid = pairwise_final_valid_for_benchmark(task.benchmark, parsed)
    return build_pairwise_row(
        task,
        base_url,
        config,
        mode="pairwise_baseline",
        winner=winner,
        verdict=parsed["verdict"],
        verdict_source=parsed["source"],
        valid=valid,
        parse_error=None if valid else "could not parse winner",
        latency_sec=time.time() - started_at,
        raw_output=response.get("content", ""),
        response=response,
    )


def judge_pairwise_self_select_skill(
    task: OpenRSPairTask,
    base_url: str,
    config: dict[str, Any],
    skill_package: dict[str, Any],
) -> dict[str, Any]:
    judgebench_profile = judgebench_system_prompt_profile(task, config) if is_judgebench(task.benchmark) else None
    use_system_prompt = not is_judgebench(task.benchmark) or judgebench_profile != "none"
    if not use_system_prompt:
        user_prompt = format_pairwise_user_prompt(task, with_skill_hint=True)
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_prompt}]
    else:
        system_prompt = format_self_select_skill_system_prompt(skill_package, config)
        if is_judgebench(task.benchmark):
            if judgebench_profile == "minimal":
                system_prompt = format_judgebench_minimal_skill_system_prompt(skill_package, config)
        user_prompt = format_pairwise_user_prompt(task, with_skill_hint=True)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    record = pairwise_task_runtime_record(task)
    formatted = {"responses": task.responses}
    skill_state: dict[str, Any] = {
        "loaded": False,
        "trigger_step": None,
        "trigger_reason": None,
        "delegated_calls": 0,
    }
    max_steps = int(config.get("max_agent_steps", 5))
    resources_viewed: list[str] = []
    tool_error_count = 0
    tool_call_count = 0
    sandbox_call_count = 0
    emergency_direct_finalization_used = False
    emergency_direct_finalization_source: str | None = None
    raw_output = ""
    final_response: dict[str, Any] = {}
    final_parsed = {"verdict": "error", "winner": "error", "source": "missing"}
    parse_error = "max agent steps exceeded"
    trace: dict[str, Any] = {
        "sample_id": task.task_id,
        "benchmark": task.benchmark,
        "mode": "self_select_skill_pairwise",
        "skill_path": skill_package["source"],
        "skill_package_sha256": skill_package["sha256"],
        "initial_context": {
            "skills_advertised": [skill_package_name(skill_package)],
            "skill_contents_loaded": bool(skill_state["loaded"]),
            "resource_index_loaded": bool(skill_state["loaded"]),
        },
        "steps": [],
    }
    started_at = time.time()
    tool_config = dict(config)
    if is_judgebench(task.benchmark):
        tool_config["_judgebench_system_prompt_active"] = use_system_prompt

    for step in range(1, max_steps + 1):
        tools = pairwise_skill_tools(skill_loaded=bool(skill_state["loaded"]), config=tool_config)
        request_tool_choice: str | dict[str, Any] | None = config.get("tool_choice", "auto")
        request_config = config
        response = call_with_retries(
            base_url,
            messages,
            request_config,
            tools=tools,
            tool_choice=request_tool_choice,
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
            final_parsed = parse_pairwise_final_answer_tool_call(final_tool)
            step_trace["final"] = final_parsed
            trace["steps"].append(step_trace)
            if pairwise_final_valid_for_benchmark(task.benchmark, final_parsed):
                parse_error = None
                break
            parse_error = f"final_answer verdict violates {task.benchmark} contract"
            final_parsed = {"verdict": "error", "winner": "error", "source": final_parsed["source"]}
            break

        if tool_calls:
            tool_call_count += len(tool_calls)
            messages.append({"role": "assistant", "content": raw_output or "", "tool_calls": tool_calls})
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

        final_parsed = parse_pairwise_final(raw_output, finish_reason=response.get("finish_reason"))
        if pairwise_final_valid_for_benchmark(task.benchmark, final_parsed):
            step_trace["final"] = final_parsed
            trace["steps"].append(step_trace)
            parse_error = None
            break
        step_trace["parse_error"] = "assistant returned neither tool call nor final verdict"
        trace["steps"].append(step_trace)
        parse_error = "assistant returned neither tool call nor final verdict"
        break

    if (
        not pairwise_final_valid_for_benchmark(task.benchmark, final_parsed)
        and bool(config.get("enable_forced_finalization", True))
        and messages
    ):
        if is_judgebench(task.benchmark):
            forced_instruction = (
                "No more tool calls are allowed. Choose exactly one forced-choice label based only on the visible "
                "instruction, Output (a), Output (b), and evidence already collected. Do not explain. Reply with "
                "exactly one line: Output (a) or Output (b)."
            )
        elif is_rmbench(task.benchmark):
            forced_instruction = (
                "No more tool calls are allowed. Choose exactly one scaled pairwise preference label based only on the visible "
                "request, responses, and evidence already collected. Do not explain. Reply with exactly one label: "
                "A>>B, A>B, A=B, B>A, or B>>A."
            )
        else:
            forced_instruction = (
                "No more tool calls are allowed. Choose Response A, Response B, or Tie based only on the visible "
                "request, responses, and evidence already collected. Do not explain. Reply with exactly one line: "
                "Final: A, Final: B, or Final: Tie."
            )
        messages.append(
            {
                "role": "user",
                "content": forced_instruction,
            }
        )
        forced_config = dict(config)
        forced_config["max_tokens"] = int(config.get("forced_finalization_max_tokens", 16))
        forced_tools = None
        forced_tool_choice = None
        if bool(config.get("forced_finalization_use_final_answer_tool", False)) or (
            is_judgebench(task.benchmark) and bool(config.get("judgebench_forced_finalization_tool", False))
        ):
            forced_tools = [pairwise_final_answer_tool(task.benchmark)]
            forced_tool_choice = {"type": "function", "function": {"name": "final_answer"}}
        forced = call_with_retries(base_url, messages, forced_config, tools=forced_tools, tool_choice=forced_tool_choice)
        raw_output = forced.get("content", "")
        final_response = forced
        forced_tool_calls = forced.get("tool_calls") or []
        forced_final_tool = first_final_answer_tool_call(forced_tool_calls)
        if forced_final_tool is not None:
            final_parsed = parse_pairwise_final_answer_tool_call(forced_final_tool)
        else:
            final_parsed = parse_pairwise_final(raw_output, finish_reason=forced.get("finish_reason"))
        if (
            forced_tools is not None
            and not pairwise_final_valid_for_benchmark(task.benchmark, final_parsed)
            and bool(config.get("forced_finalization_retry_text", True))
        ):
            if is_judgebench(task.benchmark):
                retry_contract = "Output (a) or Output (b)"
            elif is_rmbench(task.benchmark):
                retry_contract = "A>>B, A>B, A=B, B>A, or B>>A"
            else:
                retry_contract = "Final: A, Final: B, or Final: Tie"
            retry_messages = list(messages)
            retry_messages.append(
                {
                    "role": "user",
                    "content": (
                        "The previous final_answer tool call was malformed. Do not call tools now. "
                        f"Reply with exactly one of these labels and nothing else: {retry_contract}."
                    ),
                }
            )
            retry_config = dict(config)
            retry_config["max_tokens"] = int(config.get("forced_finalization_retry_max_tokens", 128))
            retry = call_with_retries(base_url, retry_messages, retry_config, tools=None, tool_choice=None)
            retry_output = retry.get("content", "")
            retry_parsed = parse_pairwise_final(retry_output, finish_reason=retry.get("finish_reason"))
            trace["steps"].append(
                {
                    "step": len(trace["steps"]) + 1,
                    "forced_finalization_retry_text": True,
                    "assistant_content": retry_output,
                    "finish_reason": retry.get("finish_reason"),
                    "latency_sec": retry.get("latency_sec"),
                    "reasoning_len": retry.get("reasoning_len", 0),
                    "request_error": retry.get("error"),
                    "tool_calls": compact_tool_calls_for_trace(retry.get("tool_calls") or []),
                    "tool_results": [],
                    "final": retry_parsed,
                }
            )
            if pairwise_final_valid_for_benchmark(task.benchmark, retry_parsed):
                raw_output = retry_output
                final_response = retry
                final_parsed = retry_parsed
        parse_error = None if pairwise_final_valid_for_benchmark(task.benchmark, final_parsed) else f"forced finalization failed after {parse_error}"
        trace["steps"].append(
            {
                "step": len(trace["steps"]) + 1,
                "forced_finalization": True,
                "assistant_content": raw_output,
                "finish_reason": forced.get("finish_reason"),
                "latency_sec": forced.get("latency_sec"),
                "reasoning_len": forced.get("reasoning_len", 0),
                "request_error": forced.get("error"),
                "tool_calls": compact_tool_calls_for_trace(forced_tool_calls),
                "tool_results": [],
                "final": final_parsed,
            }
        )

    if (
        not pairwise_final_valid_for_benchmark(task.benchmark, final_parsed)
        and bool(config.get("emergency_direct_judge_finalization", False))
    ):
        emergency_config = dict(config)
        emergency_config["baseline_use_final_answer_tool"] = True
        emergency_config["baseline_tool_choice"] = {"type": "function", "function": {"name": "final_answer"}}
        emergency_config["forced_finalization_use_final_answer_tool"] = True
        emergency_config["forced_finalization_max_tokens"] = int(
            config.get("emergency_direct_judge_max_tokens", config.get("max_tokens", 2048))
        )
        emergency_row = judge_pairwise_baseline(task, base_url, emergency_config)
        emergency_parsed = {
            "verdict": str(emergency_row.get("skill_final_verdict") or "error"),
            "winner": str(emergency_row.get("predicted_label") or "error"),
            "source": f"emergency_direct_judge.{emergency_row.get('verdict_source') or 'unknown'}",
        }
        trace["steps"].append(
            {
                "step": len(trace["steps"]) + 1,
                "emergency_direct_judge_finalization": True,
                "assistant_content": emergency_row.get("raw_output", ""),
                "finish_reason": emergency_row.get("finish_reason"),
                "latency_sec": emergency_row.get("latency_sec"),
                "reasoning_len": emergency_row.get("reasoning_len", 0),
                "request_error": emergency_row.get("request_error"),
                "tool_calls": [],
                "tool_results": [],
                "final": emergency_parsed,
            }
        )
        if pairwise_final_valid_for_benchmark(task.benchmark, emergency_parsed):
            raw_output = str(emergency_row.get("raw_output") or raw_output)
            final_response = {
                "finish_reason": emergency_row.get("finish_reason"),
                "error": emergency_row.get("request_error"),
                "reasoning_len": emergency_row.get("reasoning_len", 0),
                "thinking_field_sent": emergency_row.get("thinking_field_sent"),
            }
            final_parsed = emergency_parsed
            parse_error = None
            emergency_direct_finalization_used = True
            emergency_direct_finalization_source = emergency_parsed["source"]

    winner = final_parsed["winner"]
    valid = pairwise_final_valid_for_benchmark(task.benchmark, final_parsed)
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
        "emergency_direct_finalization_used": emergency_direct_finalization_used,
        "emergency_direct_finalization_source": emergency_direct_finalization_source,
    }
    row = build_pairwise_row(
        task,
        base_url,
        config,
        mode="self_select_skill_pairwise",
        winner=winner,
        verdict=final_parsed["verdict"],
        verdict_source=final_parsed["source"],
        valid=valid,
        parse_error=parse_error,
        latency_sec=time.time() - started_at,
        raw_output=raw_output,
        response=final_response,
    )
    row.update(
        {
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
            "emergency_direct_finalization_used": emergency_direct_finalization_used,
            "emergency_direct_finalization_source": emergency_direct_finalization_source,
            "agent_step_count": len(trace["steps"]),
            "openai_tool_calling": True,
            "trace_id": task.task_id,
            "_trace": trace,
        }
    )
    return row


def should_use_judgebench_system_prompt(task: OpenRSPairTask, config: dict[str, Any]) -> bool:
    return judgebench_system_prompt_profile(task, config) != "none"


def judgebench_system_prompt_profile(task: OpenRSPairTask, config: dict[str, Any]) -> str:
    style = str(config.get("judgebench_system_prompt_style") or "").strip().lower()
    setting = config.get("judgebench_use_system_prompt", False)
    if not bool(setting):
        return "none"
    if style == "minimal":
        return style
    return "self_select"


def format_judgebench_minimal_skill_system_prompt(skill_package: dict[str, Any], config: dict[str, Any]) -> str:
    skill_name = skill_package_name(skill_package)
    trigger_policy = str(config.get("judgebench_skill_trigger_policy") or "").lower()
    trigger_strength = str(config.get("operational_trigger_strength") or "").strip().lower()
    resource_first = (
        str(config.get("skill_allowed_setting") or "") == "skill_operational"
        and trigger_strength in {"high", "resource_first", "trigger_v1", "trigger_v2"}
    )
    verifier_priority = (
        str(config.get("skill_allowed_setting") or "") == "skill_operational"
        and str(config.get("operational_verifier_priority") or "").strip().lower() in {"high", "ground_truth", "trigger_v2"}
    )
    operational_lines: list[str] = []
    if str(config.get("skill_allowed_setting") or "") == "skill_operational":
        operational_lines = [
            "Operational resource hint: for forced-choice answer-selection, math, code, factuality, exact-format, or reasoning tasks, the optional skill may expose sample-visible task metadata, references, expected answers, checklists, or constraints after use_skill is called.",
            "Operational self-select rule: when the sample appears objective or verifiable, or when candidate final answers differ, call use_skill before finalizing unless the visible prompt and outputs alone make the winner certain. Do not assume such resources exist before loading the skill.",
        ]
        if resource_first:
            operational_lines.extend(
                [
                    "Operational trigger-v1 policy: for non-trivial forced-choice samples, prefer use_skill before final_answer so the resource interface can reveal whether sample-visible reference, expected answer, checklist, or verifier evidence exists.",
                    "Use direct final_answer without use_skill only when the winner is obvious from the visible instruction and outputs alone.",
                ]
            )
        if verifier_priority:
            operational_lines.extend(
                [
                    "Operational trigger-v2 policy: after use_skill, if a ground-truth/reference scoring verifier is listed in the resource index, prefer running it before final_answer for answer-selection, factuality, math, code, or exact-format samples.",
                    "Use verifier output only when it returns a clear A/B result, and map it back to the current Output (a)/Output (b) positions.",
                ]
            )
    if trigger_policy in {"correctness_first", "answer_selection", "operational_resource_first"} or resource_first:
        trigger_reason_parts = "visible rationale, code, arithmetic, exact format, supplied examples, or internal contradictions"
        if operational_lines:
            trigger_reason_parts = "visible rationale, code, arithmetic, exact format, supplied examples, sample-visible references, expected answers, checklists, or internal contradictions"
        trigger_lines = [
            f"Load the skill when candidate final answers differ and {trigger_reason_parts} can decide the winner.",
        ]
        if resource_first:
            trigger_lines.append(
                "For ordinary subject-matter recall, still call use_skill when the sample may have reference or expected-answer evidence behind the skill interface; otherwise decide directly."
            )
        else:
            trigger_lines.append("For ordinary subject-matter recall with no visible check, decide directly from the visible prompt and outputs.")
    else:
        trigger_reason_parts = "visible code, math, exact format, supplied examples, counts, or internal contradictions"
        if operational_lines:
            trigger_reason_parts = "visible code, math, exact format, supplied examples, supplied references, counts, or internal contradictions"
        trigger_lines = [
            f"Load the skill only for short deterministic checks over {trigger_reason_parts}.",
            "Do not load the skill merely because option letters differ, the topic is specialized, or broad background knowledge might help.",
        ]
    return "\n".join(
        [
            "You are an impartial forced-choice response judge.",
            "The user message is authoritative for the instruction text, candidate outputs, and Output (a)/Output (b) labels.",
            "This system message is authoritative for tool-use policy and finalization. Select exactly one label: Output (a) or Output (b).",
            "Internally, A maps to Output (a), and B maps to Output (b). Do not choose Tie, both, neither, or abstain.",
            "Treat Output (a) and Output (b) as arbitrary positions local to the current user message. Compare the current texts, not which side appears first.",
            "After using any skill resource, map the evidence back to the current Output (a)/Output (b) positions before calling final_answer.",
            f"An optional judging skill named `{skill_name}` is available through tools.",
            *operational_lines,
            *trigger_lines,
            "When visible evidence already gives a clear winner, answer directly without loading the skill.",
            "If you use tools, use OpenAI tool calls and finish with final_answer verdict A or B.",
            "If you do not use tools, write exactly Output (a) or Output (b), with no explanation.",
            "Use only the current prompt, outputs, and available tool evidence.",
        ]
    )

def build_pairwise_row(
    task: OpenRSPairTask,
    base_url: str,
    config: dict[str, Any],
    *,
    mode: str,
    winner: str,
    verdict: str,
    verdict_source: str,
    valid: bool,
    parse_error: str | None,
    latency_sec: float,
    raw_output: str,
    response: dict[str, Any],
) -> dict[str, Any]:
    decisive_correct = winner == task.gold_label
    decisive_wrong = winner in {"A", "B"} and winner != task.gold_label
    return {
        "sample_id": task.task_id,
        "original_sample_id": task.sample_id,
        "benchmark": task.benchmark,
        "query_type": task.query_type,
        "domain": task.domain,
        "pair": task.pair,
        "order": task.order,
        "mode": mode,
        "gold_label": task.gold_label,
        "predicted_label": winner,
        "skill_final_verdict": verdict,
        "verdict_source": verdict_source,
        "correct": decisive_correct,
        "wrong": decisive_wrong,
        "same": winner in {"Tie", "Abstain"},
        "valid": valid,
        "endpoint": base_url,
        "latency_sec": latency_sec,
        "enable_thinking": bool(config.get("enable_thinking", False)),
        "thinking_field_sent": response.get("thinking_field_sent"),
        "reasoning_len": int(response.get("reasoning_len") or 0),
        "finish_reason": response.get("finish_reason"),
        "request_error": response.get("error"),
        "raw_output": raw_output,
        "parse_error": parse_error,
    }


def pairwise_skill_tools(*, skill_loaded: bool, config: dict[str, Any]) -> list[dict[str, Any]]:
    benchmark = str(config.get("benchmark") or "")
    is_operational = str(config.get("skill_allowed_setting") or "") == "skill_operational"
    trigger_strength = str(config.get("operational_trigger_strength") or "").strip().lower()
    resource_first = is_operational and trigger_strength in {"high", "resource_first", "trigger_v1", "trigger_v2"}
    verifier_priority = is_operational and str(config.get("operational_verifier_priority") or "").strip().lower() in {
        "high",
        "ground_truth",
        "trigger_v2",
    }
    final_tool_benchmark = benchmark
    final_answer = pairwise_final_answer_tool(final_tool_benchmark)
    use_skill_description = "Load the optional judging skill when its instructions, resources, or deterministic checks may improve the pairwise judgment."
    if is_operational:
        use_skill_description = (
            "Load the optional operational judging skill when resource-rich evidence or deterministic checks may improve the pairwise judgment, "
            "especially objective-answer, math, code, factuality, exact-format, checklist, or instruction-following tasks. "
            "After loading, inspect the resource index for sample-visible metadata, references, expected answers, checklist, or constraints."
        )
        if resource_first:
            use_skill_description = (
                "Load the operational judging skill before final_answer for non-trivial pairwise comparisons where resource-rich "
                "evidence, rubric guidance, sample-visible reference/expected answer, checklist, verifier output, or a deterministic "
                "check could plausibly change the winner. Use direct final_answer only when the winner is obvious from visible text alone."
            )
        if verifier_priority:
            use_skill_description += (
                " After loading the skill, if a ground-truth/reference scoring verifier is listed, prefer running it for objective "
                "answer-selection, factuality, math, code, or exact-format samples before final_answer."
            )
    if is_judgebench(benchmark):
        trigger_policy = str(config.get("judgebench_skill_trigger_policy") or "").lower()
        if trigger_policy in {"correctness_first", "answer_selection"}:
            visible_checks = "visible question, options, code, arithmetic, examples, counts, formats, or internal contradictions"
            if is_operational:
                visible_checks = "visible question, options, code, arithmetic, examples, counts, formats, supplied references, or internal contradictions"
            use_skill_description = (
                "Load the optional judging skill for forced-choice answer-selection, math, code, exact-format, or "
                "logic tasks when candidate final answers differ, visible rationales conflict, or a concise "
                f"correctness-first check over {visible_checks} may decide the winner."
            )
        else:
            visible_checks = "visible code, arithmetic, examples, counts, formats, or internal contradictions"
            if is_operational:
                visible_checks = "visible code, arithmetic, examples, counts, formats, supplied references, or internal contradictions"
            use_skill_description = (
                "Load the optional judging skill for forced-choice programming, code review, math, exact-format, or "
                f"logic tasks only when a short deterministic check over {visible_checks} can decide the winner. "
                "For exam-style answer-selection or subject-matter factual recall, prefer final_answer directly; "
                "do not load skill merely because option letters differ, the domain is specialized, or broad background knowledge might help."
            )
        if is_operational:
            use_skill_description += (
                " In the operational setting, loaded resources may include sample-visible task metadata, references, expected answers, "
                "checklist, or constraints. For JudgeBench-style objective answer selection, call use_skill when final answers differ "
                "unless visible evidence alone already makes the winner certain."
            )
            if resource_first:
                use_skill_description += (
                    " Trigger-v1: prefer use_skill for non-trivial forced-choice samples because reference or expected-answer evidence may be available only after the skill is loaded."
                )
    if not skill_loaded:
        return [
            {
                "type": "function",
                "function": {
                    "name": "use_skill",
                    "description": use_skill_description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "skill_name": {"type": "string"},
                            "reason": {"type": "string"},
                        },
                        "required": ["skill_name", "reason"],
                        "additionalProperties": False,
                    },
                },
            },
            final_answer,
        ]

    tools: list[dict[str, Any]] = [
        {
            "type": "function",
            "function": {
                "name": "list_resources",
                "description": "List available skill resources after the skill has been loaded.",
                "parameters": {
                    "type": "object",
                    "properties": {"type": {"type": ["string", "null"]}},
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "view_resource",
                "description": "Read one skill resource by path. Use only resources needed for the current pairwise sample.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}, "reason": {"type": "string"}},
                    "required": ["path", "reason"],
                    "additionalProperties": False,
                },
            },
        },
    ]
    if runtime_resource_tools_enabled(config):
        tools.append(runtime_resource_tool_schema())
    if bool(config.get("enable_python_sandbox", True)):
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": "python_sandbox",
                    "description": "Run small Python checks over the visible request and Response A/B. No network, files, subprocesses, or external data are available.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string"},
                            "reason": {"type": "string"},
                        },
                        "required": ["code", "reason"],
                        "additionalProperties": False,
                    },
                },
            }
        )
    tools.append(final_answer)
    return tools


def write_openrs_summary(
    path: Path,
    config: dict[str, Any],
    metrics: dict[str, Any],
    examples: int,
    completed: int,
    elapsed: float,
) -> None:
    overall = metrics.get("overall") or metrics.get("task_level", {}).get("overall", {})
    lines = [
        "# Skill-RM OpenRS Benchmark Summary",
        "",
        f"- created_at: {datetime.now(timezone.utc).isoformat()}",
        f"- benchmark: {config.get('benchmark')}",
        f"- data_source: {config.get('data_source')}",
        f"- data_sources: {config.get('data_sources')}",
        f"- evaluation_mode: {config.get('evaluation_mode')}",
        f"- model: {config.get('model')}",
        f"- examples: {examples}",
        f"- completed: {completed}",
        f"- elapsed_sec: {elapsed:.2f}",
        f"- acc_rate: {overall.get('acc_rate')}",
        f"- raw_accuracy: {overall.get('raw_accuracy')}",
        f"- same_rate: {overall.get('same_rate')}",
        f"- invalid_rate: {overall.get('invalid_rate')}",
    ]
    if "rmbench" in metrics:
        lines.extend(["", "RM-Bench global by mode:", ""])
        for mode, value in metrics["rmbench"]["global"]["by_mode"].items():
            lines.append(f"- {mode}: win_rate={value.get('win_rate')}, net_win_rate={value.get('net_win_rate')}, tie_rate={value.get('tie_rate')}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_completed_rows(path: Path) -> dict[str, dict[str, Any]]:
    return load_jsonl_map(path, key="sample_id", missing_ok=True, skip_blank=True)


if __name__ == "__main__":
    main()
