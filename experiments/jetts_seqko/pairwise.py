from __future__ import annotations

import json
import time
from typing import Any

from skillrm.common.llm_client import call_with_retries
from skillrm.common.tool_calls import (
    compact_tool_calls_for_trace,
    compact_tool_result_for_trace,
    tool_call_name,
)
from skillrm.runners.rewardbench2 import execute_openai_skill_tool_call

from .prompts import baseline_system_prompt, format_pair_prompt, format_skill_system_prompt
from .io_utils import json_default
from .tool_schemas import (
    LABELS,
    extract_final_label_from_text,
    final_answer_tool_schema,
    final_from_tool_calls,
    list_resources_tool_schema,
    python_sandbox_tool_schema,
    run_resource_tool_schema,
    use_skill_tool_schema,
    view_resource_tool_schema,
)


def baseline_pairwise(
    prompt: str,
    response_a: str,
    response_b: str,
    base_url: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    use_final_answer_tool = bool(config.get("use_final_answer_tool", False))
    messages = [
        {"role": "system", "content": baseline_system_prompt(config)},
        {"role": "user", "content": format_pair_prompt(prompt, response_a, response_b)},
    ]
    start = time.time()
    row: dict[str, Any] = {
        "mode": "baseline",
        "base_url": base_url,
        "valid": False,
        "predicted_label": None,
        "request_error": None,
        "latency_s": None,
        "request_count": 1,
        "tool_call_count": 0,
        "used_final_answer_tool": use_final_answer_tool,
        "_trace": [],
    }
    try:
        completion = call_with_retries(
            base_url,
            messages,
            config,
            tools=[final_answer_tool_schema()] if use_final_answer_tool else None,
            tool_choice="auto" if use_final_answer_tool else None,
        )
    except Exception as exc:
        row["request_error"] = repr(exc)
        row["latency_s"] = time.time() - start
        return row

    tool_calls = completion.get("tool_calls") or []
    row["tool_call_count"] += len(tool_calls)
    tool_label, tool_args = final_from_tool_calls(tool_calls)
    content = completion.get("content", "") or ""
    text_label = extract_final_label_from_text(content)
    label = tool_label or text_label
    trace = [
        {
            "step": 1,
            "tool_calls": compact_tool_calls_for_trace(tool_calls),
            "content": content,
            "finish_reason": completion.get("finish_reason"),
        }
    ]

    if (
        label not in LABELS
        and not use_final_answer_tool
        and config.get("enable_no_tool_forced_finalization", True)
        and completion.get("error") is None
    ):
        row["request_count"] += 1
        forced_messages = messages + [
            {
                "role": "assistant",
                "content": content,
            },
            {
                "role": "user",
                "content": (
                    "Your previous response did not contain a valid winner. "
                    "You must choose exactly one candidate for this knockout match. "
                    "Reply with exactly `Final: A` or `Final: B`, and nothing else."
                ),
            },
        ]
        try:
            forced_completion = call_with_retries(
                base_url,
                forced_messages,
                config,
                tools=None,
                tool_choice=None,
            )
            forced_content = forced_completion.get("content", "") or ""
            forced_label = extract_final_label_from_text(forced_content)
            trace.append(
                {
                    "step": "forced_final",
                    "tool_calls": [],
                    "content": forced_content,
                    "finish_reason": forced_completion.get("finish_reason"),
                }
            )
            if forced_label in LABELS:
                label = forced_label
                content = forced_content
                completion = forced_completion
            elif forced_completion.get("error"):
                row["request_error"] = forced_completion.get("error")
                completion = forced_completion
                content = forced_content
        except Exception as exc:
            row["request_error"] = repr(exc)

    row.update(
        {
            "valid": label in LABELS,
            "predicted_label": label if label in LABELS else None,
            "request_error": row.get("request_error") or completion.get("error"),
            "raw_content": content,
            "tool_args": tool_args,
            "finish_reason": completion.get("finish_reason"),
            "latency_s": time.time() - start,
            "_trace": trace,
        }
    )
    return row


def openai_tool_schemas(skill_loaded: bool, config: dict[str, Any]) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = [final_answer_tool_schema()]
    if not skill_loaded:
        tools.insert(0, use_skill_tool_schema())
        return tools
    tools.insert(0, list_resources_tool_schema())
    tools.insert(1, view_resource_tool_schema())
    if config.get("enable_run_resource"):
        tools.insert(2, run_resource_tool_schema())
    if config.get("enable_python_sandbox", True):
        tools.insert(2, python_sandbox_tool_schema())
    return tools


def add_output_contract(tool_result: Any) -> dict[str, Any]:
    if isinstance(tool_result, dict):
        out = dict(tool_result)
    else:
        out = {"result": tool_result}
    out.setdefault(
        "adapter_output_contract",
        "For this JETTS sequential-KO adapter, final_answer.answer must be exactly A or B. Tie/Abstain are invalid.",
    )
    return out


def skill_pairwise(
    prompt: str,
    response_a: str,
    response_b: str,
    dataset: str,
    sample_id: str,
    base_url: str,
    config: dict[str, Any],
    skill_package: dict[str, Any],
) -> dict[str, Any]:
    record = {
        "id": sample_id,
        "prompt": prompt,
        "benchmark": "JETTS",
        "subset": dataset,
        "source": "jetts_seqko",
    }
    formatted = {
        "responses": {"A": response_a, "B": response_b},
        "user_prompt": format_pair_prompt(prompt, response_a, response_b),
    }
    skill_state = {
        "loaded": False,
        "viewed_resource_ids": [],
        "run_resource_count": 0,
    }
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": format_skill_system_prompt(str(skill_package.get("name") or "reward_judge"))},
        {"role": "user", "content": format_pair_prompt(prompt, response_a, response_b)},
    ]
    trace: list[dict[str, Any]] = []
    start = time.time()
    row: dict[str, Any] = {
        "mode": "skill",
        "base_url": base_url,
        "valid": False,
        "predicted_label": None,
        "request_error": None,
        "latency_s": None,
        "request_count": 0,
        "tool_call_count": 0,
        "skill_loaded": False,
        "_trace": trace,
    }

    max_steps = int(config.get("max_agent_steps", 6))
    last_content = ""
    for step in range(1, max_steps + 1):
        row["request_count"] += 1
        try:
            completion = call_with_retries(
                base_url,
                messages,
                config,
                tools=openai_tool_schemas(bool(skill_state["loaded"]), config),
                tool_choice="auto",
            )
        except Exception as exc:
            row["request_error"] = repr(exc)
            break

        if completion.get("error"):
            row["request_error"] = completion.get("error")
        tool_calls = completion.get("tool_calls") or []
        content = completion.get("content", "") or ""
        last_content = content
        finish_reason = completion.get("finish_reason")
        trace_step = {
            "step": step,
            "finish_reason": finish_reason,
            "content": content,
            "tool_calls": compact_tool_calls_for_trace(tool_calls),
            "tool_results": [],
        }
        trace.append(trace_step)

        label, tool_args = final_from_tool_calls(tool_calls)
        if label in LABELS:
            row.update(
                {
                    "valid": True,
                    "predicted_label": label,
                    "tool_args": tool_args,
                    "finish_reason": finish_reason,
                }
            )
            break
        text_label = extract_final_label_from_text(content)
        if text_label in LABELS and not tool_calls:
            row.update({"valid": True, "predicted_label": text_label, "finish_reason": finish_reason})
            break

        if not tool_calls:
            continue

        messages.append(
            {
                "role": "assistant",
                "content": content or None,
                "tool_calls": tool_calls,
            }
        )
        row["tool_call_count"] += len(tool_calls)
        for call in tool_calls:
            result = execute_openai_skill_tool_call(
                call,
                skill_state,
                skill_package,
                record,
                formatted,
                config,
                skill_state["viewed_resource_ids"],
                step,
            )
            result = add_output_contract(result)
            trace_step["tool_results"].append(
                {"name": tool_call_name(call), "result": compact_tool_result_for_trace(result)}
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id"),
                    "content": json.dumps(result, ensure_ascii=False, default=json_default),
                }
            )
        row["skill_loaded"] = bool(skill_state["loaded"])

    if not row["valid"] and config.get("enable_forced_finalization", True) and row.get("request_error") is None:
        row["request_count"] += 1
        forced_messages = messages + [
            {
                "role": "user",
                "content": "No more tools. You must choose the knockout winner. Reply exactly `Final: A` or `Final: B`.",
            }
        ]
        try:
            completion = call_with_retries(base_url, forced_messages, config, tools=None, tool_choice=None)
            content = completion.get("content", "") or ""
            label = extract_final_label_from_text(content)
            trace.append(
                {
                    "step": "forced_final",
                    "finish_reason": completion.get("finish_reason"),
                    "content": content,
                    "tool_calls": [],
                    "tool_results": [],
                }
            )
            if label in LABELS:
                row.update(
                    {
                        "valid": True,
                        "predicted_label": label,
                        "finish_reason": completion.get("finish_reason"),
                    }
                )
            if completion.get("error"):
                row["request_error"] = completion.get("error")
            last_content = content
        except Exception as exc:
            row["request_error"] = repr(exc)

    row["raw_content"] = last_content
    row["latency_s"] = time.time() - start
    row["skill_loaded"] = bool(skill_state["loaded"])
    return row


def choose_base_url(base_urls: list[str], start: int, attempt: int) -> str:
    return base_urls[(start + attempt) % len(base_urls)]


def pairwise_with_failover(
    setting_cfg: dict[str, Any],
    mode: str,
    prompt: str,
    response_a: str,
    response_b: str,
    dataset: str,
    sample_id: str,
    base_urls: list[str],
    endpoint_start: int,
    skill_package: dict[str, Any] | None,
) -> dict[str, Any]:
    attempts = max(1, int(setting_cfg.get("endpoint_failover_attempts", 1)))
    last_row: dict[str, Any] | None = None
    aggregate_trace: list[dict[str, Any]] = []
    total_request_count = 0
    total_tool_call_count = 0

    for attempt in range(attempts):
        base_url = choose_base_url(base_urls, endpoint_start, attempt)
        if mode == "baseline":
            row = baseline_pairwise(prompt, response_a, response_b, base_url, setting_cfg)
        else:
            assert skill_package is not None
            row = skill_pairwise(prompt, response_a, response_b, dataset, sample_id, base_url, setting_cfg, skill_package)
        total_request_count += int(row.get("request_count") or 0)
        total_tool_call_count += int(row.get("tool_call_count") or 0)
        aggregate_trace.append(
            {
                "attempt": attempt + 1,
                "base_url": base_url,
                "valid": row.get("valid"),
                "predicted_label": row.get("predicted_label"),
                "request_error": row.get("request_error"),
                "trace": row.pop("_trace", []),
            }
        )
        last_row = row
        if row.get("valid"):
            break

    assert last_row is not None
    last_row["_trace"] = aggregate_trace
    last_row["request_count"] = total_request_count
    last_row["tool_call_count"] = total_tool_call_count
    last_row["failover_attempts"] = len(aggregate_trace)
    return last_row
