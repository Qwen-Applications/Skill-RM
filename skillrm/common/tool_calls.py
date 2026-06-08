from __future__ import annotations

import json
from typing import Any


def tool_call_name(tool_call: dict[str, Any]) -> str:
    function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
    return str(function.get("name") or tool_call.get("name") or "")


def parse_tool_call_arguments(tool_call: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    function = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
    raw_args = function.get("arguments", {})
    if isinstance(raw_args, dict):
        return raw_args, None
    if raw_args in (None, ""):
        return {}, None
    if not isinstance(raw_args, str):
        return {}, f"tool arguments are not JSON object/string: {type(raw_args).__name__}"
    try:
        parsed = json.loads(raw_args)
    except json.JSONDecodeError as exc:
        return {}, f"tool arguments JSON decode failed: {exc}"
    if not isinstance(parsed, dict):
        return {}, "tool arguments JSON is not an object"
    return parsed, None


def first_final_answer_tool_call(tool_calls: list[dict[str, Any]]) -> dict[str, Any] | None:
    for tool_call in tool_calls:
        if tool_call_name(tool_call) == "final_answer":
            return tool_call
    return None


def compact_tool_calls_for_trace(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact = []
    for tool_call in tool_calls:
        args, arg_error = parse_tool_call_arguments(tool_call)
        compact.append(
            {
                "id": tool_call.get("id"),
                "type": tool_call.get("type"),
                "name": tool_call_name(tool_call),
                "arguments": args if not arg_error else None,
                "argument_error": arg_error,
            }
        )
    return compact


def compact_tool_result_for_trace(tool_result: dict[str, Any]) -> dict[str, Any]:
    compact = dict(tool_result)
    if "content" in compact:
        compact["chars_returned"] = len(str(compact["content"]))
        compact.pop("content", None)
    if "skill_controller" in compact:
        compact["skill_controller_chars_returned"] = len(str(compact["skill_controller"]))
        compact.pop("skill_controller", None)
    if "resource_index" in compact:
        compact["resource_index_count"] = len(compact["resource_index"]) if isinstance(compact["resource_index"], list) else None
        compact.pop("resource_index", None)
    if "raw_output" in compact:
        raw = str(compact.get("raw_output") or "")
        compact["raw_output_chars"] = len(raw)
        compact["raw_output_preview"] = raw[:500]
        compact.pop("raw_output", None)
    return compact

