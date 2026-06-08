from __future__ import annotations

import json
import re
from typing import Any

from skillrm.common.tool_calls import parse_tool_call_arguments, tool_call_name


LABELS = ("A", "B")


def final_answer_tool_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": "Submit the forced-choice winner for this pairwise knockout match.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "enum": list(LABELS),
                        "description": "Winner label. Must be exactly A or B.",
                    },
                    "rationale": {
                        "type": "string",
                        "description": "Brief reason for the choice.",
                    },
                },
                "required": ["answer"],
            },
        },
    }


def use_skill_tool_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "use_skill",
            "description": "Load the reward-judge skill instructions before deciding.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    }


def list_resources_tool_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "list_resources",
            "description": "List available resources attached to the current sample.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    }


def view_resource_tool_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "view_resource",
            "description": "View one resource attached to the current sample.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }


def python_sandbox_tool_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "python_sandbox",
            "description": "Run a short Python calculation for local reasoning.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to run."},
                },
                "required": ["code"],
            },
        },
    }


def run_resource_tool_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "run_resource",
            "description": "Run an operational resource attached to this sample.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "arguments": {"type": "object"},
                },
                "required": ["id"],
            },
        },
    }


def extract_final_label_from_text(text: str) -> str | None:
    if not text:
        return None
    patterns = [
        r"(?im)^\s*Final\s*[:：]\s*([AB])\b",
        r"(?im)^\s*Answer\s*[:：]\s*([AB])\b",
        r"(?im)^\s*Winner\s*[:：]\s*([AB])\b",
        r"(?im)\bfinal_answer\s*[:：]\s*([AB])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).upper()
    stripped = text.strip().upper()
    if stripped in LABELS:
        return stripped
    try:
        obj = json.loads(text)
    except Exception:
        obj = None
    if isinstance(obj, dict):
        for key in ("answer", "label", "winner", "choice"):
            value = obj.get(key)
            if isinstance(value, str) and value.strip().upper() in LABELS:
                return value.strip().upper()
    return None


def final_from_tool_calls(tool_calls: list[Any] | None) -> tuple[str | None, dict[str, Any] | None]:
    if not tool_calls:
        return None, None
    for call in tool_calls:
        if tool_call_name(call) != "final_answer":
            continue
        args, arg_error = parse_tool_call_arguments(call)
        if arg_error:
            continue
        if isinstance(args, dict):
            answer = args.get("answer")
            if isinstance(answer, str) and answer.strip().upper() in LABELS:
                return answer.strip().upper(), args
    return None, None
