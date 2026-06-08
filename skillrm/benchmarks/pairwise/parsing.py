from __future__ import annotations

import re
from typing import Any

from ...common.parsing import parse_first_json_object
from ...common.tool_calls import parse_tool_call_arguments
from .data import is_judgebench, is_rmbench


def pairwise_final_answer_tool(benchmark: str = "") -> dict[str, Any]:
    verdict_enum = ["A", "B", "Tie", "Abstain"]
    description = "Submit the final pairwise judgment."
    if is_judgebench(benchmark):
        verdict_enum = ["A", "B"]
        description = "Submit exactly one forced-choice winner. For exam-style answer-selection, use this directly after comparing final selected options, visible rationale, and your own judgment."
    if is_rmbench(benchmark):
        verdict_enum = ["A>>B", "A>B", "A=B", "B>A", "B>>A", "A", "B"]
        return {
            "type": "function",
            "function": {
                "name": "final_answer",
                "description": "Submit exactly one scaled pairwise preference label.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "verdict": {"type": "string", "enum": verdict_enum},
                    },
                    "required": ["verdict"],
                    "additionalProperties": False,
                },
            },
        }
    return {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "verdict": {"type": "string", "enum": verdict_enum},
                    "rationale": {"type": "string"},
                    "judgment_package": {"type": "object"},
                },
                "required": ["verdict"],
                "additionalProperties": True,
            },
        },
    }


def parse_pairwise_final_answer_tool_call(tool_call: dict[str, Any]) -> dict[str, str]:
    args, arg_error = parse_tool_call_arguments(tool_call)
    if arg_error:
        return {"verdict": "error", "winner": "error", "source": f"tool.final_answer.error:{arg_error}"}
    for key in ("verdict", "selected", "best_label", "winner"):
        value = args.get(key)
        parsed = normalize_pairwise_verdict(value)
        if parsed:
            verdict, winner = parsed
            return {"verdict": verdict, "winner": winner, "source": f"tool.final_answer.{key}"}
    judgment = args.get("judgment_package")
    if isinstance(judgment, dict):
        for key in ("verdict", "selected", "best_label", "winner"):
            parsed = normalize_pairwise_verdict(judgment.get(key))
            if parsed:
                verdict, winner = parsed
                return {"verdict": verdict, "winner": winner, "source": f"tool.final_answer.judgment_package.{key}"}
    return {"verdict": "error", "winner": "error", "source": "tool.final_answer.unparsed"}


def parse_pairwise_final(raw_output: str, *, finish_reason: str | None = None) -> dict[str, str]:
    tail = raw_output[-500:] if finish_reason == "length" else raw_output
    rm_label = re.findall(r"(?i)\b(A\s*>>\s*B|A\s*>\s*B|A\s*=\s*B|B\s*>\s*A|B\s*>>\s*A)\b", raw_output)
    tail_rm_label = re.findall(r"(?i)\b(A\s*>>\s*B|A\s*>\s*B|A\s*=\s*B|B\s*>\s*A|B\s*>>\s*A)\b", tail)
    if tail_rm_label:
        parsed = normalize_pairwise_verdict(tail_rm_label[-1])
        if parsed:
            verdict, winner = parsed
            return {"verdict": verdict, "winner": winner, "source": "rmbench_label"}
    if rm_label and finish_reason != "length":
        parsed = normalize_pairwise_verdict(rm_label[-1])
        if parsed:
            verdict, winner = parsed
            return {"verdict": verdict, "winner": winner, "source": "rmbench_label"}

    output_label = re.findall(r"(?i)Output\s*\(([ab])\)", tail)
    if output_label:
        verdict = output_label[-1].upper()
        return {"verdict": verdict, "winner": verdict, "source": "judgebench_output_label"}

    final_matches = re.findall(r"(?im)^\s*Final:\s*(A|B|Tie|Abstain)\s*\.?\s*$", tail)
    if final_matches:
        parsed = normalize_pairwise_verdict(final_matches[-1])
        if parsed:
            verdict, winner = parsed
            return {"verdict": verdict, "winner": winner, "source": "final_line"}

    bracket = re.findall(r"\[\[(A|B|Tie|Abstain)\]\]", tail, flags=re.IGNORECASE)
    if bracket:
        parsed = normalize_pairwise_verdict(bracket[-1])
        if parsed:
            verdict, winner = parsed
            return {"verdict": verdict, "winner": winner, "source": "official_bracket"}

    if finish_reason == "length":
        return {"verdict": "error", "winner": "error", "source": "truncated_unparsed"}

    parsed = parse_first_json_object(raw_output)
    for key in ("winner", "verdict", "best_label", "choice", "answer"):
        parsed_verdict = normalize_pairwise_verdict(parsed.get(key))
        if parsed_verdict:
            verdict, winner = parsed_verdict
            return {"verdict": verdict, "winner": winner, "source": f"json.{key}"}

    return {"verdict": "error", "winner": "error", "source": "unparsed"}


def normalize_pairwise_verdict(value: Any) -> tuple[str, str] | None:
    if not isinstance(value, str):
        return None
    compact = re.sub(r"\s+", "", value.strip().upper())
    if compact in {"A>>B", "A>B"}:
        return compact, "A"
    if compact in {"B>A", "B>>A"}:
        return compact, "B"
    if compact in {"A=B", "A==B"}:
        return "A=B", "Tie"
    winner = normalize_pairwise_winner(value)
    if winner:
        return winner, winner
    return None


def pairwise_final_valid_for_benchmark(benchmark: str, parsed: dict[str, str]) -> bool:
    winner = parsed.get("winner")
    verdict = re.sub(r"\s+", "", str(parsed.get("verdict") or "").upper())
    if is_judgebench(benchmark):
        return winner in {"A", "B"}
    if is_rmbench(benchmark):
        return winner in {"A", "B"} or verdict == "A=B"
    return winner in {"A", "B", "Tie", "Abstain"}


def normalize_pairwise_winner(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    upper = text.upper()
    compact = re.sub(r"\s+", "", upper)
    if compact in {"A>>B", "A>B"}:
        return "A"
    if compact in {"B>A", "B>>A"}:
        return "B"
    if compact in {"A=B", "A==B"}:
        return "Tie"
    if upper in {"A", "RESPONSE A", "ASSISTANT A"}:
        return "A"
    if upper in {"B", "RESPONSE B", "ASSISTANT B"}:
        return "B"
    if text.lower() in {"tie", "same", "draw", "equal", "no preference"}:
        return "Tie"
    if text.lower() in {"abstain", "unclear", "cannot determine"}:
        return "Abstain"
    match = re.search(r"\b(winner|best|answer|choice|verdict)\b[^AB]{0,40}\b([AB])\b", text, re.IGNORECASE)
    if match:
        return match.group(2).upper()
    if re.search(r"\b(response|assistant)\s+A\b", text, re.IGNORECASE):
        return "A"
    if re.search(r"\b(response|assistant)\s+B\b", text, re.IGNORECASE):
        return "B"
    return None
