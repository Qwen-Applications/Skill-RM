from __future__ import annotations

import re
from typing import Any

from ...common.parsing import parse_first_json_object
from ...common.tool_calls import parse_tool_call_arguments


def parse_agentic_final(raw_output: str) -> dict[str, str]:
    final_matches = re.findall(r"(?im)^\s*Final:\s*(A|B|C|D|Tie|Abstain)\s*\.?\s*$", raw_output)
    if final_matches:
        verdict = final_matches[-1]
        return {"verdict": verdict, "winner": verdict if verdict in {"A", "B", "C", "D"} else "error", "source": "final_line"}

    official_winner = parse_official_winner(raw_output)
    if official_winner in {"A", "B", "C", "D"}:
        return {"verdict": official_winner, "winner": official_winner, "source": "official_bracket"}

    parsed = parse_first_json_object(raw_output)
    if parsed.get("action") and parsed.get("action") != "final":
        return {"verdict": "error", "winner": "error", "source": "tool_action"}
    if parsed.get("action") == "final":
        final_payload = parsed.get("judgment_package") if isinstance(parsed.get("judgment_package"), dict) else parsed
        verdict = final_payload.get("verdict") if isinstance(final_payload, dict) else None
        if isinstance(verdict, str):
            normalized = verdict.strip()
            upper = normalized.upper()
            if upper in {"A", "B", "C", "D"}:
                return {"verdict": upper, "winner": upper, "source": "json.action_final.verdict"}
            if normalized.lower() in {"tie", "abstain"}:
                return {"verdict": normalized.title(), "winner": "error", "source": "json.action_final.verdict"}
    verdict = parsed.get("verdict")
    if isinstance(verdict, str):
        normalized = verdict.strip()
        upper = normalized.upper()
        if upper in {"A", "B", "C", "D"}:
            return {"verdict": upper, "winner": upper, "source": "json.verdict"}
        if normalized.lower() in {"tie", "abstain"}:
            return {"verdict": normalized.title(), "winner": "error", "source": "json.verdict"}
    return {"verdict": "error", "winner": "error", "source": "unparsed"}


def parse_final_answer_tool_call(tool_call: dict[str, Any]) -> dict[str, str]:
    args, arg_error = parse_tool_call_arguments(tool_call)
    if arg_error:
        return {"verdict": "error", "winner": "error", "source": f"tool.final_answer.error:{arg_error}"}
    for key in ("verdict", "selected", "best_label", "winner"):
        value = args.get(key)
        if isinstance(value, str):
            normalized = value.strip()
            upper = normalized.upper()
            if upper in {"A", "B", "C", "D"}:
                return {"verdict": upper, "winner": upper, "source": f"tool.final_answer.{key}"}
    judgment = args.get("judgment_package")
    if isinstance(judgment, dict):
        for key in ("verdict", "selected", "best_label", "winner"):
            value = judgment.get(key)
            if isinstance(value, str):
                upper = value.strip().upper()
                if upper in {"A", "B", "C", "D"}:
                    return {"verdict": upper, "winner": upper, "source": f"tool.final_answer.judgment_package.{key}"}
    return {"verdict": "error", "winner": "error", "source": "tool.final_answer.unparsed"}


def parse_official_winner(judgment: str) -> str:
    for label in ("A", "B", "C", "D"):
        if f"[[{label}]]" in judgment:
            return label
    return "error"


def official_ranking_score(winner: str, chosen_label: str) -> float:
    if winner == chosen_label:
        return 1.0
    if winner in {"A", "B", "C", "D"}:
        return 0.0
    return 0.25


def parse_official_rating(judgment: str) -> int:
    match = re.search(r"\b([1-9]|10)\b\s*$", judgment.strip())
    if not match:
        return -1
    rating = int(match.group(1))
    return rating if 1 <= rating <= 10 else -1
