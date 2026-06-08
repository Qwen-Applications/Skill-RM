"""Pointwise Skill-RM judge loop."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from .llm import ChatBackend
from .resources import (
    ResourceRouter,
    compact_tool_calls_for_trace,
    compact_tool_result_for_trace,
    first_tool_call,
    parse_tool_call_arguments,
    tool_call_name,
)
from .types import PointwiseSample, ScoreVerdict
from .utils import clamp_float, first_json_object, optional_int, score_from_counts


class PointwiseSkillRMJudge:
    def __init__(self, *, backend: ChatBackend, router: ResourceRouter, variant_config: dict[str, Any]) -> None:
        self.backend = backend
        self.router = router
        self.variant_config = variant_config

    def score_sample(self, sample: PointwiseSample, *, prompt_resources_section: str = "") -> ScoreVerdict:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.router.build_system_prompt(sample)},
            {"role": "user", "content": self.router.build_user_prompt(sample, prompt_resources_section=prompt_resources_section)},
        ]
        skill_state: dict[str, Any] = {"loaded": False, "trigger_step": None, "trigger_reason": None}
        resources_viewed: list[str] = []
        resources_run: list[str] = []
        raw_output = ""
        max_steps = int(self.variant_config.get("max_agent_steps", 5))
        max_invalid_retries = int(self.variant_config.get("invalid_final_answer_retries", 1))
        invalid_retries = 0
        trace: dict[str, Any] = {
            "sample_id": sample.sample_id,
            "variant": self.variant_config.get("variant"),
            "skill_name": self.router.skill_name(),
            "steps": [],
        }

        final: ScoreVerdict | None = None
        for step in range(1, max_steps + 1):
            tools = self.router.tools(skill_loaded=bool(skill_state["loaded"]))
            try:
                response = self.backend.complete_response(
                    messages,
                    max_tokens=int(self.variant_config.get("max_tokens", 2048)),
                    tools=tools,
                    tool_choice=self.variant_config.get("tool_choice", "auto"),
                )
            except Exception as exc:  # noqa: BLE001
                return fallback_verdict(
                    sample=sample,
                    reason="backend error",
                    error=str(exc),
                    trace=trace,
                    resources=resources_viewed + resources_run,
                )

            raw_output = str(response.get("content") or "")
            tool_calls = list(response.get("tool_calls") or [])
            step_trace: dict[str, Any] = {
                "step": step,
                "assistant_content": raw_output,
                "finish_reason": response.get("finish_reason"),
                "tool_calls": compact_tool_calls_for_trace(tool_calls),
                "tool_results": [],
            }

            final_tool = first_tool_call(tool_calls, "final_answer")
            if final_tool is not None:
                parsed, parse_error = parse_tool_call_arguments(final_tool)
                if parse_error:
                    parsed = {"error": parse_error}
                final = verdict_from_parsed(
                    parsed,
                    raw_output=raw_output,
                    resources=resources_viewed + resources_run,
                    trace=trace,
                )
                step_trace["final"] = parsed
                trace["steps"].append(step_trace)
                if final.valid:
                    break
                if invalid_retries < max_invalid_retries and step < max_steps:
                    invalid_retries += 1
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "The previous final_answer call was malformed or had an invalid score. "
                                "Continue the same judgment and call final_answer with score in [0, 1]."
                            ),
                        }
                    )
                    final = None
                    continue
                break

            non_final_tool_calls = [call for call in tool_calls if tool_call_name(call) != "final_answer"]
            if non_final_tool_calls:
                messages.append({"role": "assistant", "content": raw_output or "", "tool_calls": non_final_tool_calls})
                for tool_call in non_final_tool_calls:
                    result = self.router.execute_tool_call(
                        tool_call,
                        sample=sample,
                        skill_state=skill_state,
                        resources_viewed=resources_viewed,
                        resources_run=resources_run,
                        step=step,
                    )
                    step_trace["tool_results"].append(compact_tool_result_for_trace(result))
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": str(tool_call.get("id") or f"call_{step}_{len(step_trace['tool_results'])}"),
                            "name": tool_call_name(tool_call),
                            "content": json.dumps(result, ensure_ascii=False),
                        }
                    )
                trace["steps"].append(step_trace)
                continue

            parsed = first_json_object(raw_output)
            final = verdict_from_parsed(
                parsed,
                raw_output=raw_output,
                resources=resources_viewed + resources_run,
                trace=trace,
            )
            step_trace["final"] = parsed or {"source": "content", "parse_error": final.error}
            trace["steps"].append(step_trace)
            if final.valid:
                break
            if invalid_retries < max_invalid_retries and step < max_steps:
                invalid_retries += 1
                messages.append(
                    {
                        "role": "user",
                        "content": "Return exactly one JSON object or call final_answer with a valid score in [0, 1].",
                    }
                )
                final = None
                continue
            break

        if final is None or not final.valid:
            forced = self._forced_finalization(messages, resources=resources_viewed + resources_run, trace=trace)
            if forced.valid:
                final = forced
            else:
                return fallback_verdict(
                    sample=sample,
                    reason="invalid final score",
                    error=forced.error or "could not parse pointwise score",
                    raw_output=raw_output or forced.raw_output,
                    trace=trace,
                    resources=resources_viewed + resources_run,
                )

        trace["final"] = {
            "score": final.score,
            "valid": final.valid,
            "used_resources": sorted(set(final.used_resources + resources_viewed + resources_run)),
            "satisfied_count": final.satisfied_count,
            "total_count": final.total_count,
        }
        return replace(
            final,
            used_resources=sorted(set(final.used_resources + resources_viewed + resources_run)),
            trace=trace,
        )

    def _forced_finalization(self, messages: list[dict[str, Any]], *, resources: list[str], trace: dict[str, Any]) -> ScoreVerdict:
        forced_messages = list(messages)
        forced_messages.append(
            {
                "role": "user",
                "content": (
                    "No more tool use. Based on the evidence already visible, call final_answer with a valid score in [0, 1], "
                    "satisfied_count, total_count, confidence, used_resources, and a short reason."
                ),
            }
        )
        try:
            response = self.backend.complete_response(
                forced_messages,
                max_tokens=int(self.variant_config.get("forced_finalization_max_tokens", 512)),
                tools=self.router.tools(skill_loaded=True),
                tool_choice={"type": "function", "function": {"name": "final_answer"}},
            )
        except Exception as exc:  # noqa: BLE001
            return ScoreVerdict(score=0.5, valid=False, reason="forced finalization backend error", used_resources=resources, error=str(exc))
        raw_output = str(response.get("content") or "")
        tool_calls = list(response.get("tool_calls") or [])
        final_tool = first_tool_call(tool_calls, "final_answer")
        if final_tool is not None:
            parsed, parse_error = parse_tool_call_arguments(final_tool)
            if parse_error:
                parsed = {"error": parse_error}
        else:
            parsed = first_json_object(raw_output)
        verdict = verdict_from_parsed(parsed, raw_output=raw_output, resources=resources, trace=trace)
        trace.setdefault("forced_finalization", []).append(
            {
                "raw_output": raw_output,
                "tool_calls": compact_tool_calls_for_trace(tool_calls),
                "parsed": parsed,
                "valid": verdict.valid,
            }
        )
        return verdict


def verdict_from_parsed(
    parsed: dict[str, Any],
    *,
    raw_output: str,
    resources: list[str],
    trace: dict[str, Any],
) -> ScoreVerdict:
    del trace
    if not isinstance(parsed, dict) or not parsed:
        return ScoreVerdict(
            score=0.5,
            valid=False,
            reason="could not parse final JSON",
            used_resources=sorted(set(resources)),
            raw_output=raw_output,
            error="parse_failed",
            invalid_reason="parse_failed",
        )
    count_score = score_from_counts(parsed.get("satisfied_count"), parsed.get("total_count"))
    score_present = parsed.get("score") is not None
    if score_present:
        score = clamp_float(parsed.get("score"))
    elif count_score is not None:
        score = count_score
    else:
        return ScoreVerdict(
            score=0.5,
            valid=False,
            reason=str(parsed.get("reason") or "missing score"),
            used_resources=sorted(set(resources)),
            raw_output=raw_output,
            error=str(parsed.get("error") or "missing_score"),
            invalid_reason="missing_score",
        )

    satisfied = optional_int(parsed.get("satisfied_count"))
    total = optional_int(parsed.get("total_count"))
    confidence = clamp_float(parsed.get("confidence"), default=0.5)
    used = list(parsed.get("used_resources") or [])
    reason = str(parsed.get("reason") or "pointwise instruction-following score")
    return ScoreVerdict(
        score=score,
        valid=True,
        reason=reason,
        used_resources=sorted(set(str(item) for item in used) | set(resources)),
        confidence=confidence,
        satisfied_count=satisfied,
        total_count=total,
        raw_output=raw_output,
    )


def fallback_verdict(
    *,
    sample: PointwiseSample,
    reason: str,
    error: str,
    trace: dict[str, Any],
    resources: list[str],
    raw_output: str = "",
) -> ScoreVerdict:
    trace["fallback"] = {"reason": reason, "error": error, "sample_id": sample.sample_id}
    return ScoreVerdict(
        score=0.5,
        valid=False,
        reason=reason,
        used_resources=sorted(set(resources)),
        confidence=0.0,
        raw_output=raw_output,
        error=error,
        invalid_reason=reason.replace(" ", "_"),
        trace=trace,
    )

