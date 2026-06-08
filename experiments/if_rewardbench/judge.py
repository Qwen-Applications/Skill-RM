from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .backend import ChatBackend
from .io_utils import read_json
from .skill_tools import SkillLoader, ToolExecutor, tool_call_name, tool_schemas


class IFRewardBenchJudge:
    def __init__(
        self,
        *,
        backend: ChatBackend,
        config_dir: str | Path,
        skill_loader: SkillLoader,
        max_agent_steps: int = 8,
        tool_timeout: float = 10.0,
        overall_finalizer: bool = True,
    ) -> None:
        self.backend = backend
        self.config_dir = Path(config_dir)
        self.settings = read_json(self.config_dir / "setting.json")
        self.system_prompt_template = (self.config_dir / "system_prompt.txt").read_text(encoding="utf-8")
        self.user_template = (self.config_dir / "user_template.txt").read_text(encoding="utf-8")
        self.skill_loader = skill_loader
        self.tool_executor = ToolExecutor(skill_loader=skill_loader, timeout=tool_timeout)
        self.max_agent_steps = max_agent_steps
        self.overall_finalizer = overall_finalizer

    def evaluate(self, sample: dict[str, Any], *, mode: str) -> dict[str, Any]:
        user_content = self.render_user_prompt(sample, mode=mode)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.render_system_prompt()},
            {"role": "user", "content": user_content},
        ]
        trace: list[dict[str, Any]] = []
        raw_output = ""
        enable_tools = bool(self.settings.get("enable_tools", True))
        tools = tool_schemas(enable_tools=enable_tools)
        max_tokens = int(self.settings.get("max_tokens", 8192))

        for step in range(1, self.max_agent_steps + 1):
            response = self.backend.complete_response(
                messages,
                max_tokens=max_tokens,
                tools=tools if enable_tools else None,
                tool_choice="auto" if enable_tools else None,
            )
            raw_output = str(response.get("content") or "")
            tool_calls = list(response.get("tool_calls") or [])
            step_trace: dict[str, Any] = {
                "step": step,
                "assistant_content": raw_output,
                "finish_reason": response.get("finish_reason"),
                "tool_calls": compact_tool_calls(tool_calls),
                "tool_results": [],
            }
            if not tool_calls:
                trace.append(step_trace)
                cleaned_output = strip_think(raw_output)
                if mode == "overall" and self.overall_finalizer and not extract_overall_prediction(cleaned_output):
                    return self.finalize_overall_choice(
                        sample,
                        user_content=user_content,
                        raw_output=cleaned_output,
                        raw_output_raw=raw_output,
                        trace=trace,
                        reason="missing_overall_label",
                    )
                return self.build_row(sample, mode=mode, raw_output=cleaned_output, raw_output_raw=raw_output, trace=trace)

            messages.append({"role": "assistant", "content": raw_output, "tool_calls": tool_calls})
            for call in tool_calls:
                result = self.tool_executor.execute(call, sample)
                step_trace["tool_results"].append(compact_tool_result(result))
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(call.get("id") or f"call_{step}_{len(step_trace['tool_results'])}"),
                        "name": tool_call_name(call),
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
            trace.append(step_trace)

        if mode == "overall":
            return self.finalize_overall_choice(
                sample,
                user_content=user_content,
                raw_output=strip_think(raw_output),
                raw_output_raw=raw_output,
                trace=trace,
                reason="max_agent_steps_exceeded",
            )
        return {
            **sample,
            "status": "failed",
            "error": "max_agent_steps_exceeded",
            "raw_output": strip_think(raw_output),
            "raw_output_raw": raw_output,
            "reasoning_trace": trace,
        }

    def render_system_prompt(self) -> str:
        enable_tools = bool(self.settings.get("enable_tools", True))
        if not enable_tools:
            return self.system_prompt_template.replace("{{SKILLS_SECTION}}", "").strip()
        skills_section = self.skill_loader.build_system_prompt_section()
        if "{{SKILLS_SECTION}}" in self.system_prompt_template:
            return self.system_prompt_template.replace("{{SKILLS_SECTION}}", skills_section)
        return f"{self.system_prompt_template}\n\n{skills_section}"

    def render_user_prompt(self, sample: dict[str, Any], *, mode: str) -> str:
        mode = mode.strip().lower()
        base = {
            "system_prompt": str(sample.get("system_prompt") or ""),
            "history": str(sample.get("history") or ""),
            "prompt": str(sample.get("prompt") or ""),
            "response": str(sample.get("response_a") or ""),
            "response_a": str(sample.get("response_a") or ""),
            "response_b": str(sample.get("response_b") or ""),
            "checklist": str(sample.get("checklist") or ""),
        }
        if mode == "overall":
            return self.user_template.format(**base)
        if mode == "constraint":
            return self.user_template.format(**base)
        raise ValueError(f"unsupported mode: {mode}")

    def build_row(
        self,
        sample: dict[str, Any],
        *,
        mode: str,
        raw_output: str,
        raw_output_raw: str,
        trace: list[dict[str, Any]],
    ) -> dict[str, Any]:
        row = dict(sample)
        row.update(
            {
                "status": "success",
                "raw_output": raw_output,
                "raw_output_raw": raw_output_raw,
                "reasoning_trace": trace,
            }
        )
        if mode == "overall":
            prediction = extract_overall_prediction(raw_output)
            row.update({"prediction": prediction, "valid": bool(prediction)})
        elif mode == "constraint":
            parsed_count = len(parse_constraint_blocks(raw_output))
            row.update({"constraint_block_count": parsed_count, "valid": parsed_count > 0})
        return row

    def finalize_overall_choice(
        self,
        sample: dict[str, Any],
        *,
        user_content: str,
        raw_output: str,
        raw_output_raw: str,
        trace: list[dict[str, Any]],
        reason: str,
    ) -> dict[str, Any]:
        final_messages = [
            {
                "role": "system",
                "content": (
                    "You are an IF-RewardBench final-label formatter. "
                    "Choose which assistant response better follows the user instruction. "
                    "Reply with exactly one label and no other text: [[A]] or [[B]]."
                ),
            },
            {
                "role": "user",
                "content": build_overall_finalizer_prompt(
                    user_content=user_content,
                    raw_output=raw_output,
                    trace=trace,
                    reason=reason,
                ),
            },
        ]
        try:
            response = self.backend.complete_response(final_messages, max_tokens=32, tools=None, tool_choice=None)
        except Exception as exc:  # noqa: BLE001
            return {
                **sample,
                "status": "failed",
                "error": f"{reason}; overall_finalizer_failed: {exc}",
                "raw_output": raw_output,
                "raw_output_raw": raw_output_raw,
                "reasoning_trace": trace,
                "overall_finalized": False,
                "overall_finalizer_reason": reason,
            }

        final_raw = str(response.get("content") or "")
        final_clean = strip_think(final_raw)
        final_trace = {
            "step": len(trace) + 1,
            "assistant_content": final_raw,
            "finish_reason": response.get("finish_reason"),
            "tool_calls": [],
            "tool_results": [],
            "finalizer_reason": reason,
        }
        final_trace_list = [*trace, final_trace]
        if extract_overall_prediction(final_clean):
            row = self.build_row(
                sample,
                mode="overall",
                raw_output=final_clean,
                raw_output_raw=final_raw,
                trace=final_trace_list,
            )
            row.update(
                {
                    "overall_finalized": True,
                    "overall_finalizer_reason": reason,
                    "raw_output_before_finalizer": truncate_text(raw_output, 4000),
                }
            )
            return row

        return {
            **sample,
            "status": "failed",
            "error": f"{reason}; overall_finalizer_unparseable",
            "raw_output": final_clean,
            "raw_output_raw": final_raw,
            "reasoning_trace": final_trace_list,
            "valid": False,
            "prediction": None,
            "overall_finalized": False,
            "overall_finalizer_reason": reason,
            "raw_output_before_finalizer": truncate_text(raw_output, 4000),
        }


def compact_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for call in tool_calls:
        function = call.get("function") or {}
        compact.append(
            {
                "id": call.get("id"),
                "name": function.get("name") if isinstance(function, dict) else call.get("name"),
                "arguments": function.get("arguments") if isinstance(function, dict) else call.get("arguments"),
            }
        )
    return compact


def compact_tool_result(result: dict[str, Any]) -> dict[str, Any]:
    output = dict(result)
    for key in ("instructions", "stdout", "stderr"):
        if key in output and isinstance(output[key], str) and len(output[key]) > 1200:
            output[key] = output[key][:1200] + "\n[truncated]"
    return output


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>\s*", "", text or "", flags=re.S).strip()


def build_overall_finalizer_prompt(*, user_content: str, raw_output: str, trace: list[dict[str, Any]], reason: str) -> str:
    trace_context = render_trace_context(trace)
    return (
        "The previous attempt did not produce a parseable final IF-RewardBench overall label.\n"
        f"Reason: {reason}\n\n"
        "Original evaluation prompt:\n"
        "[BEGIN ORIGINAL PROMPT]\n"
        f"{truncate_text(user_content, 50000)}\n"
        "[END ORIGINAL PROMPT]\n\n"
        "Previous assistant output or analysis:\n"
        "[BEGIN PREVIOUS OUTPUT]\n"
        f"{truncate_text(raw_output, 12000)}\n"
        "[END PREVIOUS OUTPUT]\n\n"
        "Recent tool/agent trace context:\n"
        "[BEGIN TRACE]\n"
        f"{trace_context}\n"
        "[END TRACE]\n\n"
        "Now make the final choice. Output exactly one label: [[A]] or [[B]]."
    )


def render_trace_context(trace: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for step in trace[-4:]:
        lines.append(f"step={step.get('step')} finish_reason={step.get('finish_reason')}")
        assistant_content = str(step.get("assistant_content") or "")
        if assistant_content:
            lines.append("assistant=" + truncate_text(strip_think(assistant_content), 2000))
        for result in step.get("tool_results") or []:
            lines.append("tool_result=" + truncate_text(json.dumps(result, ensure_ascii=False), 1200))
    return "\n".join(lines)


def truncate_text(text: str, max_chars: int) -> str:
    text = text or ""
    if len(text) <= max_chars:
        return text
    keep = max(0, (max_chars - 40) // 2)
    return f"{text[:keep]}\n[...truncated...]\n{text[-keep:]}"


def extract_overall_prediction(raw_output: str) -> str | None:
    text = strip_think(raw_output)
    has_a = "[[A]]" in text
    has_b = "[[B]]" in text
    if has_a and not has_b:
        return "A"
    if has_b and not has_a:
        return "B"
    match = re.search(r"\[\[\s*([AB])\s*\]\]", text, flags=re.I)
    if match:
        return match.group(1).upper()
    return None


def parse_constraint_blocks(raw_output: str) -> list[dict[str, str]]:
    text = strip_think(raw_output)
    pattern = re.compile(
        r"\[检查项(?P<idx>\d+)-开始\]\s*"
        r"\n要求：(.*?)\s*"
        r"\n分析：(.*?)\s*"
        r"\n结论：(.*?)\s*"
        r"\n\[检查项(?P=idx)-结束\]",
        flags=re.S,
    )
    rows: list[dict[str, str]] = []
    for match in pattern.finditer(text):
        rows.append(
            {
                "index": match.group("idx"),
                "requirement": match.group(2).strip(),
                "analysis": match.group(3).strip(),
                "conclusion": match.group(4).strip(),
            }
        )
    return rows
