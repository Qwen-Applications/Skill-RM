from __future__ import annotations

import json
import re
from typing import Any

from ..benchmarks.pairwise.data import OpenRSPairTask
from ..benchmarks.pairwise.prompts import format_judgebench_prompt
from ..runners.rewardbench2 import (
    OFFICIAL_RANKING_SYSTEM_PROMPT,
    OFFICIAL_RANKING_USER_TEMPLATE,
)
from ..benchmarks.rewardbench2.parsing import parse_official_winner
from ..runtime.resources import operational_sample_resources


FINAL_LISTWISE_TOOL = {
    "type": "function",
    "function": {
        "name": "final_answer",
        "description": "Submit the final listwise judgment. The verdict must be exactly one of A, B, C, or D.",
        "parameters": {
            "type": "object",
            "properties": {
                "verdict": {"type": "string", "enum": ["A", "B", "C", "D"]},
                "rationale": {"type": "string"},
            },
            "required": ["verdict"],
            "additionalProperties": False,
        },
    },
}


PYTHON_SANDBOX_TOOL = {
    "type": "function",
    "function": {
        "name": "python_sandbox",
        "description": (
            "Run small Python checks over only the visible prompt and candidate responses. "
            "Use for deterministic visible-text evidence such as word counts, exact format, regex, "
            "JSON/Markdown structure, simple arithmetic, or required terms. No external data is available."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "Python code. Available variables: prompt (str), candidates (dict label->text), "
                        "sample (dict with prompt and candidates). Print compact JSON/text evidence."
                    ),
                },
                "reason": {"type": "string"},
            },
            "required": ["code", "reason"],
            "additionalProperties": False,
        },
    },
}


def flat_listwise_prompt(record: dict[str, Any], formatted: dict[str, Any], skill_package: dict[str, Any], config: dict[str, Any], *, benchmark: str) -> str:
    return "\n\n".join(
        [
            "You are an impartial reward judge. This is a flat-prompt ablation: there is no skill tool and no progressive disclosure.",
            "Use the visible task, candidates, and the flattened resource text below. Do not assume any hidden labels.",
            flat_resource_block(skill_package, benchmark, record, {"responses": formatted["responses"]}, config),
            "## Judge Task",
            OFFICIAL_RANKING_SYSTEM_PROMPT,
            OFFICIAL_RANKING_USER_TEMPLATE.format(
                question=str(record.get("prompt", "")),
                answer_a=formatted["responses"]["A"],
                answer_b=formatted["responses"]["B"],
                answer_c=formatted["responses"]["C"],
                answer_d=formatted["responses"]["D"],
            ),
            'End with exactly one final verdict token: "[[A]]", "[[B]]", "[[C]]", or "[[D]]".',
        ]
    )


def flat_pairwise_prompt(task: OpenRSPairTask, skill_package: dict[str, Any], config: dict[str, Any]) -> str:
    record = {
        "id": task.task_id,
        "prompt": task.prompt,
        "benchmark": task.benchmark,
        "query_type": task.query_type,
        "domain": task.domain,
        "pair": task.pair,
        "order": task.order,
        **dict(task.sample_resources or {}),
    }
    formatted = {"responses": task.responses}
    if task.benchmark.startswith("judgebench"):
        judge_task = format_judgebench_prompt(task)
        final = 'End with exactly one label and no explanation: Output (a) or Output (b).'
    else:
        judge_task = f"[User Prompt]\n{task.prompt}\n\n[Response A]\n{task.responses['A']}\n\n[Response B]\n{task.responses['B']}"
        final = "End with exactly one label: Final: A, Final: B, or Final: Tie."
    return "\n\n".join(
        [
            "You are an impartial reward judge. This is a flat-prompt ablation: there is no skill tool and no progressive disclosure.",
            "Use the visible task, candidates, and the flattened resource text below. Do not assume any hidden labels.",
            flat_resource_block(skill_package, str(config.get("benchmark") or ""), record, formatted, config),
            "## Judge Task",
            judge_task,
            final,
        ]
    )


def flat_resource_block(
    skill_package: dict[str, Any],
    benchmark: str,
    record: dict[str, Any],
    formatted: dict[str, Any],
    config: dict[str, Any],
) -> str:
    paths = flat_resource_paths(skill_package, benchmark)
    max_chars = int(config.get("flat_prompt_max_resource_chars", 18000))
    chunks = []
    budget = max_chars
    for path in paths:
        text = str(skill_package.get("files", {}).get(path, "")).strip()
        if not text or budget <= 0:
            continue
        clipped = text[:budget]
        budget -= len(clipped)
        chunks.append(f"### {path}\n{clipped}")
    if str(config.get("skill_allowed_setting") or "") == "skill_operational":
        runtime_index, runtime_files = operational_sample_resources(record, formatted, config)
        if runtime_files:
            chunks.append("## Sample-Visible Operational Resources")
            for path, text in runtime_files.items():
                chunks.append(f"### {path}\n{text[: int(config.get('flat_prompt_max_sample_resource_chars', 6000))]}")
        if runtime_index:
            chunks.append("### sample/resource_index.json\n" + json.dumps(runtime_index, ensure_ascii=False, indent=2))
    return "## Flattened Resources\n" + "\n\n".join(chunks)


def flat_resource_paths(skill_package: dict[str, Any], benchmark: str) -> list[str]:
    normalized = benchmark.lower().replace("-", "").replace("_", "")
    bench_paths: list[str] = []
    if "rewardbench2" in normalized or "rewardbenchv2" in normalized or normalized == "rb2":
        bench_paths = ["references/rewardbench2_principles.md", "rubrics/rewardbench2.md"]
    elif "judgebench" in normalized:
        bench_paths = ["references/judgebench_principles.md", "rubrics/judgebench.md"]
    elif "rmbench" in normalized:
        bench_paths = ["references/rmbench_principles.md", "rubrics/rmbench.md"]
    preferred = [
        "SKILL.md",
        "resources.yaml",
        "references/generic_principles.md",
        "references/bias_control.md",
        "references/output_format.md",
        "references/operational_aggregation.md",
        "rubrics/generic_pairwise.md",
        *bench_paths,
    ]
    files = skill_package.get("files", {})
    return [path for path in preferred if path in files]


def parse_listwise_final(content: str) -> dict[str, str]:
    label = parse_listwise_label(content)
    return {
        "winner": label,
        "verdict": label,
        "source": "listwise_label" if label in {"A", "B", "C", "D"} else "unparsed",
    }


def parse_listwise_label(content: str) -> str:
    official = parse_official_winner(content)
    if official in {"A", "B", "C", "D"}:
        return official
    tail = content.strip()[-300:]
    patterns = [
        r"(?im)^\s*Final\s*:\s*([ABCD])\s*\.?\s*$",
        r"(?im)^\s*(?:Answer|Verdict|Choice|Best)\s*:\s*([ABCD])\s*\.?\s*$",
        r"(?im)^\s*\[?([ABCD])\]?\s*\.?\s*$",
        r"(?i)\b(?:assistant|response|candidate)\s*([ABCD])\b",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, tail)
        if matches:
            return str(matches[-1]).upper()
    json_match = re.search(r"\{.*\}", tail, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            parsed = {}
        if isinstance(parsed, dict):
            for key in ("verdict", "winner", "best_label", "choice", "selected"):
                value = parsed.get(key)
                if isinstance(value, str) and value.strip().upper() in {"A", "B", "C", "D"}:
                    return value.strip().upper()
    return "error"


def tool_only_listwise_system_prompt() -> str:
    return (
        "You are an impartial reward judge. You may use python_sandbox only for deterministic checks over the visible "
        "prompt and candidates. No skill, metadata, reference, ground truth, checklist, or external resource is available. "
        "When ready, call final_answer with A, B, C, or D."
    )


def tool_only_pairwise_system_prompt() -> str:
    return (
        "You are an impartial pairwise reward judge. You may use python_sandbox only for deterministic checks over the visible "
        "prompt and candidates. No skill, metadata, reference, ground truth, checklist, or external resource is available. "
        "When ready, give the required pairwise final label."
    )


def tool_only_judgebench_system_prompt() -> str:
    return (
        "You are an impartial forced-choice response judge. You may use python_sandbox only for deterministic checks over the "
        "visible instruction and Output (a)/Output (b). No skill, metadata, reference, ground truth, checklist, or external "
        "resource is available. A maps to Output (a), and B maps to Output (b). When ready, call final_answer with A or B."
    )
