from __future__ import annotations

from typing import Any


def format_pair_prompt(prompt: str, response_a: str, response_b: str) -> str:
    return (
        "Task prompt:\n"
        f"{prompt}\n\n"
        "Candidate A:\n"
        f"{response_a}\n\n"
        "Candidate B:\n"
        f"{response_b}\n\n"
        "Choose the better candidate for the original task. This is a knockout match: "
        "you must pick exactly one winner, A or B. Do not answer Tie or Abstain."
    )


BASELINE_SYSTEM_PROMPT = (
    "You are an impartial reward judge in a forced-choice pairwise knockout experiment. "
    "Compare the two candidates for the original task prompt and choose exactly one winner, A or B. "
    "Do not answer Tie or Abstain; if both are flawed, choose the less flawed candidate. "
    "Reply with exactly `Final: A` or `Final: B`, and nothing else."
)


def baseline_system_prompt(config: dict[str, Any]) -> str:
    variant = str(config.get("system_prompt_variant") or "baseline")
    if variant in {"baseline", "skill_matched_plain"}:
        return BASELINE_SYSTEM_PROMPT
    if variant == "default":
        raise ValueError(
            "Unsupported JETTS baseline system_prompt_variant: default. "
            "Use system_prompt_variant: baseline or omit the field."
        )
    raise ValueError(f"Unknown baseline system_prompt_variant: {variant}")


def format_skill_system_prompt(skill_name: str) -> str:
    return (
        "You are an impartial reward judge in a forced-choice pairwise knockout experiment. "
        "You may load and use the provided reward-judge skill before deciding. "
        "Compare the two candidates for the original task prompt and choose exactly one winner, A or B. "
        "Do not answer Tie or Abstain; if both are flawed, choose the less flawed candidate. "
        "Use tools only when they help. When ready, call final_answer with answer A or B.\n\n"
        f"Available skill: {skill_name}"
    )
