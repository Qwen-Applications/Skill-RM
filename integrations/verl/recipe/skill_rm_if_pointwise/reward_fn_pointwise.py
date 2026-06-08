"""Pointwise custom reward function for verl Reward Loop."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
VERL_ROOT = THIS_DIR.parents[1]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(VERL_ROOT) not in sys.path:
    sys.path.insert(0, str(VERL_ROOT))

try:
    from recipe.skill_rm_if_pointwise.judge import PointwiseSkillRMJudge
    from recipe.skill_rm_if_pointwise.llm import build_backend_from_env
    from recipe.skill_rm_if_pointwise.resources import ResourceRouter
    from recipe.skill_rm_if_pointwise.sample_resources import normalize_sample_resources, render_sample_resources_section
    from recipe.skill_rm_if_pointwise.types import PointwiseSample
except ImportError:  # pragma: no cover - direct file loading by verl.
    from .judge import PointwiseSkillRMJudge
    from .llm import build_backend_from_env
    from .resources import ResourceRouter
    from .sample_resources import normalize_sample_resources, render_sample_resources_section
    from .types import PointwiseSample


LOGGER = logging.getLogger("skill_rm_if_pointwise.reward")
DEFAULT_FALLBACK_SCORE = 0.5
DEFAULT_VARIANT = "skill_mounted_verifier_plus_code"

VARIANT_SPECS: dict[str, dict[str, Any]] = {
    DEFAULT_VARIANT: {
        "resource_delivery_mode": "mounted_plus_code",
        "mount_sample_resources": True,
        "include_static_resources": True,
        "enable_python_sandbox": True,
        "max_agent_steps": 7,
        "max_resources_per_sample": 5,
    },
}

VARIANT_ALIASES = {
    "mounted_verifier_plus_code": DEFAULT_VARIANT,
    "skill_code": DEFAULT_VARIANT,
}

_JUDGE_CACHE: dict[str, PointwiseSkillRMJudge] = {}


def normalize_variant(variant: str) -> str:
    normalized = VARIANT_ALIASES.get(str(variant), str(variant))
    if normalized not in VARIANT_SPECS:
        supported = ", ".join(sorted(VARIANT_SPECS))
        raise ValueError(f"Unsupported pointwise Skill-RM variant '{variant}'. Supported variants: {supported}")
    return normalized


def _variant_config(variant: str) -> dict[str, Any]:
    config = dict(VARIANT_SPECS[variant])
    config.update(
        {
            "variant": variant,
            "max_tokens": int(os.getenv("MAX_TOKENS", "4096")),
            "max_agent_steps": int(os.getenv("MAX_AGENT_STEPS", str(config.get("max_agent_steps", 6)))),
            "invalid_final_answer_retries": int(os.getenv("INVALID_FINAL_ANSWER_RETRIES", "1")),
            "forced_finalization_max_tokens": int(os.getenv("FORCED_FINALIZATION_MAX_TOKENS", "512")),
            "max_resources_per_sample": int(os.getenv("MAX_RESOURCES_PER_SAMPLE", str(config.get("max_resources_per_sample", 4)))),
            "max_prompt_chars": int(os.getenv("MAX_PROMPT_CHARS", "8000")),
            "max_response_chars": int(os.getenv("MAX_RESPONSE_CHARS", "12000")),
            "max_resource_chars": int(os.getenv("MAX_RESOURCE_CHARS", "8000")),
            "enable_python_sandbox": _env_flag("ENABLE_PYTHON_SANDBOX", bool(config.get("enable_python_sandbox", False))),
            "python_sandbox_timeout": float(os.getenv("PYTHON_SANDBOX_TIMEOUT", "3.0")),
            "python_sandbox_max_code_chars": int(os.getenv("PYTHON_SANDBOX_MAX_CODE_CHARS", "6000")),
            "python_sandbox_max_output_chars": int(os.getenv("PYTHON_SANDBOX_MAX_OUTPUT_CHARS", "4000")),
            "max_python_sandbox_calls": int(os.getenv("MAX_PYTHON_SANDBOX_CALLS", "3")),
            "tool_choice": os.getenv("TOOL_CHOICE", "auto"),
        }
    )
    return config


def _get_judge(variant: str) -> PointwiseSkillRMJudge:
    variant = normalize_variant(variant)
    if variant not in _JUDGE_CACHE:
        config = _variant_config(variant)
        backend = build_backend_from_env()
        router = ResourceRouter(skill_dir=THIS_DIR / "skills" / "instruction_following_pointwise", variant_config=config)
        _JUDGE_CACHE[variant] = PointwiseSkillRMJudge(backend=backend, router=router, variant_config=config)
        LOGGER.info("Initialized pointwise Skill-RM judge | variant=%s | backend=%s", variant, os.getenv("SKILL_RM_BACKEND", "openai"))
    return _JUDGE_CACHE[variant]


async def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    variant=DEFAULT_VARIANT,
    **kwargs,
):
    del data_source, ground_truth, kwargs
    started = time.perf_counter()
    info = extra_info or {}
    normalized_variant = normalize_variant(variant)
    config = _variant_config(normalized_variant)

    query = _extract_query(info)
    if not query:
        raise ValueError("Pointwise Skill-RM reward requires extra_info['query'] or extra_info['prompt'].")

    sample_resources = None
    prompt_resources_section = ""
    if config["resource_delivery_mode"] != "none":
        sample_resources = normalize_sample_resources(info)
        if config["resource_delivery_mode"] == "prompt":
            prompt_resources_section = render_sample_resources_section(sample_resources)

    sample = PointwiseSample(
        sample_id=str(info.get("sample_id") or info.get("id") or "unknown"),
        prompt=query,
        response=solution_str or "",
        system_prompt=str(info.get("system_prompt") or ""),
        history=str(info.get("history") or ""),
        metadata=dict(info),
        sample_resources=sample_resources,
    )

    try:
        judge = _get_judge(normalized_variant)
        verdict = await _run_in_daemon_thread(judge.score_sample, sample, prompt_resources_section=prompt_resources_section)
        latency = time.perf_counter() - started
        resource_meta = sample_resources.to_metadata() if sample_resources is not None else _empty_resource_metadata(config["resource_delivery_mode"])
        return _make_reward_info_batch_safe({
            "score": verdict.score,
            "invalid": int(not verdict.valid),
            "variant": normalized_variant,
            "resource_delivery_mode": config["resource_delivery_mode"],
            "satisfied_count": verdict.satisfied_count,
            "total_count": verdict.total_count,
            "confidence": verdict.confidence,
            "used_resources": verdict.used_resources,
            "reason": verdict.reason,
            "raw_output": verdict.raw_output if _env_flag("SAVE_RAW_REWARD_OUTPUT", False) else "",
            "latency_sec": latency,
            "error": verdict.error or "",
            "invalid_reason": verdict.invalid_reason,
            **resource_meta,
        })
    except Exception as exc:  # noqa: BLE001 - keep RL alive if the judge has a transient issue.
        latency = time.perf_counter() - started
        LOGGER.warning("Pointwise Skill-RM reward failed -> fallback %.2f | sample_id=%s | variant=%s | error=%s", DEFAULT_FALLBACK_SCORE, sample.sample_id, normalized_variant, exc)
        resource_meta = sample_resources.to_metadata() if sample_resources is not None else _empty_resource_metadata(config["resource_delivery_mode"])
        return _make_reward_info_batch_safe({
            "score": DEFAULT_FALLBACK_SCORE,
            "invalid": 1,
            "variant": normalized_variant,
            "resource_delivery_mode": config["resource_delivery_mode"],
            "satisfied_count": -1,
            "total_count": -1,
            "confidence": 0.0,
            "used_resources": [],
            "reason": "reward_exception_fallback",
            "raw_output": "",
            "latency_sec": latency,
            "error": str(exc),
            "invalid_reason": "reward_exception",
            **resource_meta,
        })


def _extract_query(info: dict[str, Any]) -> str:
    for key in ("query", "prompt", "instruction", "user_prompt"):
        value = info.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _empty_resource_metadata(mode: str) -> dict[str, Any]:
    return {
        "sample_resources_delivery_mode": mode,
        "sample_resources_present": 0,
        "sample_checklist_count": 0,
        "sample_rule_count": 0,
        "sample_llm_count": 0,
        "sample_resource_parse_error": "",
    }


def _make_reward_info_batch_safe(info: dict[str, Any]) -> dict[str, Any]:
    """Keep reward metadata compatible with verl non_tensor_batch assembly."""
    safe: dict[str, Any] = {}
    for key, value in info.items():
        if value is None:
            safe[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            safe[key] = value
        else:
            safe[key] = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return safe


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


async def _run_in_daemon_thread(fn, *args, **kwargs):
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    def runner() -> None:
        try:
            result = fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            loop.call_soon_threadsafe(future.set_exception, exc)
        else:
            loop.call_soon_threadsafe(future.set_result, result)

    thread = threading.Thread(target=runner, name="skill-rm-if-pointwise-judge", daemon=True)
    thread.start()
    return await future
