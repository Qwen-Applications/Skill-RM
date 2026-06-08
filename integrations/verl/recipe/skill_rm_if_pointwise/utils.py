"""Utility helpers for pointwise Skill-RM."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def safe_text(value: Any, limit: int = 8000) -> str:
    text = "" if value is None else str(value)
    if len(text) <= limit:
        return text
    return text[:limit] + "\n[truncated]"


def first_json_object(text: str) -> dict[str, Any]:
    stripped = (text or "").strip()
    if not stripped:
        return {}
    decoder = json.JSONDecoder()
    try:
        value, _ = decoder.raw_decode(stripped)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if fenced:
        try:
            value = json.loads(fenced.group(1))
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}
    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if match:
        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def clamp_float(value: Any, default: float = 0.5, *, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = default
    if score != score:  # NaN
        score = default
    return min(hi, max(lo, score))


def optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9_.-]+", "_", str(value or "").strip().lower()).strip("_")


def score_from_counts(satisfied: Any, total: Any) -> float | None:
    sat = optional_int(satisfied)
    tot = optional_int(total)
    if sat is None or tot is None:
        return None
    if tot < 0 or sat < 0:
        return None
    if tot == 0:
        return 1.0
    return clamp_float(sat / tot)

