"""Shared dataclasses for the pointwise Skill-RM reward package."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PointwiseSample:
    sample_id: str
    prompt: str
    response: str
    system_prompt: str = ""
    history: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    sample_resources: Any = None


@dataclass(frozen=True)
class ScoreVerdict:
    score: float
    valid: bool
    reason: str
    used_resources: list[str] = field(default_factory=list)
    confidence: float = 0.5
    satisfied_count: int | None = None
    total_count: int | None = None
    raw_output: str = ""
    error: str | None = None
    invalid_reason: str = ""
    trace: dict[str, Any] = field(default_factory=dict)

