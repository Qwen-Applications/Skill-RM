from __future__ import annotations

from pathlib import Path

from experiments.if_rewardbench.backend import MockBackend, OpenAICompatibleBackend, build_backend
from experiments.if_rewardbench.judge import IFRewardBenchJudge
from experiments.if_rewardbench.skill_tools import SkillLoader


def build_judge(bundle: str) -> IFRewardBenchJudge:
    return IFRewardBenchJudge(
        backend=MockBackend(),
        config_dir=Path("configs/if_rewardbench") / bundle,
        skill_loader=SkillLoader([Path("skills")], allowed_skill_names=["instruction_following"]),
    )


def test_if_mock_overall_agentic_returns_label() -> None:
    judge = build_judge("if_rb_overall_agentic")
    row = judge.evaluate(
        {
            "id": "ifrb_1_0_1",
            "original_id": "1",
            "pair_indices": [0, 1],
            "prompt": "Choose the better response.",
            "system_prompt": "",
            "history": "",
            "response_a": "A follows the instruction.",
            "response_b": "B ignores it.",
        },
        mode="overall",
    )
    assert row["status"] == "success"
    assert row["prediction"] in {"A", "B"}


def test_if_mock_constraint_baseline_emits_blocks() -> None:
    judge = build_judge("if_rb_constraint_baseline")
    row = judge.evaluate(
        {
            "id": "ifrb_1_0",
            "original_id": "1",
            "response_index": 0,
            "prompt": "Answer in one sentence.",
            "system_prompt": "",
            "history": "",
            "response_a": "One sentence.",
            "checklist": "[检查项1-开始]\nAnswer in one sentence.\n[检查项1-结束]",
        },
        mode="constraint",
    )
    assert row["status"] == "success"
    assert row["constraint_block_count"] == 1


def test_if_backend_defaults_disable_thinking() -> None:
    backend = build_backend(
        {
            "backend": "vllm",
            "base_urls": ["http://localhost:8000/v1"],
            "model": "Qwen3.5-27B",
        }
    )
    assert isinstance(backend, OpenAICompatibleBackend)
    assert backend.send_thinking_field is True
    assert backend.enable_thinking is False

