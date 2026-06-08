from __future__ import annotations

from pathlib import Path

from experiments.if_rewardbench.skill_tools import SkillLoader
from skillrm.runtime.skill_package import load_skill_package, skill_package_name


def test_if_skill_loader_finds_instruction_following() -> None:
    loader = SkillLoader([Path("skills")], allowed_skill_names=["instruction_following"])
    skills = loader.scan_skills()
    assert [skill.name for skill in skills] == ["instruction_following"]
    loaded = loader.load_skill("instruction_following")
    assert loaded is not None
    assert "instruction" in loaded.metadata.name


def test_reward_skill_package_loads_fair_skill() -> None:
    package = load_skill_package(
        {
            "skill_path": "skills/reward_judge_fair",
            "skill_loading_mode": "progressive",
            "benchmark": "rewardbench2",
        }
    )
    assert skill_package_name(package) == "reward_judge_fair"
    assert "SKILL.md" in package["files"]
    assert package["manifest"]

