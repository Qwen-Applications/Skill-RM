from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_install_verl_recipe_copies_into_compatible_checkout(tmp_path: Path) -> None:
    verl_root = tmp_path / "verl"
    (verl_root / "verl" / "version").mkdir(parents=True)
    (verl_root / "verl" / "version" / "version").write_text("0.8.0.dev\n", encoding="utf-8")
    (verl_root / "recipe").mkdir()

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "install_verl_recipe.py"),
            "--verl-root",
            str(verl_root),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    target = verl_root / "recipe" / "skill_rm_if_pointwise"
    assert (target / "reward_fn_pointwise.py").exists()
    assert (target / "run_grpo.sh").exists()


def test_install_verl_recipe_rejects_version_mismatch(tmp_path: Path) -> None:
    verl_root = tmp_path / "verl"
    (verl_root / "verl" / "version").mkdir(parents=True)
    (verl_root / "verl" / "version" / "version").write_text("0.7.0\n", encoding="utf-8")
    (verl_root / "recipe").mkdir()

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "install_verl_recipe.py"),
            "--verl-root",
            str(verl_root),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "version mismatch" in result.stderr


def test_prepare_verinstruct_pointwise_from_jsonl(tmp_path: Path) -> None:
    pytest.importorskip("datasets")
    source = tmp_path / "toy.jsonl"
    rows = [
        {
            "id": "sample_1",
            "prompt": "List exactly three fruits.",
            "checkers": [{"id": "c1", "text": "The answer lists exactly three fruits."}],
            "functions": [],
        },
        {
            "id": "sample_2",
            "instruction": "Reply with JSON only.",
            "checkers": [],
            "functions": [],
        },
        {
            "id": "sample_3",
            "query": "Answer in French.",
            "checkers": [],
            "functions": [],
        },
    ]
    source.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    output_dir = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "prepare_verinstruct_pointwise.py"),
            "--input-jsonl",
            str(source),
            "--output-dir",
            str(output_dir),
            "--validation-size",
            "0.34",
            "--seed",
            "7",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (output_dir / "train.parquet").exists()
    assert (output_dir / "validation.parquet").exists()
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["reference_response_generated"] is False
    assert metadata["train_rows"] + metadata["validation_rows"] == 3


def test_verl_recipe_mock_reward_smoke() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "integrations" / "verl")
    env["SKILL_RM_BACKEND"] = "mock"
    code = """
import asyncio
from recipe.skill_rm_if_pointwise.reward_fn_pointwise import compute_score

async def main():
    result = await compute_score(
        data_source=None,
        solution_str="1. Apples\\n2. Bananas\\n3. Cherries",
        ground_truth=None,
        extra_info={"query": "List exactly three fruits using a numbered list.", "sample_id": "smoke"},
    )
    assert 0.0 <= float(result["score"]) <= 1.0
    assert result["variant"] == "skill_mounted_verifier_plus_code"

asyncio.run(main())
"""
    result = subprocess.run([sys.executable, "-c", code], check=False, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
