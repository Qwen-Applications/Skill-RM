from __future__ import annotations

import subprocess
import sys


def run_help(args: list[str]) -> None:
    result = subprocess.run(args, check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


def test_public_cli_help() -> None:
    run_help([sys.executable, "-m", "skillrm.runners.rewardbench2", "--help"])
    run_help([sys.executable, "-m", "skillrm.runners.pairwise", "--help"])
    run_help([sys.executable, "-m", "experiments.if_rewardbench.runner", "--help"])
    run_help([sys.executable, "experiments/jetts_seqko/run_seqko.py", "--help"])
    run_help([sys.executable, "scripts/check_data_paths.py", "--help"])
    run_help([sys.executable, "scripts/install_verl_recipe.py", "--help"])
    run_help([sys.executable, "scripts/prepare_verinstruct_pointwise.py", "--help"])
