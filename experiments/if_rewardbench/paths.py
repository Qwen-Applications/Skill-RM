from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]
ASSET_ROOT = Path(os.environ.get("SKILLRM_IF_REWARDBENCH_ROOT", PROJECT_ROOT / "data" / "if_rewardbench"))
DEFAULT_DATA_PATH = ASSET_ROOT / "data" / "if_rewardbench.json"
DEFAULT_POSITION_MAP_PATH = ASSET_ROOT / "inference" / "position_maps.json"
DEFAULT_METRICS_DIR = ASSET_ROOT / "metrics"
DEFAULT_SKILLS_DIR = PROJECT_ROOT / "skills"
DEFAULT_CONFIGS_DIR = PROJECT_ROOT / "configs" / "if_rewardbench"
DEFAULT_RUNS_DIR = PROJECT_ROOT / "outputs" / "if_rewardbench"


def default_config_for_mode(mode: str) -> str:
    key = mode.strip().lower()
    if key == "overall":
        return "if_rb_overall_agentic"
    if key == "constraint":
        return "if_rb_constraint_agentic"
    raise ValueError(f"unsupported IF-RewardBench mode: {mode}")
