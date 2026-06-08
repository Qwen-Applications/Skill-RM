#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


REQUIRED = {
    "SKILLRM_DATA_ROOT": [
        "rewardbench_v2/rewardbench_v2.jsonl",
        "judgebench/gpt.jsonl",
        "judgebench/claude.jsonl",
        "rmbench/rmbench.jsonl",
    ],
    "JETTS_DATA_DIR": [
        "gsm8k_Qwen2.5-72B-Instruct.jsonl",
        "bigcodebench_Qwen2.5-72B-Instruct.jsonl",
        "ifeval_Qwen2.5-72B-Instruct.jsonl",
        "humaneval_Qwen2.5-72B-Instruct.jsonl",
    ],
    "SKILLRM_IF_REWARDBENCH_ROOT": [
        "data/if_rewardbench.json",
        "inference/position_maps.json",
        "metrics/analysis_constraint_assessment.py",
        "metrics/analysis_overall_assessment.py",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate external Skill-RM benchmark data paths.")
    parser.add_argument("--skip-openrs", action="store_true", help="Skip SKILLRM_DATA_ROOT checks for RewardBench2/JudgeBench/RM-Bench.")
    parser.add_argument("--skip-jetts", action="store_true", help="Skip JETTS_DATA_DIR checks.")
    parser.add_argument("--skip-if-rewardbench", action="store_true", help="Skip SKILLRM_IF_REWARDBENCH_ROOT checks.")
    return parser.parse_args()


def selected_requirements(args: argparse.Namespace) -> dict[str, list[str]]:
    required = dict(REQUIRED)
    if args.skip_openrs:
        required.pop("SKILLRM_DATA_ROOT", None)
    if args.skip_jetts:
        required.pop("JETTS_DATA_DIR", None)
    if args.skip_if_rewardbench:
        required.pop("SKILLRM_IF_REWARDBENCH_ROOT", None)
    return required


def main() -> int:
    args = parse_args()
    missing: list[str] = []
    required = selected_requirements(args)
    if not required:
        print("No data path checks selected.")
        return 0
    for env_name, rel_paths in required.items():
        root = os.environ.get(env_name)
        if not root:
            missing.append(f"{env_name} is not set")
            continue
        for rel_path in rel_paths:
            path = Path(root) / rel_path
            if not path.is_file() or path.stat().st_size == 0:
                missing.append(str(path))
    if missing:
        print("Missing data paths:")
        for item in missing:
            print(f"- {item}")
        return 1
    print("All configured data paths exist.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
