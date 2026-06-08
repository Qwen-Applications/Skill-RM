#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECIPE = REPO_ROOT / "integrations" / "verl" / "recipe" / "skill_rm_if_pointwise"
REQUIRED_VERL = REPO_ROOT / "integrations" / "verl" / "REQUIRED_VERL.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install the Skill-RM pointwise IF-RL recipe into an external verl checkout.")
    parser.add_argument("--verl-root", type=Path, help="Existing verl checkout root.")
    parser.add_argument("--target", type=Path, help="Target path used with --clone-url.")
    parser.add_argument("--clone-url", help="Optional verl git URL to clone before installing the recipe.")
    parser.add_argument("--verl-ref", help="Required with --clone-url. Git tag, branch, or commit to checkout.")
    parser.add_argument("--recipe-src", type=Path, default=DEFAULT_RECIPE)
    parser.add_argument("--expected-version", default=read_required_version())
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing target recipe directory.")
    return parser.parse_args()


def read_required_version() -> str:
    if not REQUIRED_VERL.exists():
        return "0.8.0.dev"
    for line in REQUIRED_VERL.read_text(encoding="utf-8").splitlines():
        if line.startswith("version_file:"):
            return line.split(":", 1)[1].strip()
    return "0.8.0.dev"


def main() -> int:
    args = parse_args()
    verl_root = resolve_verl_root(args)
    validate_verl_root(verl_root, args.expected_version)
    install_recipe(args.recipe_src, verl_root, overwrite=args.overwrite)
    print(f"[OK] Installed Skill-RM verl recipe into {verl_root / 'recipe' / 'skill_rm_if_pointwise'}")
    return 0


def resolve_verl_root(args: argparse.Namespace) -> Path:
    if args.clone_url:
        if not args.verl_ref:
            raise SystemExit("--verl-ref is required when using --clone-url.")
        if not args.target:
            raise SystemExit("--target is required when using --clone-url.")
        target = args.target.expanduser().resolve()
        if target.exists() and any(target.iterdir()):
            raise SystemExit(f"Clone target already exists and is not empty: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--recursive", args.clone_url, str(target)], check=True)
        subprocess.run(["git", "-C", str(target), "checkout", args.verl_ref], check=True)
        subprocess.run(["git", "-C", str(target), "submodule", "update", "--init", "--recursive"], check=True)
        return target

    if not args.verl_root:
        raise SystemExit("Pass --verl-root for an existing checkout, or --clone-url with --target and --verl-ref.")
    return args.verl_root.expanduser().resolve()


def validate_verl_root(verl_root: Path, expected_version: str) -> None:
    version_file = verl_root / "verl" / "version" / "version"
    recipe_dir = verl_root / "recipe"
    if not version_file.exists():
        raise SystemExit(f"Not a valid verl checkout: missing {version_file}")
    if not recipe_dir.exists():
        raise SystemExit(f"Not a compatible verl checkout: missing {recipe_dir}")
    actual_version = version_file.read_text(encoding="utf-8").strip()
    if actual_version != expected_version:
        raise SystemExit(
            f"verl version mismatch: expected {expected_version!r}, found {actual_version!r}. "
            "Use a compatible verl checkout or pass --expected-version deliberately."
        )


def install_recipe(recipe_src: Path, verl_root: Path, *, overwrite: bool) -> None:
    recipe_src = recipe_src.expanduser().resolve()
    if not recipe_src.exists():
        raise SystemExit(f"Recipe source does not exist: {recipe_src}")
    target = verl_root / "recipe" / "skill_rm_if_pointwise"
    if target.exists():
        if not overwrite:
            raise SystemExit(f"Target recipe already exists: {target}. Pass --overwrite to replace it.")
        shutil.rmtree(target)
    shutil.copytree(recipe_src, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"))


if __name__ == "__main__":
    sys.exit(main())
