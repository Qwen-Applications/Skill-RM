from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_MARKDOWN_FILES = sorted(ROOT.rglob("*.md"))


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def markdown_local_targets(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    targets: list[str] = []
    patterns = [
        r"!\[[^\]]*\]\(([^)]+)\)",
        r"<img\s+[^>]*src=[\"']([^\"']+)[\"']",
        r"\[[^\]]+\]\(([^)]+)\)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            target = match.group(1).strip()
            if target.startswith(("http://", "https://", "#", "mailto:", "data:")):
                continue
            if target.startswith("<") and target.endswith(">"):
                target = target[1:-1]
            target = target.split("#", 1)[0]
            if target:
                targets.append(target)
    return targets


def test_readme_framework_asset_is_release_ready() -> None:
    readme = read("README.md")
    assert "assets/skill-rm-framework.jpeg" in readme
    assert (ROOT / "assets" / "skill-rm-framework.jpeg").is_file()

    gitignore = read(".gitignore")
    assert "assets/*" in gitignore
    assert "!assets/skill-rm-framework.jpeg" in gitignore


def test_markdown_local_links_exist() -> None:
    missing: list[str] = []
    for markdown_file in PUBLIC_MARKDOWN_FILES:
        for target in markdown_local_targets(markdown_file):
            if not (markdown_file.parent / target).exists():
                missing.append(f"{markdown_file.relative_to(ROOT)} -> {target}")
    assert not missing


def test_removed_result_docs_are_not_referenced() -> None:
    text = "\n".join(path.read_text(encoding="utf-8") for path in PUBLIC_MARKDOWN_FILES)
    removed_docs = ["main" + "_results.md", "ablation" + "_results.md"]
    for removed_doc in removed_docs:
        assert removed_doc not in text


def test_rl_recipe_docs_only_reference_released_variant() -> None:
    text = "\n".join(path.read_text(encoding="utf-8") for path in PUBLIC_MARKDOWN_FILES)
    forbidden = [
        "当前 recipe 有 5 个 setting",
        "`skill_only`",
        "`skill_prompt_verifier`",
        "`skill_mounted_verifier`",
        "`skill_mounted_verifier_plus`",
        "run_tulu3_8b_pointwise_skill_rm_grpo_2gpu.sh",
    ]
    for phrase in forbidden:
        assert phrase not in text
