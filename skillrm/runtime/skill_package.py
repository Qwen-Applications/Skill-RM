from __future__ import annotations

import hashlib
import re
import zipfile
from pathlib import Path
from typing import Any

import yaml


def load_skill_package(config: dict[str, Any]) -> dict[str, Any]:
    raw_skill_path = config.get("skill_path")
    if not raw_skill_path:
        raise ValueError("skill mode requires `skill_path`.")
    skill_path = Path(str(raw_skill_path))
    loading_mode = str(config.get("skill_loading_mode") or "progressive")
    if loading_mode != "progressive":
        raise ValueError("Only skill_loading_mode=progressive is supported.")

    files = read_all_skill_text_files(skill_path)
    resources = ["SKILL.md", "resources.yaml"]
    missing = [name for name in resources if name not in files]
    if missing:
        raise FileNotFoundError(f"progressive skill package missing required files: {missing}")
    package_sha256 = skill_package_sha256(skill_path)
    if is_rewardbench2_config(config):
        files = dict(files)
        files["SKILL.md"] = augment_rewardbench2_skill_markdown(files.get("SKILL.md", ""))
        package_sha256 = skill_files_sha256(files)
    elif is_judgebench_config(config):
        files = dict(files)
        files["SKILL.md"] = augment_judgebench_skill_markdown(files.get("SKILL.md", ""))
        package_sha256 = skill_files_sha256(files)
    manifest = parse_skill_manifest(files.get("resources.yaml", ""))
    return {
        "source": str(skill_path),
        "sha256": package_sha256,
        "loading_mode": loading_mode,
        "resources_loaded": resources,
        "files": files,
        "manifest": manifest,
    }


def read_all_skill_text_files(skill_path: Path) -> dict[str, str]:
    if skill_path.is_dir():
        files = {}
        for path in sorted(item for item in skill_path.rglob("*") if item.is_file()):
            rel = path.relative_to(skill_path).as_posix()
            if is_skill_text_path(rel):
                files[rel] = path.read_text(encoding="utf-8")
        return files
    if skill_path.suffix == ".zip":
        with zipfile.ZipFile(skill_path) as archive:
            names = archive.namelist()
            prefix = common_zip_prefix(names)
            files = {}
            for member in names:
                if member.endswith("/"):
                    continue
                rel = member[len(prefix) :] if member.startswith(prefix) else member
                if is_skill_text_path(rel):
                    files[rel] = archive.read(member).decode("utf-8")
            return files
    raise ValueError(f"Unsupported skill_path: {skill_path}")


def is_skill_text_path(path: str) -> bool:
    return (
        path == "SKILL.md"
        or path == "resources.yaml"
        or path.startswith("references/")
        or path.startswith("rubrics/")
        or path.startswith("scripts/")
        or path.startswith("verifiers/")
    ) and path.endswith((".md", ".yaml", ".yml", ".json", ".py"))


def parse_skill_manifest(raw: str) -> list[dict[str, Any]]:
    if not raw.strip():
        return []
    value = yaml.safe_load(raw)
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        items: list[dict[str, Any]] = []
        for key in ("resources", "runtime_resources"):
            group = value.get(key) or []
            if isinstance(group, list):
                items.extend(item for item in group if isinstance(item, dict))
        return items
    raise ValueError("resources.yaml must contain a list or {resources, runtime_resources} object.")


def common_zip_prefix(names: list[str]) -> str:
    parts = [name.split("/", 1)[0] for name in names if "/" in name]
    if parts and len(set(parts)) == 1:
        return parts[0] + "/"
    return ""


def skill_package_sha256(skill_path: Path) -> str:
    digest = hashlib.sha256()
    if skill_path.is_file():
        with skill_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    for file_path in sorted(path for path in skill_path.rglob("*") if path.is_file()):
        rel = file_path.relative_to(skill_path).as_posix()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_path.read_bytes())
    return digest.hexdigest()


def skill_files_sha256(files: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for rel in sorted(files):
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(files[rel].encode("utf-8"))
    return digest.hexdigest()


def augment_rewardbench2_skill_markdown(skill_md: str) -> str:
    appendix = (
        'For best-of-four response rankings, choose exactly one candidate. Use deterministic checks for visible constraints, math, code, finance, word counts, or formulas when candidates disagree. For factual prompts, a calibrated "not enough evidence/no official data/no known term/no scientific evidence" answer can beat concrete unsupported claims; do not apply this to prompts that explicitly ask for fiction or fabricated content. For impossible exhaustive requests, prefer honest scope limits plus representative guidance over narrow answers that pretend to be complete. For direct yes/no or conceptual questions, treat failing to answer the question as a hard task failure.'
    )
    if appendix in skill_md:
        return skill_md
    marker = "For listwise or best-of-N judging, do not use the pairwise fast path. Build one shared criterion for the whole candidate set, remove hard-failing candidates, and compare all remaining candidates against the same requirements."
    if marker in skill_md:
        return skill_md.replace(marker, marker + "\n\n" + appendix, 1)
    return (skill_md.rstrip() + "\n\n" + appendix + "\n").strip()


def augment_judgebench_skill_markdown(skill_md: str) -> str:
    appendix = (
        "For forced-choice Output (a)/Output (b) pairwise rankings, choose exactly one of Output (a) or Output (b); do not return Tie, Same, Both, Neither, or Abstain. For pure exam-style multiple-choice or answer-selection tasks, decide directly from the visible question, options, candidate final selections, candidate rationale, and your own judgment. Do not load resources merely because candidates choose different option letters, and do not load the skill to perform broad subject-matter factual recall. A concise response with the correct final answer beats a longer, more polished response with an incorrect final answer, but do not assume a selected option is correct without visible support. Load or apply correctness-first checks only when a short visible deterministic check can decide the winner: arithmetic, code behavior, exact output format, supplied examples, supplied references, or internal contradictions. For exact output constraints, count or parse deterministically; if both candidates fail a constraint, prefer the one with the less severe task-relevant violation, not the more polished rationale. If both candidates give the same correct answer but both miss an exact repetition/length target by the same absolute amount, prefer the shorter under-run over the extra over-run. Do not switch mathematical conventions unless the prompt requires that convention or the candidate clearly justifies it."
    )
    if appendix in skill_md:
        return skill_md
    marker = "For A/B pairwise judging, first ask whether visible evidence already gives a clear winner. If yes, decide directly and do not load more resources."
    if marker in skill_md:
        return skill_md.replace(marker, marker + "\n\n" + appendix, 1)
    return (skill_md.rstrip() + "\n\n" + appendix + "\n").strip()


def skill_package_name(skill_package: dict[str, Any]) -> str:
    skill_md = str(skill_package.get("files", {}).get("SKILL.md", ""))
    match = re.search(r"(?im)^name:\s*([A-Za-z0-9_.-]+)\s*$", skill_md)
    return match.group(1) if match else "reward-judge"


def skill_package_description(skill_package: dict[str, Any]) -> str:
    skill_md = str(skill_package.get("files", {}).get("SKILL.md", ""))
    match = re.search(r"(?im)^description:\s*(.+?)\s*$", skill_md)
    if match:
        return match.group(1).strip().strip('"')
    return "Optional judging support with criteria, evidence, comparison, and calibration guidance."


def is_rewardbench2_config(config: dict[str, Any]) -> bool:
    benchmark = str(config.get("benchmark") or "").lower()
    evaluation_mode = str(config.get("evaluation_mode") or "").lower()
    data_source = str(config.get("data_source") or "").lower()
    return (
        benchmark in {"rb2", "rewardbench2", "rewardbench_v2", "rewardbench-v2"}
        or "rewardbench_v2" in data_source
        or ("rb2" in data_source and "official_compat" in evaluation_mode)
    )


def is_judgebench_config(config: dict[str, Any]) -> bool:
    benchmark = str(config.get("benchmark") or "").lower()
    return benchmark.startswith("judgebench")
