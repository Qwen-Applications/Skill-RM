from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .skill_package import is_rewardbench2_config


RESOURCE_ID_PATHS = {
    "sample.task_metadata": "sample/task_metadata.json",
    "sample.reference_or_ground_truth": "sample/reference_or_ground_truth.json",
    "sample.checklist": "sample/checklist_or_constraints.json",
    "sample.checklist_or_constraints": "sample/checklist_or_constraints.json",
    "rubric.generic_pairwise": "rubrics/generic_pairwise.md",
    "rubric.rewardbench2": "rubrics/rewardbench2.md",
    "rubric.rmbench": "rubrics/rmbench.md",
    "rubric.judgebench": "rubrics/judgebench.md",
    "principle.generic": "references/generic_principles.md",
    "principle.rewardbench2": "references/rewardbench2_principles.md",
    "principle.rmbench": "references/rmbench_principles.md",
    "principle.judgebench": "references/judgebench_principles.md",
    "verifier.protocol": "verifiers/hard_verifier_protocol.md",
    "tool.math_verify": "references/tool_math_verify.md",
    "tool.evalplus": "references/tool_evalplus.md",
    "tool.factool": "references/tool_factool.md",
    "bias_control": "references/bias_control.md",
    "aggregation.operational": "references/operational_aggregation.md",
    "aggregation.generic": "references/generic_aggregation.md",
    "output_format": "references/output_format.md",
}


def build_resource_index(skill_package: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    allowed_setting = str(config.get("skill_allowed_setting", "normal"))
    items = []
    for entry in skill_package.get("manifest") or []:
        if not resource_allowed(entry, allowed_setting):
            continue
        item = {
            "id": entry.get("id"),
            "type": entry.get("type"),
            "implementation_kind": entry.get("implementation_kind"),
            "subtype": entry.get("subtype"),
            "applies_to": entry.get("applies_to"),
            "cost": entry.get("cost"),
            "hard_or_soft": entry.get("hard_or_soft"),
            "decision_impact": entry.get("decision_impact"),
            "failure_modes_mitigated": entry.get("failure_modes_mitigated"),
            "inputs_required": entry.get("inputs_required"),
            "outputs_produced": entry.get("outputs_produced"),
            "leakage_level": entry.get("leakage_level"),
            "allowed_setting": entry.get("allowed_setting"),
            "path": resource_path_for_entry(entry),
        }
        items.append(item)
    return items


OPERATIONAL_METADATA_BLOCKED_KEYS = {
    "label",
    "labels",
    "gold",
    "gold_label",
    "winner",
    "preference",
    "preferred",
    "chosen",
    "rejected",
    "chosen_label",
    "rejected_label",
    "correct",
    "is_correct",
    "gt_is_chosen_correct",
    "score",
    "scores",
    "valid",
    "model",
    "models",
    "model_name",
    "model_names",
    "model_a",
    "model_b",
    "response_model",
    "response_models",
    "assistant_a_model",
    "assistant_b_model",
    "generator",
    "generators",
    "source_model",
    "source_models",
    "chosen_model",
    "rejected_model",
    "origin",
    "origins",
    "model_order",
    "response_order",
}


def sanitize_operational_metadata(value: Any) -> Any:
    """Remove direct answer/preference labels from sample metadata resources."""
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text.lower() in OPERATIONAL_METADATA_BLOCKED_KEYS:
                continue
            cleaned_item = sanitize_operational_metadata(item)
            if cleaned_item not in (None, "", [], {}):
                cleaned[key_text] = cleaned_item
        return cleaned
    if isinstance(value, list):
        cleaned_list = [sanitize_operational_metadata(item) for item in value]
        return [item for item in cleaned_list if item not in (None, "", [], {})]
    return value


def operational_sample_resources(
    record: dict[str, Any],
    formatted: dict[str, Any],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Build per-sample resources exposed only in the operational setting.

    These resources intentionally omit chosen/rejected origin and metric labels.
    They are visible through the skill resource interface after `use_skill`.
    """
    if str(config.get("skill_allowed_setting", "")) != "skill_operational":
        return [], {}

    metadata: dict[str, Any] = {}
    for key in (
        "benchmark",
        "subset",
        "query_type",
        "domain",
        "pair",
        "order",
        "category",
        "task_type",
        "source",
        "gt_question_type",
    ):
        if record.get(key) not in (None, ""):
            metadata[key] = record.get(key)
    if is_rewardbench2_config(config) and "benchmark" not in metadata:
        metadata["benchmark"] = "RewardBench2"
    if record.get("subset") and "task_type" not in metadata:
        metadata["task_type"] = record.get("subset")
    additional_metadata = sanitize_operational_metadata(record.get("additional_metadata"))
    if additional_metadata not in (None, "", [], {}):
        metadata["additional_metadata"] = additional_metadata

    files: dict[str, str] = {}
    index: list[dict[str, Any]] = []

    if metadata:
        path = "sample/task_metadata.json"
        files[path] = json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True)
        index.append(
            {
                "id": "sample.task_metadata",
                "type": "metadata",
                "implementation_kind": "sample_visible_json",
                "subtype": "benchmark_task_metadata",
                "applies_to": ["pairwise", "listwise", "scoring"],
                "cost": "low",
                "hard_or_soft": "soft",
                "decision_impact": "routing",
                "leakage_level": "benchmark_visible",
                "allowed_setting": ["skill_operational"],
                "path": path,
            }
        )

    reference: dict[str, Any] = {}
    for key in (
        "ground_truth",
        "reference",
        "answer",
        "expected_answer",
        "correct_answer",
        "gt",
        "gt_explanation",
    ):
        if record.get(key) not in (None, "", []):
            reference[key] = record.get(key)
    if reference:
        path = "sample/reference_or_ground_truth.json"
        files[path] = json.dumps(reference, ensure_ascii=False, indent=2, sort_keys=True)
        index.append(
            {
                "id": "sample.reference_or_ground_truth",
                "type": "reference",
                "implementation_kind": "sample_visible_json",
                "subtype": "reference_or_ground_truth",
                "applies_to": ["math", "factuality", "answer_selection", "listwise", "pairwise"],
                "cost": "low",
                "hard_or_soft": "hard",
                "decision_impact": "veto",
                "leakage_level": "sample_visible",
                "allowed_setting": ["skill_operational"],
                "path": path,
            }
        )

    checklist: dict[str, Any] = {}
    for key in ("constraints", "check_list", "checklist", "criteria", "rubric", "verifier_signal"):
        if record.get(key) not in (None, "", []):
            checklist[key] = record.get(key)
    if checklist:
        path = "sample/checklist_or_constraints.json"
        files[path] = json.dumps(checklist, ensure_ascii=False, indent=2, sort_keys=True)
        index.append(
            {
                "id": "sample.checklist_or_constraints",
                "type": "checklist",
                "implementation_kind": "sample_visible_json",
                "subtype": "constraints_or_checklist",
                "applies_to": ["instruction_following", "formatting", "pairwise", "listwise"],
                "cost": "low",
                "hard_or_soft": "hard",
                "decision_impact": "veto",
                "leakage_level": "sample_visible",
                "allowed_setting": ["skill_operational"],
                "path": path,
            }
        )

    return index, files


def combined_resource_index(
    skill_package: dict[str, Any],
    config: dict[str, Any],
    runtime_index: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    return build_resource_index(skill_package, config) + list(runtime_index or [])


def view_runtime_resource(
    path: str,
    runtime_files: dict[str, str],
    resources_viewed: list[str],
    *,
    max_chars: int,
) -> dict[str, Any] | None:
    normalized = normalize_skill_resource_path(path)
    candidates = [normalized]
    mapped = RESOURCE_ID_PATHS.get(normalized)
    if mapped:
        candidates.append(mapped)
    if normalized.startswith("sample/") and not Path(normalized).suffix:
        candidates.append(f"{normalized}.json")
    selected = next((candidate for candidate in candidates if candidate in runtime_files), None)
    if selected is None:
        return None
    resources_viewed.append(selected)
    content = runtime_files[selected]
    truncated = len(content) > max_chars
    return {
        "ok": True,
        "tool": "view_resource",
        "path": selected,
        "resource_id": selected.replace("/", ".").replace(".json", ""),
        "resource_type": "sample_resource",
        "leakage_level": "sample_visible",
        "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "truncated": truncated,
        "content": content[:max_chars],
    }


def normalize_resource_id(value: str) -> str:
    return str(value or "").strip()


def visible_reference_payload(record: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key in (
        "ground_truth",
        "reference",
        "answer",
        "expected_answer",
        "correct_answer",
        "gt",
        "gt_explanation",
    ):
        value = record.get(key)
        if value not in (None, "", []):
            payload[key] = value
    return payload


def normalize_answer_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", " ", str(text).lower())).strip()


def truncate_text(text: str, max_chars: int) -> str:
    value = str(text or "")
    return value if len(value) <= max_chars else value[:max_chars] + "...[truncated]"


def view_skill_resource(
    path: str,
    skill_package: dict[str, Any],
    config: dict[str, Any],
    resources_viewed: list[str],
) -> dict[str, Any]:
    normalized = normalize_skill_resource_path(path)
    if normalized not in skill_package["files"]:
        return {"ok": False, "tool": "view_resource", "path": normalized, "error": "resource not found"}
    max_resources = int(config.get("max_resources_per_sample", 4))
    if normalized not in resources_viewed and len(set(resources_viewed)) >= max_resources:
        return {
            "ok": False,
            "tool": "view_resource",
            "path": normalized,
            "error": f"max resources per sample exceeded: {max_resources}",
            "next_step": "Return the final judgment now using already viewed resources.",
        }
    entry = manifest_entry_for_path(skill_package, normalized)
    allowed_setting = str(config.get("skill_allowed_setting", "normal"))
    if entry and not resource_allowed(entry, allowed_setting):
        return {
            "ok": False,
            "tool": "view_resource",
            "path": normalized,
            "error": f"resource not allowed in {allowed_setting} setting",
            "leakage_level": entry.get("leakage_level"),
            "allowed_setting": entry.get("allowed_setting"),
        }
    resources_viewed.append(normalized)
    content = skill_package["files"][normalized]
    max_chars = int(config.get("max_resource_chars", 8000))
    truncated = len(content) > max_chars
    return {
        "ok": True,
        "tool": "view_resource",
        "path": normalized,
        "resource_id": entry.get("id") if entry else normalized,
        "resource_type": entry.get("type") if entry else None,
        "leakage_level": entry.get("leakage_level") if entry else None,
        "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "truncated": truncated,
        "content": content[:max_chars],
    }


def normalize_skill_resource_path(path: str) -> str:
    normalized = path.strip().lstrip("/")
    if not normalized or normalized.endswith("/") or ".." in Path(normalized).parts:
        raise ValueError(f"unsafe skill resource path: {path}")
    return normalized


def resource_path_for_entry(entry: dict[str, Any]) -> str | None:
    resource_id = str(entry.get("id") or "")
    path = entry.get("path")
    if isinstance(path, str) and path:
        return normalize_skill_resource_path(path)
    return RESOURCE_ID_PATHS.get(resource_id)


def manifest_entry_for_path(skill_package: dict[str, Any], path: str) -> dict[str, Any] | None:
    for entry in skill_package.get("manifest") or []:
        if resource_path_for_entry(entry) == path:
            return entry
    return None


def resource_allowed(entry: dict[str, Any], setting: str) -> bool:
    allowed_settings = entry.get("allowed_setting") or []
    leakage_level = str(entry.get("leakage_level") or "")
    if setting == "normal" and leakage_level == "oracle_only":
        return False
    return not allowed_settings or setting in allowed_settings

