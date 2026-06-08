"""Normalize VerInstruct-style sample verifier resources."""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


_CHECKER_CONTAINER_KEYS = ("checkers", "items", "constraints", "rules", "rubrics", "data")
_FUNCTION_CONTAINER_KEYS = ("functions", "items", "rules", "verifiers", "data")
_CHECKER_TEXT_KEYS = (
    "text",
    "checker",
    "constraint",
    "description",
    "instruction",
    "content",
    "rubric",
    "requirement",
    "value",
    "question",
)
_CHECKER_ID_KEYS = ("item_id", "id", "checker_id", "check_id", "uid")
_FUNCTION_REF_KEYS = (
    "function",
    "function_name",
    "function_id",
    "function_key",
    "verifier",
    "verifier_name",
    "verifier_id",
    "verifier_key",
    "function_ref",
    "verifier_ref",
    "checker_function",
)
_FUNCTION_NAME_KEYS = ("name", "function_name", "verifier_name", "id", "function_id", "key", "verifier_key")
_FUNCTION_CODE_KEYS = (
    "code",
    "python",
    "python_code",
    "source",
    "source_code",
    "function_code",
    "verifier_code",
    "implementation",
    "script",
)
_RULE_TAG_KEYS = ("tag", "type", "checker_type", "verifier_type", "mode", "kind", "source", "method")
_RULE_BOOL_KEYS = ("is_rule", "rule_based", "uses_code", "use_code", "has_verifier")


@dataclass(frozen=True)
class VerifierFunction:
    name: str = ""
    identifier: str = ""
    key: str = ""
    code: str = ""

    @property
    def display_name(self) -> str:
        return self.name or self.identifier or self.key

    @property
    def aliases(self) -> set[str]:
        return {_normalize_token(value) for value in (self.name, self.identifier, self.key) if _normalize_token(value)}


@dataclass(frozen=True)
class ChecklistItem:
    item_id: str
    text: str
    tag: str
    verifier_name: str = ""
    verifier_code: str = ""


@dataclass(frozen=True)
class SampleVerificationResources:
    checklist_items: list[ChecklistItem]
    parse_error: str = ""

    @property
    def sample_resources_present(self) -> int:
        return int(bool(self.checklist_items))

    @property
    def sample_checklist_count(self) -> int:
        return len(self.checklist_items)

    @property
    def sample_rule_count(self) -> int:
        return sum(1 for item in self.checklist_items if item.tag == "rule")

    @property
    def sample_llm_count(self) -> int:
        return sum(1 for item in self.checklist_items if item.tag != "rule")

    def to_metadata(self) -> dict[str, Any]:
        return {
            "sample_resources_present": self.sample_resources_present,
            "sample_checklist_count": self.sample_checklist_count,
            "sample_rule_count": self.sample_rule_count,
            "sample_llm_count": self.sample_llm_count,
            "sample_resource_parse_error": self.parse_error,
        }


def normalize_sample_resources(extra_info: dict[str, Any] | None) -> SampleVerificationResources:
    info = extra_info or {}
    parse_errors: list[str] = []
    raw_checkers, checker_error = _parse_json_like(info.get("checkers"), "extra_info['checkers']")
    raw_functions, function_error = _parse_json_like(info.get("functions"), "extra_info['functions']")
    if checker_error:
        parse_errors.append(checker_error)
    if function_error:
        parse_errors.append(function_error)

    checker_entries = _coerce_entries(raw_checkers, _CHECKER_CONTAINER_KEYS, _CHECKER_TEXT_KEYS + _CHECKER_ID_KEYS)
    function_entries = _coerce_entries(raw_functions, _FUNCTION_CONTAINER_KEYS, _FUNCTION_NAME_KEYS + _FUNCTION_CODE_KEYS)
    verifier_functions = [_normalize_function_entry(entry) for entry in function_entries]
    function_lookup = _build_function_lookup(verifier_functions)
    allow_index_fallback = len(checker_entries) == len(verifier_functions) and bool(checker_entries)

    items: list[ChecklistItem] = []
    for index, checker_entry in enumerate(checker_entries, start=1):
        item = _normalize_checker_entry(
            checker_entry=checker_entry,
            index=index,
            verifier_functions=verifier_functions,
            function_lookup=function_lookup,
            allow_index_fallback=allow_index_fallback,
        )
        if item:
            items.append(item)
    if checker_entries and not items:
        parse_errors.append("No checklist items could be extracted from extra_info['checkers'].")
    return SampleVerificationResources(checklist_items=items, parse_error="; ".join(parse_errors))


def render_sample_resources_section(resources: SampleVerificationResources | None) -> str:
    if resources is None or not resources.checklist_items:
        return ""
    lines = [
        "[VerInstruct Verifier Resources - begin]",
        "These resources are visible evidence for the current sample. They may be incomplete or noisy; use them to score the response, not as hidden labels.",
        "",
        "[Checklist]",
    ]
    for item in resources.checklist_items:
        line = f"- {item.item_id} [{item.tag}]: {item.text}"
        if item.tag == "rule" and item.verifier_name:
            line += f" ({item.verifier_name})"
        elif item.tag == "rule" and not item.verifier_code:
            line += " (verifier code unavailable)"
        lines.append(line)

    grouped = _group_rule_verifier_code(resources.checklist_items)
    if grouped:
        lines.extend(["", "[Rule Verifier Code]"])
        for idx, (code, meta) in enumerate(grouped.items(), start=1):
            title = meta["name"] or f"verifier_{idx}"
            lines.append(f"- {title} -> {', '.join(meta['item_ids'])}")
            lines.append("```python")
            lines.extend(code.splitlines() or [""])
            lines.append("```")
    lines.append("[VerInstruct Verifier Resources - end]")
    return "\n".join(lines)


def _parse_json_like(value: Any, label: str) -> tuple[Any, str]:
    if value is None:
        return None, ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None, ""
        try:
            return json.loads(stripped), ""
        except json.JSONDecodeError as exc:
            return None, f"Failed to parse {label}: {exc}"
    return value, ""


def _coerce_entries(payload: Any, container_keys: tuple[str, ...], direct_keys: tuple[str, ...]) -> list[Any]:
    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    if isinstance(payload, tuple):
        return list(payload)
    if isinstance(payload, dict):
        for key in container_keys:
            candidate = payload.get(key)
            if isinstance(candidate, list):
                return candidate
        if any(key in payload for key in direct_keys):
            return [payload]
        entries: list[Any] = []
        for key, value in payload.items():
            if isinstance(value, dict):
                entry = dict(value)
                entry.setdefault("id", str(key))
                entry.setdefault("name", str(key))
                entries.append(entry)
            elif isinstance(value, str):
                entries.append({"id": str(key), "text": value, "code": value})
        return entries
    return []


def _normalize_function_entry(entry: Any) -> VerifierFunction:
    if isinstance(entry, str):
        return VerifierFunction(code=entry)
    if not isinstance(entry, dict):
        return VerifierFunction()
    return VerifierFunction(
        name=_extract_string(entry, _FUNCTION_NAME_KEYS),
        identifier=_extract_string(entry, ("id", "function_id", "verifier_id", "uid")),
        key=_extract_string(entry, ("key", "function_key", "verifier_key")),
        code=_extract_string(entry, _FUNCTION_CODE_KEYS),
    )


def _build_function_lookup(verifier_functions: list[VerifierFunction]) -> dict[str, VerifierFunction]:
    lookup: dict[str, VerifierFunction] = {}
    for verifier in verifier_functions:
        for alias in verifier.aliases:
            lookup.setdefault(alias, verifier)
    return lookup


def _normalize_checker_entry(
    *,
    checker_entry: Any,
    index: int,
    verifier_functions: list[VerifierFunction],
    function_lookup: dict[str, VerifierFunction],
    allow_index_fallback: bool,
) -> ChecklistItem | None:
    if isinstance(checker_entry, str):
        text = checker_entry.strip()
        return ChecklistItem(item_id=f"C{index}", text=text, tag="llm") if text else None
    if not isinstance(checker_entry, dict):
        return None

    text = _extract_string(checker_entry, _CHECKER_TEXT_KEYS)
    if not text:
        return None
    item_id = _extract_string(checker_entry, _CHECKER_ID_KEYS) or f"C{index}"
    inline_code = _extract_string(checker_entry, _FUNCTION_CODE_KEYS)
    explicit_rule = _is_explicit_rule(checker_entry)
    explicit_llm = _is_explicit_llm(checker_entry)
    matched = _match_verifier_function(
        checker_entry=checker_entry,
        verifier_functions=verifier_functions,
        function_lookup=function_lookup,
        index=index,
        allow_index_fallback=allow_index_fallback,
    )
    code = inline_code or (matched.code if matched else "")
    verifier_name = (matched.display_name if matched else "") or _extract_string(checker_entry, _FUNCTION_REF_KEYS)
    tag = "rule" if (explicit_rule or code) and not explicit_llm else "llm"
    return ChecklistItem(item_id=item_id, text=text, tag=tag, verifier_name=verifier_name, verifier_code=code)


def _match_verifier_function(
    *,
    checker_entry: dict[str, Any],
    verifier_functions: list[VerifierFunction],
    function_lookup: dict[str, VerifierFunction],
    index: int,
    allow_index_fallback: bool,
) -> VerifierFunction | None:
    for key in _FUNCTION_REF_KEYS:
        token = _normalize_token(checker_entry.get(key))
        if token and token in function_lookup:
            return function_lookup[token]
    if allow_index_fallback and 0 <= index - 1 < len(verifier_functions):
        return verifier_functions[index - 1]
    return None


def _extract_string(entry: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = entry.get(key)
        if value is None:
            continue
        if isinstance(value, (dict, list, tuple)):
            text = json.dumps(value, ensure_ascii=False)
        else:
            text = str(value)
        text = text.strip()
        if text:
            return text
    return ""


def _is_explicit_rule(entry: dict[str, Any]) -> bool:
    for key in _RULE_BOOL_KEYS:
        if entry.get(key) is True:
            return True
    for key in _RULE_TAG_KEYS:
        value = str(entry.get(key, "")).lower()
        if any(token in value for token in ("rule", "code", "python", "program", "deterministic")):
            return True
    return False


def _is_explicit_llm(entry: dict[str, Any]) -> bool:
    for key in _RULE_TAG_KEYS:
        value = str(entry.get(key, "")).lower()
        if any(token in value for token in ("llm", "model", "semantic", "human")):
            return True
    return False


def _normalize_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _group_rule_verifier_code(items: list[ChecklistItem]) -> OrderedDict[str, dict[str, Any]]:
    grouped: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for item in items:
        if item.tag != "rule" or not item.verifier_code:
            continue
        grouped.setdefault(item.verifier_code, {"name": item.verifier_name, "item_ids": []})
        grouped[item.verifier_code]["item_ids"].append(item.item_id)
    return grouped

