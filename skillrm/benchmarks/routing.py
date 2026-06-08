from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RouteDecision:
    action: str
    key: str
    group_by: str
    policy_name: str | None
    reason: str


def config_string_set(value: Any) -> set[str]:
    if value is None or value is False:
        return set()
    if isinstance(value, str):
        return {item.strip() for item in value.split(",") if item.strip()}
    return {str(item).strip() for item in value if str(item).strip()}


def load_route_policy(config: dict[str, Any]) -> dict[str, Any] | None:
    raw_policy = config.get("route_policy")
    if raw_policy is None or raw_policy is False:
        return None
    if isinstance(raw_policy, dict):
        return raw_policy
    policy_path = Path(str(raw_policy))
    with policy_path.open("r", encoding="utf-8") as handle:
        if policy_path.suffix.lower() in {".yaml", ".yml"}:
            loaded = yaml.safe_load(handle)
        else:
            loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"route_policy must resolve to an object: {raw_policy}")
    loaded.setdefault("path", str(policy_path))
    return loaded


def route_decision_for_metadata(metadata: dict[str, Any], config: dict[str, Any]) -> RouteDecision | None:
    policy = load_route_policy(config)
    if not policy:
        return None
    group_by = str(policy.get("group_by") or "query_type")
    key = route_key(metadata, group_by)
    default_action = normalize_action(policy.get("default_action", "skill"))
    action_by_group = {
        str(group): normalize_action(action)
        for group, action in dict(policy.get("action_by_group") or policy.get("routes") or {}).items()
    }
    if key in action_by_group:
        action = action_by_group[key]
        source = "action_by_group"
    else:
        skill_groups = config_string_set(policy.get("skill_groups"))
        baseline_groups = config_string_set(policy.get("baseline_groups"))
        if key in skill_groups:
            action = "skill"
            source = "skill_groups"
        elif key in baseline_groups:
            action = "baseline"
            source = "baseline_groups"
        else:
            action = default_action
            source = "default_action"
    name = policy.get("name") or policy.get("path")
    return RouteDecision(
        action=action,
        key=key,
        group_by=group_by,
        policy_name=str(name) if name else None,
        reason=f"route_policy:{source}:{group_by}:{key}:{action}",
    )


def route_key(metadata: dict[str, Any], group_by: str) -> str:
    if group_by == "domain_pair":
        return f"{metadata.get('domain') or metadata.get('query_type') or 'unknown'}:{metadata.get('pair') or 'unknown'}"
    if group_by == "query_type_pair":
        return f"{metadata.get('query_type') or metadata.get('domain') or 'unknown'}:{metadata.get('pair') or 'unknown'}"
    if group_by == "subset":
        return str(metadata.get("subset") or metadata.get("subset_for_metrics_only") or "unknown")
    if group_by == "domain":
        return str(metadata.get("domain") or metadata.get("query_type") or "unknown")
    if group_by == "pair":
        return str(metadata.get("pair") or "unknown")
    return str(metadata.get("query_type") or metadata.get("domain") or "unknown")


def normalize_action(value: Any) -> str:
    action = str(value or "skill").strip().lower()
    if action in {"baseline", "fallback", "baseline_fallback", "baseline_cached", "no_skill"}:
        return "baseline"
    if action in {"skill", "self_select_skill", "use_skill"}:
        return "skill"
    raise ValueError(f"unsupported route action: {value}")


def route_row_fields(decision: RouteDecision | None) -> dict[str, Any]:
    if decision is None:
        return {}
    return {
        "route_policy": decision.policy_name,
        "route_group_by": decision.group_by,
        "route_key": decision.key,
        "route_action": decision.action,
        "route_reason": decision.reason,
    }


def route_source_selection(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any] | None:
    routed_rows = [row for row in rows if row.get("route_action")]
    if not routed_rows:
        return None
    by_action = Counter(str(row.get("route_action")) for row in routed_rows)
    by_key_action: dict[str, Counter[str]] = defaultdict(Counter)
    for row in routed_rows:
        by_key_action[str(row.get("route_key"))][str(row.get("route_action"))] += 1
    selected_skill_groups = sorted(key for key, counts in by_key_action.items() if counts.get("skill", 0) > 0)
    selected_baseline_groups = sorted(key for key, counts in by_key_action.items() if counts.get("baseline", 0) > 0)
    return {
        "route_policy": routed_rows[0].get("route_policy"),
        "route_group_by": routed_rows[0].get("route_group_by"),
        "source_counts": dict(sorted(by_action.items())),
        "selected_skill_groups": selected_skill_groups,
        "selected_baseline_groups": selected_baseline_groups,
        "group_source_counts": {key: dict(sorted(counts.items())) for key, counts in sorted(by_key_action.items())},
        "config_route_policy": config.get("route_policy"),
    }
