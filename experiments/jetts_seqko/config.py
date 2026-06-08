from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml


def merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge_config(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        cfg = expand_env_vars(yaml.safe_load(f))
    if not isinstance(cfg, dict):
        raise ValueError(f"Expected mapping config at {path}")
    return cfg


def expand_env_vars(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [expand_env_vars(item) for item in value]
    if isinstance(value, tuple):
        return tuple(expand_env_vars(item) for item in value)
    if isinstance(value, dict):
        return {key: expand_env_vars(item) for key, item in value.items()}
    return value


def normalize_endpoint_list(value: Any) -> list[str]:
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
    else:
        items = [str(item).strip() for item in (value or []) if str(item).strip()]
    if not items:
        raise ValueError("No endpoints configured. Set SKILLRM_BASE_URLS or provide endpoints in the config.")
    if any("$" in item or item.startswith("<") for item in items):
        raise ValueError("Endpoint config still contains an unresolved placeholder.")
    return [item.rstrip("/") for item in items]

