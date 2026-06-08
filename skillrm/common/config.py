from __future__ import annotations

from os import path as os_path
from pathlib import Path
from typing import Any

import yaml


DEFAULT_ENDPOINTS: list[str] = []


def load_config(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        return expand_env_vars(yaml.safe_load(handle) or {})


def expand_env_vars(value: Any) -> Any:
    if isinstance(value, str):
        return os_path.expandvars(value)
    if isinstance(value, list):
        return [expand_env_vars(item) for item in value]
    if isinstance(value, tuple):
        return tuple(expand_env_vars(item) for item in value)
    if isinstance(value, dict):
        return {key: expand_env_vars(item) for key, item in value.items()}
    return value


def normalize_base_urls(value: Any) -> list[str]:
    if isinstance(value, str):
        urls = [item.strip() for item in value.split(",") if item.strip()]
    else:
        urls = [str(item).strip() for item in value if str(item).strip()]
    if not urls:
        raise ValueError("At least one base URL is required. Set SKILLRM_BASE_URLS or pass --base-urls.")
    unresolved = [url for url in urls if "$" in url or url.startswith("<")]
    if unresolved:
        raise ValueError("Base URL configuration contains unresolved placeholders.")
    return [url.rstrip("/") for url in urls]

