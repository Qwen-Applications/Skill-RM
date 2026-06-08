from __future__ import annotations

import json
import threading
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from skillrm.common.io import append_jsonl as append_jsonl_row, load_jsonl_map


WRITE_LOCKS: dict[Path, threading.Lock] = defaultdict(threading.Lock)


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    return repr(obj)


def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    append_jsonl_row(path, row, default=json_default, lock=WRITE_LOCKS[path])


def load_completed(path: Path) -> dict[str, dict[str, Any]]:
    return load_jsonl_map(
        path,
        key="sample_id",
        missing_ok=True,
        skip_blank=True,
        ignore_missing_key=True,
        json_error_context=True,
    )


def maybe_truncate_outputs(output_paths: list[Path], force: bool) -> None:
    if not force:
        return
    for path in output_paths:
        if path.exists():
            path.unlink()


@dataclass
class RunPaths:
    predictions: Path
    traces: Path
    metrics: Path
    manifest: Path


def paths_for(output_dir: Path, setting: str, dataset: str, seed: int) -> RunPaths:
    run_dir = output_dir / setting / dataset / f"seed_{seed}"
    return RunPaths(
        predictions=run_dir / "predictions.jsonl",
        traces=run_dir / "traces.jsonl",
        metrics=run_dir / "metrics.json",
        manifest=run_dir / "manifest.json",
    )
