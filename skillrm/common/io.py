from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


def print_progress(done: int, total: int, started_at: float) -> None:
    elapsed = max(time.time() - started_at, 1e-6)
    rate = done / elapsed
    remaining = (total - done) / rate if rate > 0 else None
    print(
        json.dumps(
            {
                "completed_this_run": done,
                "pending_this_run": total,
                "rate_per_sec": rate,
                "eta_sec": remaining,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_jsonl(
    path: Path,
    row: dict[str, Any],
    *,
    default: Callable[[Any], Any] | None = None,
    lock: Any | None = None,
    make_parent: bool = True,
) -> None:
    path = Path(path)
    if make_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False, default=default)

    def write_line() -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()

    if lock is None:
        write_line()
    else:
        with lock:
            write_line()


def read_jsonl(
    path: Path,
    *,
    missing_ok: bool = False,
    skip_blank: bool = True,
    errors: str | None = None,
    dict_only: bool = False,
) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        if missing_ok:
            return []
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors=errors or "strict") as handle:
        for line in handle:
            text = line.strip() if skip_blank else line
            if skip_blank and not text:
                continue
            value = json.loads(text)
            if dict_only and not isinstance(value, dict):
                continue
            rows.append(value)
    return rows


def load_jsonl_map(
    path: Path,
    *,
    key: str = "sample_id",
    missing_ok: bool = True,
    skip_blank: bool = True,
    ignore_missing_key: bool = False,
    json_error_context: bool = False,
) -> dict[str, dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        if missing_ok:
            return {}
        raise FileNotFoundError(path)
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, start=1):
            text = line.strip() if skip_blank else line
            if skip_blank and not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                if json_error_context:
                    raise RuntimeError(f"Invalid JSONL in {path}:{lineno}: {exc}") from exc
                raise
            if ignore_missing_key and row.get(key) is None:
                continue
            rows[str(row[key])] = row
    return rows


def write_summary(
    path: Path,
    config: dict[str, Any],
    examples: list[Any],
    rows: list[dict[str, Any]],
    metrics: dict[str, Any],
    elapsed: float,
) -> None:
    example_count = len(examples) if examples else int(metrics.get("n") or 0)
    micro_accuracy = metrics.get("micro_accuracy", metrics.get("micro_accuracy_non_ties"))
    macro_accuracy = metrics.get(
        "macro_accuracy_by_subset",
        metrics.get("macro_accuracy_non_ties_by_subset"),
    )
    invalid_rate = metrics.get("invalid_rate", metrics.get("invalid_rate_non_ties"))
    lines = [
        "# Qwen RBv2 Baseline Summary",
        "",
        f"- created_at: {datetime.now(timezone.utc).isoformat()}",
        f"- data_source: {config.get('data_source')}",
        f"- evaluation_mode: {config.get('evaluation_mode', 'official_compat')}",
        f"- model: {config.get('model')}",
        f"- examples: {example_count}",
        f"- completed: {len(rows)}",
        f"- elapsed_sec: {elapsed:.2f}",
        f"- micro_accuracy: {micro_accuracy}",
        f"- macro_accuracy_by_subset: {macro_accuracy}",
        f"- official_leaderboard_average: {metrics.get('official_leaderboard_average')}",
        f"- invalid_rate: {invalid_rate}",
        "",
        "Subset is hidden from the model and used only for metrics.",
    ]
    if "official_results_grouped" in metrics:
        lines.extend(["", "Official grouped results:", ""])
        for subset, value in sorted(metrics["official_results_grouped"].items()):
            lines.append(f"- {subset}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
