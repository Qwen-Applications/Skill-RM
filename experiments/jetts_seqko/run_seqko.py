#!/usr/bin/env python3
"""Run an isolated JETTS sequential pairwise-KO BoN experiment."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.jetts_seqko.config import load_config, merge_config, normalize_endpoint_list  # noqa: E402
from experiments.jetts_seqko.data import load_dataset_records, response_score, response_text  # noqa: E402
from experiments.jetts_seqko.metrics import compute_metrics, render_metrics_md  # noqa: E402
from experiments.jetts_seqko.pairwise import openai_tool_schemas  # noqa: E402
from experiments.jetts_seqko.prompts import baseline_system_prompt, format_pair_prompt  # noqa: E402
from experiments.jetts_seqko.io_utils import (  # noqa: E402
    append_jsonl,
    atomic_write_json,
    json_default,
    load_completed,
    maybe_truncate_outputs,
    paths_for,
)
from experiments.jetts_seqko.seqko import map_label_to_index, process_sample  # noqa: E402
from experiments.jetts_seqko.tool_schemas import (  # noqa: E402
    LABELS,
    final_answer_tool_schema,
)
from skillrm.runners.rewardbench2 import load_skill_package  # noqa: E402


PRINT_LOCK = threading.Lock()


def run_one_setting_dataset(
    cfg: dict[str, Any],
    setting_name: str,
    dataset: str,
    records: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 0))
    output_dir = Path(cfg["output_dir"])
    run_paths = paths_for(output_dir, setting_name, dataset, seed)
    maybe_truncate_outputs(
        [run_paths.predictions, run_paths.traces, run_paths.metrics, run_paths.manifest],
        bool(args.force),
    )

    settings = cfg["settings"]
    setting_cfg = merge_config({k: v for k, v in cfg.items() if k != "settings"}, settings[setting_name])
    if args.timeout is not None:
        setting_cfg["timeout"] = float(args.timeout)
    if args.retries is not None:
        setting_cfg["retries"] = int(args.retries)
    if args.endpoint_failover_attempts is not None:
        setting_cfg["endpoint_failover_attempts"] = int(args.endpoint_failover_attempts)
    else:
        setting_cfg["endpoint_failover_attempts"] = cfg.get("endpoint_failover_attempts", 1)
    base_urls = normalize_endpoint_list(cfg.get("endpoints"))

    skill_package = load_skill_package(setting_cfg) if setting_cfg.get("mode") == "skill" else None
    existing = load_completed(run_paths.predictions) if args.resume else {}
    limit = args.limit if args.limit is not None else None
    selected_records = records[:limit] if limit is not None else records
    pending = [record for record in selected_records if str(record["sample_id"]) not in existing]

    manifest = {
        "setting": setting_name,
        "dataset": dataset,
        "generator": cfg["generator"],
        "seed": seed,
        "candidate_limit": int(args.candidate_limit or cfg.get("candidate_limit", 10)),
        "checkpoints": cfg.get("checkpoints", [1, 2, 4, 8, 10]),
        "workers": int(args.workers or cfg.get("workers", 32)),
        "selected_samples": len(selected_records),
        "completed_before_run": len(existing),
        "pending_at_start": len(pending),
        "output_files": {
            "predictions": str(run_paths.predictions),
            "traces": str(run_paths.traces),
            "metrics": str(run_paths.metrics),
        },
    }
    atomic_write_json(run_paths.manifest, manifest)

    with PRINT_LOCK:
        print(
            f"[{setting_name}/{dataset}] selected={len(selected_records)} "
            f"completed={len(existing)} pending={len(pending)} output={run_paths.predictions}",
            flush=True,
        )

    rows_by_sample = dict(existing)
    if pending:
        workers = int(args.workers or cfg.get("workers", 32))
        candidate_limit = int(args.candidate_limit or cfg.get("candidate_limit", 10))
        checkpoints = [int(x) for x in cfg.get("checkpoints", [1, 2, 4, 8, 10])]
        ordinal_by_id = {str(record["sample_id"]): i for i, record in enumerate(selected_records)}
        done_since_metrics = 0
        with futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(
                    process_sample,
                    record,
                    ordinal_by_id[str(record["sample_id"])],
                    setting_name,
                    setting_cfg,
                    base_urls,
                    seed,
                    candidate_limit,
                    checkpoints,
                    skill_package,
                ): record
                for record in pending
            }
            for future in futures.as_completed(future_map):
                record = future_map[future]
                sample_id = str(record["sample_id"])
                try:
                    row, trace_row = future.result()
                except Exception as exc:
                    row = {
                        "sample_id": sample_id,
                        "source_index": record.get("source_index"),
                        "dataset": dataset,
                        "generator": cfg["generator"],
                        "setting": setting_name,
                        "seed": seed,
                        "fatal_error": repr(exc),
                        "counts": {},
                    }
                    trace_row = {"sample_id": sample_id, "fatal_error": repr(exc)}
                append_jsonl(run_paths.predictions, row)
                append_jsonl(run_paths.traces, trace_row)
                rows_by_sample[sample_id] = row
                done_since_metrics += 1
                progress_every = int(cfg.get("progress_every", 10))
                if done_since_metrics >= progress_every:
                    atomic_write_json(run_paths.metrics, compute_metrics(rows_by_sample, cfg.get("checkpoints", [1, 2, 4, 8, 10])))
                    done_since_metrics = 0
                    with PRINT_LOCK:
                        print(
                            f"[{setting_name}/{dataset}] progress {len(rows_by_sample)}/{len(selected_records)}",
                            flush=True,
                        )

    metrics = compute_metrics(rows_by_sample, cfg.get("checkpoints", [1, 2, 4, 8, 10]))
    atomic_write_json(run_paths.metrics, metrics)
    with PRINT_LOCK:
        print(f"[{setting_name}/{dataset}] done n={metrics['n']} metrics={run_paths.metrics}", flush=True)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "jetts_seqko" / "qwen72b.example.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples per dataset for quick/debug runs.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--settings", nargs="+", default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--timeout", type=float, default=None, help="Override request timeout for this run.")
    parser.add_argument("--retries", type=int, default=None, help="Override request retries for this run.")
    parser.add_argument("--endpoint-failover-attempts", type=int, default=None)
    parser.add_argument("--candidate-limit", type=int, default=None, help="Override candidate_limit for quick/debug runs.")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--force", action="store_true", help="Truncate this experiment's own output files before running.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    normalize_endpoint_list(cfg.get("endpoints"))
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 0))
    settings = args.settings or list(cfg.get("settings", {}).keys())
    datasets = args.datasets or list(cfg.get("datasets", []))
    missing_settings = [name for name in settings if name not in cfg.get("settings", {})]
    if missing_settings:
        raise ValueError(f"Unknown settings: {missing_settings}")
    missing_datasets = [name for name in datasets if name not in cfg.get("datasets", [])]
    if missing_datasets:
        raise ValueError(f"Unknown datasets: {missing_datasets}")

    if "$" in str(cfg.get("data_dir", "")):
        raise ValueError("data_dir contains an unresolved placeholder. Set JETTS_DATA_DIR or edit the config.")
    data_dir = Path(cfg["data_dir"])
    records_by_dataset = {
        dataset: load_dataset_records(data_dir, dataset, str(cfg["generator"]))
        for dataset in datasets
    }

    output_dir = Path(cfg["output_dir"])
    aggregate_json = output_dir / f"aggregate_seed_{seed}.json"
    aggregate_md = output_dir / f"aggregate_seed_{seed}.md"
    all_metrics: dict[str, Any] = {}
    for setting in settings:
        for dataset in datasets:
            metrics = run_one_setting_dataset(cfg, setting, dataset, records_by_dataset[dataset], args)
            all_metrics[f"{setting}/{dataset}"] = metrics
            atomic_write_json(aggregate_json, all_metrics)
            aggregate_md.write_text(render_metrics_md(all_metrics), encoding="utf-8")

    atomic_write_json(aggregate_json, all_metrics)
    aggregate_md.write_text(render_metrics_md(all_metrics), encoding="utf-8")
    print(f"Aggregate metrics: {aggregate_json}", flush=True)
    print(f"Aggregate table: {aggregate_md}", flush=True)


if __name__ == "__main__":
    main()
