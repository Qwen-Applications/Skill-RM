from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .backend import build_backend
from .data import load_if_rewardbench
from .io_utils import file_sha256, load_jsonl, read_json, tree_fingerprint, write_json
from .judge import IFRewardBenchJudge
from .metrics import build_metrics, metrics_summary_markdown
from .paths import (
    DEFAULT_CONFIGS_DIR,
    DEFAULT_DATA_PATH,
    DEFAULT_METRICS_DIR,
    DEFAULT_POSITION_MAP_PATH,
    DEFAULT_RUNS_DIR,
    DEFAULT_SKILLS_DIR,
    default_config_for_mode,
)
from .skill_tools import SkillLoader


DEFAULT_ALLOWED_SKILLS = ["instruction_following"]


def main() -> None:
    args = parse_args()
    config = merge_config(read_config(args.config), args)
    run(config)


def run(config: dict[str, Any]) -> dict[str, Any]:
    mode = str(config["mode"]).lower()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(str(config.get("output_dir") or (DEFAULT_RUNS_DIR / f"{mode}_{timestamp}")))
    validate_input_paths(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.get("dataset_cache_path"):
        samples = read_json(Path(str(config["dataset_cache_path"])))
    else:
        samples = load_if_rewardbench(
            mode=mode,
            data_path=Path(str(config["data_path"])),
            position_map_path=Path(str(config["position_map_path"])),
            max_samples=int(config["samples"]) if config.get("samples") else None,
            seed=int(config.get("seed", 42)),
        )
    if config.get("limit_rows"):
        samples = samples[: int(config["limit_rows"])]

    backend = build_backend(config)
    skill_loader = SkillLoader(
        [Path(str(config["skills_dir"]))],
        allowed_skill_names=list(config.get("allowed_skills") or []),
    )
    judge = IFRewardBenchJudge(
        backend=backend,
        config_dir=Path(str(config["config_dir"])),
        skill_loader=skill_loader,
        max_agent_steps=int(config.get("max_agent_steps", 8)),
        tool_timeout=float(config.get("tool_timeout", 10.0)),
        overall_finalizer=bool(config.get("overall_finalizer", True)),
    )
    if config.get("max_tokens") is not None:
        judge.settings["max_tokens"] = int(config["max_tokens"])

    write_json(output_dir / "config_resolved.json", normalize_for_json(config))
    write_json(output_dir / "dataset_summary.json", summarize_samples(samples, mode=mode))
    write_json(output_dir / "run_manifest.json", build_manifest(config, samples=samples))

    completed = {}
    pred_path = output_dir / "predictions.jsonl"
    if config.get("resume") and pred_path.exists():
        completed = {str(row.get("id")): row for row in load_jsonl(pred_path)}
    pending = [sample for sample in samples if str(sample.get("id")) not in completed]
    rows_by_id: dict[str, dict[str, Any]] = dict(completed)

    started = time.time()
    output_mode = "a" if config.get("resume") else "w"
    with pred_path.open(output_mode, encoding="utf-8") as handle:
        workers = max(1, int(config.get("workers", 1)))
        if workers == 1:
            for idx, sample in enumerate(pending, start=1):
                row = evaluate_one(judge, sample, mode)
                rows_by_id[str(sample.get("id"))] = row
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                maybe_progress(idx, len(pending), config)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {executor.submit(evaluate_one, judge, sample, mode): sample for sample in pending}
                for idx, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                    sample = futures[future]
                    row = future.result()
                    rows_by_id[str(sample.get("id"))] = row
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    maybe_progress(idx, len(pending), config)

    rows = [rows_by_id[str(sample.get("id"))] for sample in samples if str(sample.get("id")) in rows_by_id]
    write_json(output_dir / "eval_results.json", rows)
    metrics = build_metrics(
        rows,
        mode=mode,
        data_path=Path(str(config["data_path"])),
        position_map_path=Path(str(config["position_map_path"])),
        metrics_dir=Path(str(config["metrics_dir"])),
    )
    metrics["runtime"] = {"seconds": round(time.time() - started, 3), "workers": int(config.get("workers", 1))}
    write_json(output_dir / "metrics.json", metrics)
    (output_dir / "summary.md").write_text(metrics_summary_markdown(metrics), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved run to: {output_dir}")
    return metrics


def evaluate_one(judge: IFRewardBenchJudge, sample: dict[str, Any], mode: str) -> dict[str, Any]:
    try:
        return judge.evaluate(sample, mode=mode)
    except Exception as exc:  # noqa: BLE001
        row = dict(sample)
        row.update({"status": "failed", "error": str(exc), "raw_output": "", "raw_output_raw": ""})
        return row


def maybe_progress(done: int, total: int, config: dict[str, Any]) -> None:
    every = int(config.get("progress_every", 25))
    if every > 0 and (done % every == 0 or done == total):
        print(f"{datetime.now().isoformat(timespec='seconds')} {done}/{total}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run self-contained Skill-RM IF-RewardBench experiments.")
    parser.add_argument("--config", type=str, default=None, help="Optional JSON/YAML config.")
    parser.add_argument("--mode", choices=["overall", "constraint"], default="overall")
    parser.add_argument("--config-bundle", type=str, default=None, help="Prompt/config bundle name under configs/if_rewardbench.")
    parser.add_argument("--data-path", type=str, default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--position-map-path", type=str, default=str(DEFAULT_POSITION_MAP_PATH))
    parser.add_argument("--metrics-dir", type=str, default=str(DEFAULT_METRICS_DIR))
    parser.add_argument("--skills-dir", type=str, default=str(DEFAULT_SKILLS_DIR))
    parser.add_argument("--allowed-skills", type=str, default=None, help="Comma-separated IF skills exposed to agentic runs.")
    parser.add_argument("--output", dest="output_dir", type=str, default=None)
    parser.add_argument("--backend", choices=["mock", "openai", "vllm"], default="mock")
    parser.add_argument("--base-url", dest="base_url", type=str, default=None)
    parser.add_argument("--base-urls", dest="base_urls", type=str, default=None, help="Comma-separated OpenAI-compatible base URLs.")
    parser.add_argument("--model", type=str, default="Qwen3.5-27B")
    parser.add_argument("--api-key", type=str, default="EMPTY")
    parser.add_argument("--samples", type=int, default=None, help="Prompt-level stratified sample count.")
    parser.add_argument(
        "--dataset-cache-path",
        type=str,
        default=None,
        help=(
            "Use an externally prepared expanded IF-RewardBench JSON. "
            "Official metrics still require --data-path, --metrics-dir, and --position-map-path for overall."
        ),
    )
    parser.add_argument("--limit-rows", type=int, default=None, help="Hard cap after mode expansion for quick or partial runs.")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-agent-steps", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-backend-attempts", type=int, default=None, help="Per request, try this many round-robin endpoints before failing.")
    parser.add_argument("--tool-timeout", type=float, default=10.0)
    parser.add_argument("--overall-finalizer", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--send-thinking-field",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Send chat_template_kwargs.enable_thinking to compatible Qwen/vLLM endpoints. Defaults to enabled.",
    )
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable model thinking when --send-thinking-field is active. Defaults to disabled for judge reproducibility.",
    )
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        if config_path.suffix.lower() in {".yaml", ".yml"}:
            return expand_env_vars(yaml.safe_load(handle) or {})
        return expand_env_vars(json.load(handle) or {})


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


def merge_config(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    merged = dict(config)
    for key, value in vars(args).items():
        if key == "config":
            continue
        if value is not None:
            merged[key] = value
    mode = str(merged.get("mode", "overall")).lower()
    bundle = str(merged.get("config_bundle") or default_config_for_mode(mode))
    merged["mode"] = mode
    merged["config_bundle"] = bundle
    merged["config_dir"] = str(Path(str(merged.get("config_dir") or (DEFAULT_CONFIGS_DIR / bundle))).resolve())
    for path_key, default in [
        ("data_path", DEFAULT_DATA_PATH),
        ("position_map_path", DEFAULT_POSITION_MAP_PATH),
        ("metrics_dir", DEFAULT_METRICS_DIR),
        ("skills_dir", DEFAULT_SKILLS_DIR),
    ]:
        merged[path_key] = str(Path(str(merged.get(path_key) or default)).resolve())
    if isinstance(merged.get("base_urls"), str):
        merged["base_urls"] = [item.strip() for item in str(merged["base_urls"]).split(",") if item.strip()]
    if merged.get("base_url") and not merged.get("base_urls"):
        merged["base_urls"] = [merged["base_url"]]
    merged["allowed_skills"] = normalize_allowed_skills(merged.get("allowed_skills"))
    merged.setdefault("backend", "mock")
    merged.setdefault("workers", 1)
    merged.setdefault("overall_finalizer", True)
    merged.setdefault("send_thinking_field", True)
    merged.setdefault("enable_thinking", False)
    merged.setdefault("progress_every", 25)
    merged.setdefault("resume", False)
    if merged.get("max_tokens") is not None:
        merged["max_tokens"] = int(merged["max_tokens"])
    if "use_cached_100" in merged:
        raise ValueError(
            "use_cached_100 is not a supported option. Use --limit-rows for smoke runs, "
            "or pass --dataset-cache-path with externally prepared data."
        )
    if merged.get("dataset_cache_path"):
        merged["dataset_cache_path"] = str(Path(str(merged["dataset_cache_path"])).resolve())
    return merged


def normalize_allowed_skills(value: Any) -> list[str]:
    if value is None:
        return list(DEFAULT_ALLOWED_SKILLS)
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",")]
    else:
        items = [str(item).strip() for item in value or []]
    return [item for item in items if item]


def validate_input_paths(config: dict[str, Any]) -> None:
    mode = str(config["mode"]).lower()
    missing: list[str] = []

    data_path = Path(str(config["data_path"]))
    if not data_path.exists():
        missing.append(f"data file not found: {data_path}")

    if config.get("dataset_cache_path"):
        cache_path = Path(str(config["dataset_cache_path"]))
        if not cache_path.exists():
            missing.append(f"expanded dataset cache not found: {cache_path}")

    if mode == "overall":
        position_map_path = Path(str(config["position_map_path"]))
        if not position_map_path.exists():
            missing.append(f"position map not found: {position_map_path}")

    metrics_dir = Path(str(config["metrics_dir"]))
    if not metrics_dir.is_dir():
        missing.append(f"metrics directory not found: {metrics_dir}")
    else:
        required_metric = {
            "overall": "analysis_overall_assessment.py",
            "constraint": "analysis_constraint_assessment.py",
        }.get(mode)
        if required_metric and not (metrics_dir / required_metric).exists():
            missing.append(f"metric script not found: {metrics_dir / required_metric}")

    if missing:
        detail = "\n".join(f"- {item}" for item in missing)
        raise FileNotFoundError(
            "IF-RewardBench inputs are not bundled with this repository.\n"
            f"{detail}\n"
            "Download IF-RewardBench separately and set SKILLRM_IF_REWARDBENCH_ROOT, "
            "or pass --data-path/--position-map-path/--metrics-dir explicitly. "
            "See docs/data_preparation.md."
        )


def summarize_samples(samples: list[dict[str, Any]], *, mode: str) -> dict[str, Any]:
    by_domain: dict[str, int] = {}
    for sample in samples:
        key = str(sample.get("domain") or "unknown")
        by_domain[key] = by_domain.get(key, 0) + 1
    return {"mode": mode, "n": len(samples), "by_domain": by_domain}


def build_manifest(config: dict[str, Any], *, samples: list[dict[str, Any]]) -> dict[str, Any]:
    data_path = Path(str(config["data_path"]))
    position_map = Path(str(config["position_map_path"]))
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": config.get("mode"),
        "config_bundle": config.get("config_bundle"),
        "backend": config.get("backend"),
        "model": config.get("model"),
        "dataset_n_loaded": len(samples),
        "data": path_fingerprint(data_path),
        "dataset_cache": path_fingerprint(Path(str(config["dataset_cache_path"]))) if config.get("dataset_cache_path") else {"path": None, "exists": False},
        "position_map": path_fingerprint(position_map),
        "config_dir": tree_fingerprint(Path(str(config["config_dir"])), {".txt", ".json"}),
        "skills_dir": tree_fingerprint(Path(str(config["skills_dir"])), {".md", ".yaml", ".yml", ".py"}),
        "allowed_skills": list(config.get("allowed_skills") or []),
        "metrics_dir": tree_fingerprint(Path(str(config["metrics_dir"])), {".py"}),
    }


def path_fingerprint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    data = {"path": str(path), "exists": True, "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if path.is_file() and stat.st_size < 250_000_000:
        data["sha256"] = file_sha256(path)
    return data


def normalize_for_json(config: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(config, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
