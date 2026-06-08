from __future__ import annotations

from collections import Counter
from typing import Any


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def collect_metric_values(rows: list[dict[str, Any]], metric: str) -> list[float]:
    out = []
    for row in rows:
        if metric.startswith("seqko@"):
            k = metric.split("@", 1)[1]
            value = row.get("seqko", {}).get(k, {}).get("score")
        else:
            value = row.get("references", {}).get(metric, {}).get("score")
        if value is not None:
            out.append(float(value))
    return out


def compute_metrics(rows_by_sample: dict[str, dict[str, Any]], checkpoints: list[int]) -> dict[str, Any]:
    rows = list(rows_by_sample.values())
    metrics: dict[str, Any] = {"n": len(rows), "scores": {}, "counts": {}}
    for metric in ["pass@1", "random@10", "oracle@10", "mv@10"]:
        values = collect_metric_values(rows, metric)
        metrics["scores"][metric] = {"mean": mean(values), "sum": sum(values), "n": len(values)}
    for checkpoint in checkpoints:
        metric = f"seqko@{checkpoint}"
        values = collect_metric_values(rows, metric)
        metrics["scores"][metric] = {"mean": mean(values), "sum": sum(values), "n": len(values)}
    counts = Counter()
    for row in rows:
        for key, value in (row.get("counts") or {}).items():
            if isinstance(value, (int, float)):
                counts[key] += value
    metrics["counts"] = dict(counts)
    if counts.get("pairwise_matches"):
        metrics["rates"] = {
            "invalid_decision_rate": counts.get("invalid_decisions", 0) / counts["pairwise_matches"],
            "fallback_decision_rate": counts.get("fallback_decisions", 0) / counts["pairwise_matches"],
        }
    else:
        metrics["rates"] = {}
    return metrics


def render_metrics_md(all_metrics: dict[str, Any]) -> str:
    lines = ["# JETTS Sequential KO Metrics", ""]
    lines.append("| Setting | Dataset | N | Pass@1 | Random@10 | Oracle@10 | MV@10 | SeqKO@1 | SeqKO@2 | SeqKO@4 | SeqKO@8 | SeqKO@10 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key in sorted(all_metrics):
        setting, dataset = key.split("/", 1)
        metrics = all_metrics[key]
        scores = metrics.get("scores", {})

        def fmt(name: str) -> str:
            value = scores.get(name, {}).get("mean")
            return "" if value is None else f"{value:.4f}"

        lines.append(
            "| "
            + " | ".join(
                [
                    setting,
                    dataset,
                    str(metrics.get("n", 0)),
                    fmt("pass@1"),
                    fmt("random@10"),
                    fmt("oracle@10"),
                    fmt("mv@10"),
                    fmt("seqko@1"),
                    fmt("seqko@2"),
                    fmt("seqko@4"),
                    fmt("seqko@8"),
                    fmt("seqko@10"),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"

