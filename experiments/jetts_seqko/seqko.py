from __future__ import annotations

import time
from typing import Any

from .data import response_score, response_text
from .pairwise import pairwise_with_failover
from .randomization import seeded_random
from .references import references_for_sample, score_value


def map_label_to_index(label: str, display: dict[str, int]) -> int | None:
    return display.get(label)


def process_sample(
    record: dict[str, Any],
    sample_ordinal: int,
    setting_name: str,
    setting_cfg: dict[str, Any],
    base_urls: list[str],
    seed: int,
    candidate_limit: int,
    checkpoints: list[int],
    skill_package: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset = record["dataset"]
    sample_id = str(record["sample_id"])
    responses = record["responses"]
    k = min(candidate_limit, len(responses))
    candidate_indices = list(range(k))
    order = candidate_indices[:]
    seeded_random(seed, dataset, sample_id, "candidate_order").shuffle(order)

    references = references_for_sample(dataset, sample_id, responses, candidate_indices, seed)
    seqko: dict[str, Any] = {}
    matches: list[dict[str, Any]] = []
    match_traces: list[dict[str, Any]] = []

    incumbent = order[0]
    seen_count = 1
    if 1 in checkpoints:
        seqko["1"] = {
            "index": incumbent,
            "score": score_value(response_score(responses[incumbent])),
            "effective_k": 1,
            "order_position": 0,
        }

    request_count = 0
    tool_call_count = 0
    invalid_decisions = 0
    fallback_decisions = 0
    start = time.time()
    for pos, challenger in enumerate(order[1:], start=2):
        display_rng = seeded_random(seed, dataset, sample_id, "display_order", pos, incumbent, challenger)
        if display_rng.randrange(2) == 0:
            display = {"A": incumbent, "B": challenger}
        else:
            display = {"A": challenger, "B": incumbent}
        display_inverse = {idx: label for label, idx in display.items()}
        response_a = response_text(responses[display["A"]])
        response_b = response_text(responses[display["B"]])
        pair_row = pairwise_with_failover(
            setting_cfg=setting_cfg,
            mode=str(setting_cfg["mode"]),
            prompt=record["prompt"],
            response_a=response_a,
            response_b=response_b,
            dataset=dataset,
            sample_id=sample_id,
            base_urls=base_urls,
            endpoint_start=sample_ordinal + pos,
            skill_package=skill_package,
        )
        request_count += int(pair_row.get("request_count") or 0)
        tool_call_count += int(pair_row.get("tool_call_count") or 0)

        predicted_label = pair_row.get("predicted_label")
        winner = map_label_to_index(str(predicted_label), display) if predicted_label else None
        fallback_reason = None
        if winner is None:
            invalid_decisions += 1
            fallback_decisions += 1
            fallback_rng = seeded_random(seed, dataset, sample_id, "invalid_fallback", pos, incumbent, challenger)
            winner = fallback_rng.choice([incumbent, challenger])
            fallback_reason = "invalid_or_missing_judge_label"
        incumbent_before = incumbent
        incumbent = winner
        seen_count = pos

        match_summary = {
            "step": pos,
            "incumbent_before": incumbent_before,
            "challenger": challenger,
            "display": display,
            "incumbent_display_label": display_inverse[incumbent_before],
            "challenger_display_label": display_inverse[challenger],
            "predicted_label": predicted_label,
            "valid": bool(pair_row.get("valid")),
            "winner": winner,
            "winner_score": score_value(response_score(responses[winner])),
            "fallback_reason": fallback_reason,
            "base_url": pair_row.get("base_url"),
            "failover_attempts": pair_row.get("failover_attempts"),
            "request_error": pair_row.get("request_error"),
            "latency_s": pair_row.get("latency_s"),
        }
        matches.append(match_summary)
        match_traces.append({"step": pos, "summary": match_summary, "trace": pair_row.get("_trace", [])})

        if pos in checkpoints:
            seqko[str(pos)] = {
                "index": incumbent,
                "score": score_value(response_score(responses[incumbent])),
                "effective_k": seen_count,
                "order_position": order.index(incumbent),
            }

    for checkpoint in checkpoints:
        if str(checkpoint) not in seqko and checkpoint > k:
            seqko[str(checkpoint)] = {
                "index": incumbent,
                "score": score_value(response_score(responses[incumbent])),
                "effective_k": k,
                "order_position": order.index(incumbent),
                "reason": f"candidate_count={k} < checkpoint={checkpoint}",
            }

    row = {
        "sample_id": sample_id,
        "source_index": record["source_index"],
        "dataset": dataset,
        "generator": record["generator"],
        "setting": setting_name,
        "seed": seed,
        "candidate_count": k,
        "candidate_order": order,
        "references": references,
        "seqko": seqko,
        "matches": matches,
        "counts": {
            "pairwise_matches": len(matches),
            "request_count": request_count,
            "tool_call_count": tool_call_count,
            "invalid_decisions": invalid_decisions,
            "fallback_decisions": fallback_decisions,
        },
        "elapsed_s": time.time() - start,
    }
    trace_row = {
        "sample_id": sample_id,
        "dataset": dataset,
        "setting": setting_name,
        "seed": seed,
        "candidate_order": order,
        "match_traces": match_traces,
    }
    return row, trace_row
