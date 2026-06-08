from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from .data import response_score, response_text
from .randomization import seeded_random


def score_value(score: float | None) -> float | None:
    return None if score is None else float(score)


GSM8K_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
GSM8K_HASH_RE = re.compile(r"####\s*([^\n]+)")
GSM8K_NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?")


def normalize_gsm8k_answer(text: str) -> str | None:
    if not text:
        return None
    match = GSM8K_HASH_RE.search(text)
    if match:
        raw = match.group(1)
    else:
        boxed = GSM8K_BOXED_RE.findall(text)
        if boxed:
            raw = boxed[-1]
        else:
            nums = GSM8K_NUM_RE.findall(text)
            if not nums:
                return None
            raw = nums[-1]
    raw = raw.strip().replace(",", "")
    raw = re.sub(r"\s+", "", raw)
    raw = raw.strip(".。,:;，；")
    return raw or None


def majority_vote_reference(dataset: str, responses: list[dict[str, Any]], candidate_indices: list[int], seed: int, sample_id: str) -> dict[str, Any]:
    if dataset != "gsm8k":
        return {
            "score": None,
            "index": None,
            "answer": None,
            "support": None,
            "parse_rate": None,
            "reason": "MV@10 is implemented only for GSM8K answer strings.",
        }
    answer_to_indices: dict[str, list[int]] = defaultdict(list)
    parsed = 0
    for idx in candidate_indices:
        answer = normalize_gsm8k_answer(response_text(responses[idx]))
        if answer is None:
            continue
        parsed += 1
        answer_to_indices[answer].append(idx)
    if not answer_to_indices:
        return {
            "score": None,
            "index": None,
            "answer": None,
            "support": 0,
            "parse_rate": 0.0,
            "reason": "No GSM8K answers parsed.",
        }
    max_support = max(len(v) for v in answer_to_indices.values())
    tied_answers = sorted(answer for answer, indices in answer_to_indices.items() if len(indices) == max_support)
    chosen_answer = seeded_random(seed, sample_id, "mv_tie").choice(tied_answers)
    chosen_index = answer_to_indices[chosen_answer][0]
    return {
        "score": score_value(response_score(responses[chosen_index])),
        "index": chosen_index,
        "answer": chosen_answer,
        "support": max_support,
        "parse_rate": parsed / max(1, len(candidate_indices)),
        "reason": None,
    }


def references_for_sample(
    dataset: str,
    sample_id: str,
    responses: list[dict[str, Any]],
    candidate_indices: list[int],
    seed: int,
) -> dict[str, Any]:
    first_idx = 0
    rng = seeded_random(seed, dataset, sample_id, "random@10")
    random_idx = rng.choice(candidate_indices)
    scored = [(idx, response_score(responses[idx])) for idx in candidate_indices]
    scored_valid = [(idx, score) for idx, score in scored if score is not None]
    oracle_idx, oracle_score = max(scored_valid, key=lambda x: x[1]) if scored_valid else (None, None)
    return {
        "pass@1": {
            "index": first_idx,
            "score": score_value(response_score(responses[first_idx])),
            "source": "original first response",
        },
        "random@10": {
            "index": random_idx,
            "score": score_value(response_score(responses[random_idx])),
            "source": "seeded sampled candidate",
        },
        "oracle@10": {
            "index": oracle_idx,
            "score": score_value(oracle_score),
            "source": "max metadata.score among candidates",
        },
        "mv@10": majority_vote_reference(dataset, responses, candidate_indices, seed, sample_id),
    }
