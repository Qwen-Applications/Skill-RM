from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from .io_utils import read_json
from .paths import DEFAULT_DATA_PATH, DEFAULT_POSITION_MAP_PATH


def build_if_history(messages: list[dict[str, str]]) -> tuple[str, str, str]:
    """Match the copied IF-RewardBench prompt/history extraction."""
    system_prompt = ""
    history = ""
    cnt = 1
    for turn in messages[:-1]:
        role = turn.get("role")
        content = str(turn.get("content") or "")
        if role == "system":
            system_prompt = content
        elif role == "user":
            if history:
                history += "\n\n"
            history += f"[第{cnt}轮用户指令-开始]\n{content.strip()}\n[第{cnt}轮用户指令-结束]"
        elif role == "assistant":
            if history:
                history += "\n\n"
            history += f"[第{cnt}轮人工智能助手的回复-开始]\n{content.strip()}\n[第{cnt}轮人工智能助手的回复-结束]"
            cnt += 1
    user_prompt = str(messages[-1].get("content") or "") if messages else ""
    return system_prompt, history, user_prompt


def load_if_rewardbench(
    *,
    mode: str,
    data_path: str | Path = DEFAULT_DATA_PATH,
    position_map_path: str | Path = DEFAULT_POSITION_MAP_PATH,
    max_samples: int | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Load IF-RewardBench in the same expanded shapes as the copied protocol."""
    mode = mode.strip().lower()
    data = read_json(data_path)
    if max_samples:
        data = stratified_prompt_sample(data, max_samples=max_samples, seed=seed)

    if mode == "overall":
        position_maps = read_json(position_map_path) if Path(position_map_path).exists() else {}
        return expand_overall(data, position_maps=position_maps, seed=seed)
    if mode == "constraint":
        return expand_constraint(data)
    raise ValueError(f"unsupported IF-RewardBench mode: {mode}")


def stratified_prompt_sample(data: list[dict[str, Any]], *, max_samples: int, seed: int) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in data:
        buckets[str(row.get("instruction_type", "Unknown"))].append(row)
    target = min(max_samples, len(data))
    ratio = target / max(1, len(data))
    rnd = random.Random(seed)
    sampled: list[dict[str, Any]] = []
    for _itype, items in buckets.items():
        bucket_target = max(1, int(len(items) * ratio))
        bucket_target = min(bucket_target, len(items))
        sampled.extend(rnd.sample(items, bucket_target))
    return sampled[:target]


def expand_overall(data: list[dict[str, Any]], *, position_maps: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    rnd = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for d in data:
        system_prompt, history, user_prompt = build_if_history(d.get("messages", []))
        responses = d.get("responses", [])
        for u in range(len(responses)):
            for v in range(len(responses)):
                if u == v:
                    continue
                pair_key = f"{min(u, v)}_{max(u, v)}"
                pos_map_for_id = position_maps.get(str(d.get("id")), {})
                position = pos_map_for_id.get(pair_key)
                if position is None:
                    position = 0 if rnd.random() < 0.5 else 1
                if (position == 0 and u < v) or (position == 1 and u > v):
                    rows.append(
                        {
                            "id": f"ifrb_{d['id']}_{u}_{v}",
                            "original_id": str(d["id"]),
                            "pair_indices": [u, v],
                            "prompt": user_prompt,
                            "system_prompt": system_prompt,
                            "history": history,
                            "response_a": str(responses[u].get("response") or ""),
                            "response_b": str(responses[v].get("response") or ""),
                            "domain": d.get("instruction_type", "unknown"),
                            "label": "unknown",
                        }
                    )
    return rows


def expand_constraint(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for d in data:
        system_prompt, history, user_prompt = build_if_history(d.get("messages", []))
        checklist = render_checklist(d.get("checklist") or [])
        for j, response in enumerate(d.get("responses", [])):
            rows.append(
                {
                    "id": f"ifrb_{d['id']}_{j}",
                    "original_id": str(d["id"]),
                    "response_index": j,
                    "prompt": user_prompt,
                    "system_prompt": system_prompt,
                    "history": history,
                    "response_a": str(response.get("response") or ""),
                    "response_b": "N/A",
                    "checklist": checklist,
                    "domain": d.get("instruction_type", "unknown"),
                    "label": "unknown",
                }
            )
    return rows


def render_checklist(checklist: list[Any]) -> str:
    chunks: list[str] = []
    for idx, item in enumerate(checklist, start=1):
        text = str(item).strip()
        if not text:
            continue
        chunks.append(f"[检查项{idx}-开始]\n{text}\n[检查项{idx}-结束]")
    return "\n\n".join(chunks)
