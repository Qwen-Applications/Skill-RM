from __future__ import annotations

import time
from typing import Any

import requests


def call_with_retries(
    base_url: str,
    messages: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
) -> dict[str, Any]:
    started_at = time.time()
    response_meta: dict[str, Any] = {}
    error = None
    for attempt in range(1, int(config.get("retries", 2)) + 2):
        try:
            response_meta = call_chat_completion(base_url, messages, config, tools=tools, tool_choice=tool_choice)
            break
        except requests.RequestException as exc:
            error = f"{type(exc).__name__}: {exc}"
            if attempt > int(config.get("retries", 2)):
                break
            time.sleep(min(2**attempt, 8))
    reasoning = response_meta.get("reasoning")
    output = {
        "latency_sec": time.time() - started_at,
        "content": response_meta.get("content", ""),
        "reasoning": reasoning,
        "thinking_field_sent": response_meta.get("thinking_field_sent"),
        "reasoning_len": len(reasoning) if isinstance(reasoning, str) else 0,
        "finish_reason": response_meta.get("finish_reason"),
        "tool_calls": response_meta.get("tool_calls") or [],
        "error": error,
    }
    return output


def call_chat_completion(
    base_url: str,
    messages: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    base_payload = {
        "model": config.get("model", "Qwen3.5-27B"),
        "messages": messages,
        "temperature": float(config.get("temperature", 0.0)),
        "top_p": float(config.get("top_p", 1.0)),
        "max_tokens": int(config.get("max_tokens", 512)),
    }
    if tools is not None:
        base_payload["tools"] = tools
    if tool_choice is not None:
        base_payload["tool_choice"] = tool_choice

    headers = {"Authorization": f"Bearer {config.get('api_key', 'EMPTY')}"}
    include_thinking_field = bool(config.get("send_thinking_field", True))
    response = None
    for _ in range(2):
        payload = dict(base_payload)
        if include_thinking_field:
            payload["chat_template_kwargs"] = {
                "enable_thinking": bool(config.get("enable_thinking", False))
            }
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=float(config.get("timeout", 180)),
        )
        if (
            response.status_code == 400
            and include_thinking_field
            and _should_retry_without_thinking_field(response.text)
        ):
            include_thinking_field = False
            continue
        response.raise_for_status()
        break
    if response is None:
        raise RuntimeError("No response returned from chat completion endpoint.")

    data = response.json()
    choice = data["choices"][0]
    message = choice["message"]
    reasoning = (
        message.get("reasoning")
        or message.get("reasoning_content")
        or message.get("reasoning_text")
    )
    return {
        "content": message.get("content") or "",
        "reasoning": reasoning,
        "message": message,
        "tool_calls": message.get("tool_calls") or [],
        "finish_reason": choice.get("finish_reason"),
        "thinking_field_sent": include_thinking_field,
    }


def _should_retry_without_thinking_field(body: str) -> bool:
    text = (body or "").lower()
    if "chat_template_kwargs" in text or "enable_thinking" in text:
        return True
    if (
        ("unknown field" in text or "extra fields" in text or "unexpected field" in text or "unrecognized field" in text)
        and ("thinking" in text or "template" in text)
    ):
        return True
    return False
