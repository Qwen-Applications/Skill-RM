"""OpenAI-compatible and mock chat backends for pointwise Skill-RM."""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Protocol

import requests
from requests.adapters import HTTPAdapter


class ChatBackend(Protocol):
    def complete_response(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ...


def _tool_response(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "content": "",
        "tool_calls": [
            {
                "id": f"call_mock_{name}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
            }
        ],
        "finish_reason": "tool_calls",
    }


class MockBackend:
    """Deterministic backend for offline smoke tests."""

    def complete_response(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del max_tokens, tool_choice
        transcript = "\n".join(str(message.get("content") or "") for message in messages)
        tool_names = {
            str((tool.get("function") or {}).get("name"))
            for tool in (tools or [])
            if isinstance(tool.get("function"), dict)
        }
        if "use_skill" in tool_names and '"tool": "use_skill"' not in transcript:
            return _tool_response(
                "use_skill",
                {
                    "skill_name": "instruction_following_pointwise",
                    "reason": "load the pointwise instruction-following skill for smoke testing",
                },
            )
        if "run_resource" in tool_names and "sample.verinstruct.verify_all" in transcript and '"tool": "run_resource"' not in transcript:
            return _tool_response(
                "run_resource",
                {
                    "resource_id": "sample.verinstruct.verify_all",
                    "reason": "use sample verifier evidence",
                },
            )
        if "python_sandbox" in tool_names and '"tool": "python_sandbox"' not in transcript:
            return _tool_response(
                "python_sandbox",
                {
                    "reason": "smoke-test deterministic visible response checks",
                    "constraint_id": "mock_word_and_list_check",
                    "code": (
                        "result = {"
                        "'word_count': word_count(response), "
                        "'numbered_items': numbered_list_count(response), "
                        "'balanced_brackets': balanced_brackets(response)"
                        "}"
                    ),
                },
            )
        score = 0.8
        satisfied_count = 4
        total_count = 5
        if "ignore the request" in transcript.lower() or "[response]\nno" in transcript.lower():
            score = 0.2
            satisfied_count = 1
        payload = {
            "score": score,
            "satisfied_count": satisfied_count,
            "total_count": total_count,
            "confidence": 0.7,
            "used_resources": ["mock_backend"],
            "reason": "deterministic mock score",
        }
        if "final_answer" in tool_names:
            return _tool_response("final_answer", payload)
        return {"content": json.dumps(payload), "tool_calls": [], "finish_reason": "stop"}


class OpenAICompatibleBackend:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str,
        temperature: float,
        top_p: float,
        timeout: float,
        retries: int,
        enable_thinking: bool,
        send_thinking_field: bool,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        self.retries = retries
        self.enable_thinking = enable_thinking
        self.send_thinking_field = send_thinking_field
        self.session = requests.Session()
        self.session.trust_env = False
        pool_connections = _env_int("HTTP_POOL_CONNECTIONS", 64)
        pool_maxsize = _env_int("HTTP_POOL_MAXSIZE", max(64, _env_int("DEFAULT_CONCURRENCY", 128)))
        adapter = HTTPAdapter(pool_connections=pool_connections, pool_maxsize=pool_maxsize, pool_block=False)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def complete_response(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"
        if self.send_thinking_field:
            payload["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}

        last_error = ""
        for attempt in range(self.retries + 1):
            try:
                response = self.session.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout,
                )
                if response.status_code >= 400 and "chat_template_kwargs" in payload:
                    retry_payload = dict(payload)
                    retry_payload.pop("chat_template_kwargs", None)
                    response = self.session.post(
                        f"{self.base_url}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        json=retry_payload,
                        timeout=self.timeout,
                    )
                response.raise_for_status()
                data = response.json()
                choice = data["choices"][0]
                message = choice.get("message") or {}
                return {
                    "content": message.get("content") or "",
                    "tool_calls": message.get("tool_calls") or [],
                    "finish_reason": choice.get("finish_reason"),
                    "raw_response": data,
                    "base_url": self.base_url,
                    "model": self.model,
                }
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                if attempt < self.retries:
                    time.sleep(min(2.0 * (attempt + 1), 6.0))
        raise RuntimeError(f"chat completion failed at {self.base_url}: {last_error}")


class RoundRobinBackend:
    def __init__(self, backends: list[OpenAICompatibleBackend]) -> None:
        if not backends:
            raise ValueError("RoundRobinBackend requires at least one backend")
        self.backends = backends
        self._lock = threading.Lock()
        self._index = 0

    def complete_response(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            backend = self.backends[self._index % len(self.backends)]
            self._index += 1
        return backend.complete_response(messages, max_tokens=max_tokens, tools=tools, tool_choice=tool_choice)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def build_backend_from_env() -> ChatBackend:
    backend = os.getenv("SKILL_RM_BACKEND", "openai").strip().lower()
    if backend == "mock":
        return MockBackend()
    if backend not in {"openai", "vllm"}:
        raise ValueError(f"Unsupported SKILL_RM_BACKEND={backend!r}")

    urls_env = os.getenv("MODEL_BASE_URLS")
    if urls_env:
        base_urls = [item.strip() for item in urls_env.split(",") if item.strip()]
    else:
        single = os.getenv("MODEL_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        base_urls = [single.strip()] if single else []
    if not base_urls:
        raise ValueError("Set MODEL_BASE_URL or MODEL_BASE_URLS for the reward judge, or SKILL_RM_BACKEND=mock for smoke tests.")

    model = os.getenv("MODEL_NAME", "Qwen3.5-27B")
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("MODEL_API_KEY") or "EMPTY"
    backends = [
        OpenAICompatibleBackend(
            base_url=url,
            model=model,
            api_key=api_key,
            temperature=_env_float("MODEL_TEMPERATURE", 0.0),
            top_p=_env_float("MODEL_TOP_P", 1.0),
            timeout=_env_float("MODEL_TIMEOUT", 300.0),
            retries=_env_int("MODEL_RETRIES", 2),
            enable_thinking=_env_flag("ENABLE_THINKING", False),
            send_thinking_field=_env_flag("SEND_THINKING_FIELD", True),
        )
        for url in base_urls
    ]
    return backends[0] if len(backends) == 1 else RoundRobinBackend(backends)
