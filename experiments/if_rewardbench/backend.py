from __future__ import annotations

import json
import re
import threading
import time
from typing import Any, Protocol

import requests


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


class MockBackend:
    """Deterministic local backend for quick plumbing checks."""

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
        if "load_skill" in tool_names and '"tool": "load_skill"' not in transcript:
            return tool_response("load_skill", {"skill_name": "instruction_following"})
        if "execute_python" in tool_names and "word_count(response" in transcript and '"tool": "execute_python"' not in transcript:
            return tool_response("execute_python", {"code": "print({'ok': True})"})
        if "[助手 A 的回答-开始]" in transcript:
            return {"content": "逐项比较后，助手 A 更好。\n[[A]]", "tool_calls": [], "finish_reason": "stop"}
        if "[检查项列表-开始]" in transcript:
            checklist = between(transcript, "[检查项列表-开始]", "[检查项列表-结束]")
            items = re.findall(r"\[检查项(\d+)-开始\]\s*(.*?)\s*\[检查项\1-结束\]", checklist, flags=re.S)
            if not items:
                items = [("1", "回复应遵循用户指令。")]
            blocks = []
            for idx, text in items:
                blocks.append(
                    f"[检查项{idx}-开始]\n"
                    f"要求：{text.strip()}\n"
                    f"分析：mock backend marks this requirement as satisfied for plumbing validation.\n"
                    f"结论：[[人工智能助手的回复满足了该要求]]\n"
                    f"[检查项{idx}-结束]"
                )
            return {"content": "\n\n".join(blocks), "tool_calls": [], "finish_reason": "stop"}
        return {
            "content": (
                "[检查项1-开始]\n"
                "要求：回复应遵循用户指令。\n"
                "分析：mock backend marks the response as acceptable for plumbing validation.\n"
                "结论：[[人工智能助手的回复满足了该要求]]\n"
                "[检查项1-结束]"
            ),
            "tool_calls": [],
            "finish_reason": "stop",
        }


def tool_response(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
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


def between(text: str, start: str, end: str) -> str:
    if start not in text:
        return ""
    tail = text.split(start, 1)[1]
    if end in tail:
        return tail.split(end, 1)[0]
    return tail


class OpenAICompatibleBackend:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        temperature: float = 0.0,
        top_p: float = 1.0,
        timeout: float = 300.0,
        retries: int = 2,
        send_thinking_field: bool = True,
        enable_thinking: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        self.retries = retries
        self.send_thinking_field = send_thinking_field
        self.enable_thinking = enable_thinking
        self.session = requests.Session()
        self.session.trust_env = False

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

        last_error: str | None = None
        for attempt in range(self.retries + 1):
            try:
                response = self.session.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    json=payload,
                    timeout=self.timeout,
                )
                if response.status_code >= 400 and "chat_template_kwargs" in payload:
                    retry_payload = dict(payload)
                    retry_payload.pop("chat_template_kwargs", None)
                    response = self.session.post(
                        f"{self.base_url}/chat/completions",
                        headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
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
                }
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                if attempt < self.retries:
                    time.sleep(min(2.0 * (attempt + 1), 6.0))
        raise RuntimeError(f"chat completion failed: {last_error}")


class RoundRobinBackend:
    def __init__(self, backends: list[OpenAICompatibleBackend], *, max_backend_attempts: int | None = None) -> None:
        if not backends:
            raise ValueError("RoundRobinBackend requires at least one backend")
        self.backends = backends
        self.max_backend_attempts = max_backend_attempts or min(4, len(backends))
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
        errors: list[str] = []
        for _attempt in range(max(1, min(self.max_backend_attempts, len(self.backends)))):
            with self._lock:
                backend = self.backends[self._index % len(self.backends)]
                self._index += 1
            try:
                return backend.complete_response(messages, max_tokens=max_tokens, tools=tools, tool_choice=tool_choice)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{backend.base_url}: {exc}")
        raise RuntimeError("all round-robin backend attempts failed: " + " | ".join(errors))


def build_backend(config: dict[str, Any]) -> ChatBackend:
    backend = str(config.get("backend", "mock")).lower()
    if backend == "mock":
        return MockBackend()
    if backend not in {"openai", "vllm"}:
        raise ValueError(f"unsupported backend: {backend}")

    base_urls = config.get("base_urls") or config.get("base_url")
    if not base_urls and config.get("endpoint_hosts"):
        ports = config.get("endpoint_ports") or [8000]
        base_urls = [
            f"http://{str(host).strip()}:{int(port)}/v1"
            for host in config.get("endpoint_hosts") or []
            for port in ports
            if str(host).strip()
        ]
    if isinstance(base_urls, list):
        urls = [str(item).strip() for item in base_urls if str(item).strip()]
    elif base_urls:
        urls = [str(base_urls).strip()]
    else:
        raise ValueError("OpenAI/vLLM backend requires base_url/base_urls or endpoint_hosts.")
    if any("$" in url or url.startswith("<") for url in urls):
        raise ValueError("OpenAI/vLLM backend base URL contains an unresolved placeholder.")

    backends = [
        OpenAICompatibleBackend(
            base_url=url,
            model=str(config.get("model", "Qwen3.5-27B")),
            api_key=str(config.get("api_key", "EMPTY")),
            temperature=float(config.get("temperature", 0.0)),
            top_p=float(config.get("top_p", 1.0)),
            timeout=float(config.get("timeout", 300)),
            retries=int(config.get("retries", 2)),
            send_thinking_field=bool(config.get("send_thinking_field", True)),
            enable_thinking=bool(config.get("enable_thinking", False)),
        )
        for url in urls
    ]
    if len(backends) == 1:
        return backends[0]
    return RoundRobinBackend(backends, max_backend_attempts=int(config.get("max_backend_attempts", min(4, len(backends)))))
