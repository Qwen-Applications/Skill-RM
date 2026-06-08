#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import requests


def split_urls(raw: str) -> list[str]:
    return [item.strip().rstrip("/") for item in raw.split(",") if item.strip()]


def request_json(session: requests.Session, url: str, *, timeout: float) -> tuple[bool, Any]:
    try:
        response = session.get(url, timeout=timeout)
        response.raise_for_status()
        return True, response.json()
    except Exception as exc:  # noqa: BLE001
        return False, repr(exc)


def check_chat(session: requests.Session, base_url: str, *, model: str, timeout: float) -> tuple[bool, Any]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Reply with exactly OK."},
            {"role": "user", "content": "Ping"},
        ],
        "temperature": 0,
        "max_tokens": 8,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        response = session.post(
            f"{base_url}/chat/completions",
            json=payload,
            headers={"Authorization": "Bearer EMPTY", "Content-Type": "application/json"},
            timeout=timeout,
        )
        if response.status_code >= 400:
            payload.pop("chat_template_kwargs", None)
            response = session.post(
                f"{base_url}/chat/completions",
                json=payload,
                headers={"Authorization": "Bearer EMPTY", "Content-Type": "application/json"},
                timeout=timeout,
            )
        response.raise_for_status()
        data = response.json()
        content = ((data.get("choices") or [{}])[0].get("message") or {}).get("content")
        return True, content
    except Exception as exc:  # noqa: BLE001
        return False, repr(exc)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check OpenAI-compatible Skill-RM endpoints.")
    parser.add_argument("--base-urls", default=os.environ.get("SKILLRM_BASE_URLS", ""))
    parser.add_argument("--model", default=os.environ.get("SKILLRM_MODEL", "Qwen3.5-27B"))
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--chat", action="store_true", help="Also send one tiny chat completion request.")
    parser.add_argument("--trust-env", action="store_true", help="Allow requests to use proxy settings from the environment.")
    args = parser.parse_args()

    urls = split_urls(args.base_urls)
    if not urls:
        print("No endpoints configured. Set SKILLRM_BASE_URLS or pass --base-urls.", file=sys.stderr)
        return 2

    session = requests.Session()
    session.trust_env = bool(args.trust_env)
    failures = 0
    for base_url in urls:
        ok, payload = request_json(session, f"{base_url}/models", timeout=args.timeout)
        status = "ok" if ok else "failed"
        print(f"{base_url}/models: {status} {payload if not ok else ''}")
        if not ok:
            failures += 1
            continue
        if args.chat:
            chat_ok, chat_payload = check_chat(session, base_url, model=args.model, timeout=args.timeout)
            print(f"{base_url}/chat/completions: {'ok' if chat_ok else 'failed'} {chat_payload}")
            failures += 0 if chat_ok else 1
    print(f"checked={len(urls)} failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

