from __future__ import annotations

import json
import re
from typing import Any


def parse_first_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        return {}
    decoder = json.JSONDecoder()
    try:
        value, _ = decoder.raw_decode(stripped)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if fenced:
        try:
            value = json.loads(fenced.group(1))
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}
