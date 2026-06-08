from __future__ import annotations

import hashlib
import random
from typing import Any


def stable_int(*parts: Any) -> int:
    text = "\x1f".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def seeded_random(*parts: Any) -> random.Random:
    return random.Random(stable_int(*parts))
