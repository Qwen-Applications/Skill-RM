from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def iter_files(root: str | Path, suffixes: set[str]) -> Iterable[Path]:
    root = Path(root)
    if not root.exists():
        return
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in suffixes:
            yield path


def tree_fingerprint(root: str | Path, suffixes: set[str]) -> dict[str, Any]:
    root = Path(root)
    if not root.exists():
        return {"path": str(root), "exists": False}
    files = []
    digest = hashlib.sha256()
    for path in iter_files(root, suffixes):
        rel = str(path.relative_to(root))
        stat = path.stat()
        content_hash = file_sha256(path)
        files.append({"path": rel, "size_bytes": stat.st_size, "sha256": content_hash})
        digest.update(rel.encode("utf-8"))
        digest.update(content_hash.encode("utf-8"))
    return {"path": str(root), "exists": True, "file_count": len(files), "sha256": digest.hexdigest(), "files": files}
