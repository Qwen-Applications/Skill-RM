from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
import time
from typing import Any


def run_python_sandbox_tool(
    args: dict[str, Any],
    record: dict[str, Any],
    formatted: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    code = str(args.get("code") or "")
    reason = str(args.get("reason") or "")
    if not bool(config.get("enable_python_sandbox", True)):
        return {"ok": False, "tool": "python_sandbox", "error": "python_sandbox disabled by config"}
    max_code_chars = int(config.get("python_sandbox_max_code_chars", 6000))
    if not code.strip():
        return {"ok": False, "tool": "python_sandbox", "reason": reason, "error": "empty code"}
    if len(code) > max_code_chars:
        return {
            "ok": False,
            "tool": "python_sandbox",
            "reason": reason,
            "error": f"code too long: {len(code)} > {max_code_chars}",
        }
    validation_error = validate_python_sandbox_code(code)
    if validation_error:
        return {"ok": False, "tool": "python_sandbox", "reason": reason, "error": validation_error}

    sample = {
        "prompt": str(record.get("prompt", "")),
        "candidates": dict(formatted.get("responses") or {}),
    }
    payload = {"code": code, "sample": sample}
    started = time.time()
    timeout = float(config.get("python_sandbox_timeout", 3.0))
    max_output_chars = int(config.get("python_sandbox_max_output_chars", 4000))
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-S", "-c", PYTHON_SANDBOX_WRAPPER],
            input=json.dumps(payload, ensure_ascii=False),
            text=True,
            capture_output=True,
            timeout=timeout,
            cwd=tempfile.gettempdir(),
            env={},
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
        return {
            "ok": False,
            "tool": "python_sandbox",
            "reason": reason,
            "timeout": True,
            "timeout_sec": timeout,
            "latency_sec": time.time() - started,
            "stdout": stdout[:max_output_chars],
            "stderr": stderr[:max_output_chars],
            "error": "python_sandbox timed out",
        }

    stdout = completed.stdout[:max_output_chars]
    stderr = completed.stderr[:max_output_chars]
    return {
        "ok": completed.returncode == 0,
        "tool": "python_sandbox",
        "reason": reason,
        "timeout": False,
        "latency_sec": time.time() - started,
        "returncode": completed.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "stdout_truncated": len(completed.stdout) > max_output_chars,
        "stderr_truncated": len(completed.stderr) > max_output_chars,
        "error": None if completed.returncode == 0 else "python_sandbox returned non-zero",
    }


PYTHON_SANDBOX_ALLOWED_IMPORTS = {
    "collections",
    "decimal",
    "fractions",
    "functools",
    "itertools",
    "json",
    "math",
    "re",
    "statistics",
    "string",
}
PYTHON_SANDBOX_FORBIDDEN_NAMES = {
    "__builtins__",
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
}


def validate_python_sandbox_code(code: str) -> str | None:
    if "__" in code:
        return "dunder names are not allowed in python_sandbox code"
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        return f"syntax error: {exc}"
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imported = [alias.name.split(".", 1)[0] for alias in getattr(node, "names", [])]
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module.split(".", 1)[0])
            blocked = sorted({name for name in imported if name not in PYTHON_SANDBOX_ALLOWED_IMPORTS})
            if blocked:
                return f"import not allowed: {', '.join(blocked)}"
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return "dunder attributes are not allowed"
        if isinstance(node, ast.Name) and node.id in PYTHON_SANDBOX_FORBIDDEN_NAMES:
            return f"name not allowed: {node.id}"
    return None


PYTHON_SANDBOX_WRAPPER = r'''
import collections
import contextlib
import decimal
import fractions
import functools
import io
import itertools
import json
import math
import re
import statistics
import string
import sys

ALLOWED_IMPORTS = {
    "collections": collections,
    "decimal": decimal,
    "fractions": fractions,
    "functools": functools,
    "itertools": itertools,
    "json": json,
    "math": math,
    "re": re,
    "statistics": statistics,
    "string": string,
}

def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".", 1)[0]
    if level != 0 or root not in ALLOWED_IMPORTS:
        raise ImportError(f"import not allowed: {name}")
    return ALLOWED_IMPORTS[root]

SAFE_BUILTINS = {
    "__import__": safe_import,
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "Exception": Exception,
    "filter": filter,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "pow": pow,
    "print": print,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "ValueError": ValueError,
    "zip": zip,
}

payload = json.loads(sys.stdin.read())
sample = payload["sample"]
scope = {
    "__builtins__": SAFE_BUILTINS,
    "collections": collections,
    "decimal": decimal,
    "fractions": fractions,
    "functools": functools,
    "itertools": itertools,
    "json": json,
    "math": math,
    "re": re,
    "statistics": statistics,
    "string": string,
    "sample": sample,
    "prompt": sample["prompt"],
    "candidates": sample["candidates"],
}
exec(compile(payload["code"], "<python_sandbox>", "exec"), scope, scope)
if "result" in scope:
    print(json.dumps(scope["result"], ensure_ascii=False, sort_keys=True))
'''

