from __future__ import annotations

import ast
import json
import re
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from .paths import DEFAULT_SKILLS_DIR


@dataclass(frozen=True)
class SkillMetadata:
    name: str
    description: str
    skill_path: Path


@dataclass(frozen=True)
class SkillContent:
    metadata: SkillMetadata
    instructions: str


class SkillLoader:
    def __init__(
        self,
        skill_paths: list[Path] | None = None,
        *,
        allowed_skill_names: Iterable[str] | None = None,
    ) -> None:
        self.skill_paths = skill_paths or [DEFAULT_SKILLS_DIR]
        self.allowed_skill_names = (
            {str(name).strip() for name in allowed_skill_names or [] if str(name).strip()}
            if allowed_skill_names is not None
            else None
        )
        self._metadata_cache: dict[str, SkillMetadata] = {}

    def scan_skills(self) -> list[SkillMetadata]:
        skills: list[SkillMetadata] = []
        seen: set[str] = set()
        for base_path in self.skill_paths:
            if not base_path.exists():
                continue
            for skill_dir in sorted(base_path.iterdir()):
                if not skill_dir.is_dir():
                    continue
                skill_md = skill_dir / "SKILL.md"
                if not skill_md.exists():
                    continue
                metadata = self._parse_skill_metadata(skill_md)
                if metadata and self.allowed_skill_names is not None and metadata.name not in self.allowed_skill_names:
                    continue
                if metadata and metadata.name not in seen:
                    skills.append(metadata)
                    seen.add(metadata.name)
                    self._metadata_cache[metadata.name] = metadata
        return skills

    def _parse_skill_metadata(self, skill_md_path: Path) -> SkillMetadata | None:
        content = skill_md_path.read_text(encoding="utf-8")
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, flags=re.S)
        if not match:
            return None
        frontmatter = yaml.safe_load(match.group(1)) or {}
        name = str(frontmatter.get("name") or "").strip()
        if not name:
            return None
        description = str(frontmatter.get("description") or "").strip()
        return SkillMetadata(name=name, description=" ".join(description.split()), skill_path=skill_md_path.parent)

    def load_skill(self, skill_name: str) -> SkillContent | None:
        if not self._metadata_cache:
            self.scan_skills()
        metadata = self._metadata_cache.get(skill_name)
        if not metadata:
            return None
        content = (metadata.skill_path / "SKILL.md").read_text(encoding="utf-8")
        body = re.sub(r"^---\s*\n.*?\n---\s*\n", "", content, flags=re.S).strip()
        return SkillContent(metadata=metadata, instructions=body)

    def build_system_prompt_section(self) -> str:
        skills = self.scan_skills()
        if not skills:
            return "## Skills\n\nNo specific verification skills currently available."
        lines = [
            "## Available Verification Skills",
            "",
            "You have access to the following specialized verification skills:",
            "",
        ]
        for skill in skills:
            lines.append(f"- **{skill.name}**: {skill.description}")
        lines.extend(
            [
                "",
                "### How to Use Skills",
                "",
                "1. **Discover**: Review the skills list above to find the appropriate verification method for the current task.",
                "2. **Load**: Use the `load_skill(skill_name)` tool to get detailed instructions for the chosen skill.",
                "3. **Execute**: Follow the skill's instructions carefully and use `execute_python` for exact visible-text checks.",
            ]
        )
        return "\n".join(lines)


def tool_schemas(enable_tools: bool = True) -> list[dict[str, Any]]:
    if not enable_tools:
        return []
    return [
        {
            "type": "function",
            "function": {
                "name": "load_skill",
                "description": "Load a skill's detailed instructions.",
                "parameters": {
                    "type": "object",
                    "properties": {"skill_name": {"type": "string"}},
                    "required": ["skill_name"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "execute_python",
                "description": "Execute short Python checks over visible IF-RewardBench sample fields.",
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_script",
                "description": "Inspect or run a Python script bundled with a loaded skill. Only use script names returned by load_skill.available_scripts; for instruction_following this is typically constraint_tools.py. Prefer execute_python for custom checks.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "skill_name": {"type": "string"},
                        "script_name": {"type": "string"},
                        "input_text": {"type": "string"},
                        "extra_args": {"type": "string"},
                    },
                    "required": ["skill_name", "script_name"],
                    "additionalProperties": False,
                },
            },
        },
    ]


class ToolExecutor:
    def __init__(self, *, skill_loader: SkillLoader, timeout: float = 10.0, max_output_chars: int = 3000) -> None:
        self.skill_loader = skill_loader
        self.timeout = timeout
        self.max_output_chars = max_output_chars

    def execute(self, tool_call: dict[str, Any], sample: dict[str, Any]) -> dict[str, Any]:
        name = tool_call_name(tool_call)
        args, error = parse_tool_call_arguments(tool_call)
        if error:
            return {"ok": False, "tool": name, "error": error}
        if name == "load_skill":
            return self.load_skill(str(args.get("skill_name") or ""))
        if name == "execute_python":
            return self.execute_python(str(args.get("code") or ""), sample)
        if name == "run_script":
            return self.run_script(
                skill_name=str(args.get("skill_name") or ""),
                script_name=str(args.get("script_name") or ""),
                input_text=str(args.get("input_text") or ""),
                extra_args=str(args.get("extra_args") or ""),
            )
        return {"ok": False, "tool": name, "error": f"unknown tool: {name}"}

    def load_skill(self, skill_name: str) -> dict[str, Any]:
        skill = self.skill_loader.load_skill(skill_name)
        if not skill:
            available = [item.name for item in self.skill_loader.scan_skills()]
            return {"ok": False, "tool": "load_skill", "error": f"skill not found: {skill_name}", "available": available}
        scripts_dir = skill.metadata.skill_path / "scripts"
        available_scripts = sorted(path.name for path in scripts_dir.glob("*.py")) if scripts_dir.exists() else []
        return {
            "ok": True,
            "tool": "load_skill",
            "skill_name": skill.metadata.name,
            "instructions": skill.instructions,
            "skill_path": str(skill.metadata.skill_path),
            "scripts_dir": str(scripts_dir),
            "available_scripts": available_scripts,
        }

    def run_script(self, *, skill_name: str, script_name: str, input_text: str, extra_args: str) -> dict[str, Any]:
        if "/" in script_name or "\\" in script_name or ".." in script_name:
            return {"ok": False, "tool": "run_script", "error": "script_name must be a plain filename"}
        skill = self.skill_loader.load_skill(skill_name)
        if not skill:
            return {"ok": False, "tool": "run_script", "error": f"skill not found: {skill_name}"}
        script_path = skill.metadata.skill_path / "scripts" / script_name
        if not script_path.exists():
            return {"ok": False, "tool": "run_script", "error": f"script not found: {script_name}"}
        if script_name == "constraint_tools.py" and not extra_args.strip():
            source = script_path.read_text(encoding="utf-8")
            return {"ok": True, "tool": "run_script", "script_name": script_name, "stdout": source[: self.max_output_chars]}
        cmd = [sys.executable, "-X", "utf8", str(script_path)]
        tmp_path: str | None = None
        try:
            if input_text:
                with tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8", delete=False) as handle:
                    handle.write(input_text)
                    tmp_path = handle.name
                cmd.extend(["--file", tmp_path])
            if extra_args.strip():
                cmd.extend(shlex.split(extra_args))
            completed = subprocess.run(cmd, text=True, capture_output=True, timeout=self.timeout, check=False)
            return {
                "ok": completed.returncode == 0,
                "tool": "run_script",
                "script_name": script_name,
                "returncode": completed.returncode,
                "stdout": completed.stdout[: self.max_output_chars],
                "stderr": completed.stderr[: self.max_output_chars],
            }
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "tool": "run_script", "script_name": script_name, "error": str(exc)}
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

    def execute_python(self, code: str, sample: dict[str, Any]) -> dict[str, Any]:
        if not code.strip():
            return {"ok": False, "tool": "execute_python", "error": "empty code"}
        validation_error = validate_python_code(code)
        if validation_error:
            return {"ok": False, "tool": "execute_python", "error": validation_error}
        helper_source = helper_source_from_skills(self.skill_loader)
        payload = {"code": code, "sample": visible_sample_payload(sample), "helper_source": helper_source}
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-S", "-c", PYTHON_WRAPPER],
                input=json.dumps(payload, ensure_ascii=False),
                text=True,
                capture_output=True,
                timeout=self.timeout,
                cwd=tempfile.gettempdir(),
                env={},
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "ok": False,
                "tool": "execute_python",
                "timeout": True,
                "latency_sec": round(time.perf_counter() - started, 4),
                "stdout": (exc.stdout or "")[: self.max_output_chars] if isinstance(exc.stdout, str) else "",
                "stderr": (exc.stderr or "")[: self.max_output_chars] if isinstance(exc.stderr, str) else "",
                "error": "execute_python timed out",
            }
        return {
            "ok": completed.returncode == 0,
            "tool": "execute_python",
            "timeout": False,
            "latency_sec": round(time.perf_counter() - started, 4),
            "returncode": completed.returncode,
            "stdout": completed.stdout[: self.max_output_chars],
            "stderr": completed.stderr[: self.max_output_chars],
            "error": None if completed.returncode == 0 else "execute_python returned non-zero",
        }


def helper_source_from_skills(loader: SkillLoader) -> str:
    for skill in loader.scan_skills():
        path = skill.skill_path / "scripts" / "constraint_tools.py"
        if path.exists():
            return path.read_text(encoding="utf-8")
    return ""


def visible_sample_payload(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "prompt": str(sample.get("prompt") or ""),
        "instruction": str(sample.get("prompt") or ""),
        "response": str(sample.get("response_a") or ""),
        "response_a": str(sample.get("response_a") or ""),
        "response_b": str(sample.get("response_b") or ""),
        "system_prompt": str(sample.get("system_prompt") or ""),
        "history": str(sample.get("history") or ""),
        "checklist": str(sample.get("checklist") or ""),
    }


ALLOWED_IMPORTS = {"collections", "decimal", "fractions", "functools", "itertools", "json", "math", "re", "statistics", "string", "unicodedata"}
FORBIDDEN_NAMES = {"__builtins__", "__import__", "breakpoint", "compile", "eval", "exec", "getattr", "globals", "help", "input", "locals", "open", "setattr", "vars"}


def validate_python_code(code: str) -> str | None:
    if "__" in code:
        return "dunder names are not allowed"
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        return f"syntax error: {exc}"
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            imported = [alias.name.split(".", 1)[0] for alias in getattr(node, "names", [])]
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module.split(".", 1)[0])
            blocked = sorted({name for name in imported if name not in ALLOWED_IMPORTS})
            if blocked:
                return f"import not allowed: {', '.join(blocked)}"
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return "dunder attributes are not allowed"
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            return f"name not allowed: {node.id}"
    return None


def tool_call_name(tool_call: dict[str, Any]) -> str:
    function = tool_call.get("function") if isinstance(tool_call, dict) else None
    if isinstance(function, dict):
        return str(function.get("name") or "")
    return str(tool_call.get("name") or "")


def parse_tool_call_arguments(tool_call: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    function = tool_call.get("function") if isinstance(tool_call, dict) else None
    raw = function.get("arguments") if isinstance(function, dict) else tool_call.get("arguments")
    if raw in (None, ""):
        return {}, None
    if isinstance(raw, dict):
        return raw, None
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError as exc:
        return {}, f"invalid tool JSON arguments: {exc}"
    if not isinstance(parsed, dict):
        return {}, "tool arguments must be a JSON object"
    return parsed, None


PYTHON_WRAPPER = r'''
import collections
import decimal
import fractions
import functools
import itertools
import json
import math
import re
import statistics
import string
import sys
import unicodedata

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
    "unicodedata": unicodedata,
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
    "unicodedata": unicodedata,
    "sample": sample,
    "prompt": sample["prompt"],
    "instruction": sample["instruction"],
    "response": sample["response"],
    "response_a": sample["response_a"],
    "response_b": sample["response_b"],
    "system_prompt": sample["system_prompt"],
    "history": sample["history"],
    "checklist": sample["checklist"],
}
helper_source = payload.get("helper_source") or ""
if helper_source:
    exec(compile(helper_source, "<constraint_tools>", "exec"), scope, scope)
exec(compile(payload["code"], "<execute_python>", "exec"), scope, scope)
if "result" in scope:
    print(json.dumps(scope["result"], ensure_ascii=False, sort_keys=True))
'''
