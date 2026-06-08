"""Resource router for the pointwise IF Skill-RM harness."""

from __future__ import annotations

import contextlib
import ast
import hashlib
import io
import json
import math
import re
import statistics
import string
import subprocess
import sys
import tempfile
import textwrap
import time
import unicodedata
from pathlib import Path
from typing import Any

import yaml

from .sample_resources import ChecklistItem, SampleVerificationResources
from .types import PointwiseSample
from .utils import normalize_key, safe_text


class ResourceRouter:
    def __init__(
        self,
        *,
        skill_dir: str | Path,
        variant_config: dict[str, Any],
    ) -> None:
        self.skill_dir = Path(skill_dir)
        self.variant_config = variant_config
        self.skill_md = (self.skill_dir / "SKILL.md").read_text(encoding="utf-8")
        resources_path = self.skill_dir / "resources.yaml"
        self.resources = yaml.safe_load(resources_path.read_text(encoding="utf-8")) if resources_path.exists() else {}

    def skill_name(self) -> str:
        match = re.search(r"(?im)^name:\s*([A-Za-z0-9_.-]+)\s*$", self.skill_md)
        return match.group(1) if match else self.skill_dir.name

    def skill_description(self) -> str:
        match = re.search(r"(?ims)^description:\s*(.+?)(?:\n[a-zA-Z_-]+:|\n---)", self.skill_md)
        if match:
            return " ".join(match.group(1).replace("|", " ").split())
        return "Pointwise instruction-following reward skill."

    def build_system_prompt(self, sample: PointwiseSample) -> str:
        del sample
        max_steps = int(self.variant_config.get("max_agent_steps", 5))
        max_resources = int(self.variant_config.get("max_resources_per_sample", 4))
        lines = [
            "You are an impartial pointwise judge for one visible instruction-following sample.",
            "Score one assistant response against the visible instruction, system prompt, and conversation history only.",
            "Never use hidden labels, chosen/rejected origin, benchmark gold labels, or dataset construction artifacts.",
            "A score of 1.0 means the response fully satisfies all mandatory instructions and is useful.",
            "A score of 0.0 means the response fails the main task, refuses incorrectly, or violates critical constraints.",
            "Use intermediate values for partial instruction following. Prefer verified exact constraints over style impressions.",
            "You may optionally load an external judging skill through tool calls. The skill is not loaded by default.",
            "If skill instructions, sample resources, or deterministic checks may materially improve the score, call use_skill.",
            f"Use at most one skill load, at most {max_resources} viewed resources, and at most {max_steps} assistant turns.",
            "When ready, use the final_answer tool rather than prose. If you answer in text, return exactly one JSON object and no extra prose.",
            "Final answer contract:",
            '{"score":0.0,"satisfied_count":0,"total_count":0,"confidence":0.0,"used_resources":[],"reason":"short reason"}',
        ]
        if bool(self.variant_config.get("enable_python_sandbox", False)):
            lines.extend(
                [
                    "",
                    "If the skill is loaded, a python_sandbox tool may be available for short deterministic checks over only the visible prompt/response.",
                    "Use it for count, regex, JSON, Markdown/list structure, required/forbidden term, and simple arithmetic constraints.",
                ]
            )
        lines.extend(
            [
                "",
                "Available optional skill:",
                json.dumps(
                    {
                        "name": self.skill_name(),
                        "description": self.skill_description(),
                        "loading": "self-select; SKILL.md and resource index are hidden until use_skill is called",
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            ]
        )
        return "\n".join(lines)

    def build_user_prompt(self, sample: PointwiseSample, *, prompt_resources_section: str = "") -> str:
        lines = ["Score this instruction-following response."]
        if sample.system_prompt:
            lines.extend(["", "[System Prompt]", safe_text(sample.system_prompt, int(self.variant_config.get("max_prompt_chars", 8000)))])
        if sample.history:
            lines.extend(["", "[Conversation History]", safe_text(sample.history, int(self.variant_config.get("max_prompt_chars", 8000)))])
        lines.extend(
            [
                "",
                "[User Prompt]",
                safe_text(sample.prompt, int(self.variant_config.get("max_prompt_chars", 8000))),
            ]
        )
        if prompt_resources_section:
            lines.extend(["", prompt_resources_section])
        lines.extend(
            [
                "",
                "[Response]",
                safe_text(sample.response, int(self.variant_config.get("max_response_chars", 12000))),
                "",
                "Return a pointwise scalar reward in [0, 1]. If a checklist is available, use satisfied_count/total_count as the primary scoring basis and adjust only for clearly important quality or safety failures.",
            ]
        )
        return "\n".join(lines)

    def tools(self, *, skill_loaded: bool) -> list[dict[str, Any]]:
        final_answer = {
            "type": "function",
            "function": {
                "name": "final_answer",
                "description": "Submit the final pointwise reward score.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "satisfied_count": {"type": ["integer", "null"]},
                        "total_count": {"type": ["integer", "null"]},
                        "confidence": {"type": "number"},
                        "used_resources": {"type": "array", "items": {"type": "string"}},
                        "reason": {"type": "string"},
                        "satisfied_constraints": {"type": "array", "items": {"type": "string"}},
                        "failed_constraints": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["score"],
                    "additionalProperties": True,
                },
            },
        }
        if not skill_loaded:
            return [
                {
                    "type": "function",
                    "function": {
                        "name": "use_skill",
                        "description": "Load the instruction-following pointwise scoring skill when checklist decomposition, constraint verification, or mounted resources can improve the score.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "skill_name": {"type": "string"},
                                "reason": {"type": "string"},
                            },
                            "required": ["reason"],
                            "additionalProperties": False,
                        },
                    },
                },
                final_answer,
            ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "list_resources",
                    "description": "List resources available through the loaded skill for this sample.",
                    "parameters": {
                        "type": "object",
                        "properties": {"type": {"type": ["string", "null"]}},
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "view_resource",
                    "description": "Read one resource by resource_id. Use only resources needed for this sample.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "resource_id": {"type": "string"},
                            "path": {"type": "string"},
                            "reason": {"type": "string"},
                        },
                        "required": ["reason"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_resource",
                    "description": "Run a mounted runtime verifier resource. Only use resources whose implementation_kind is runtime_verifier.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "resource_id": {"type": "string"},
                            "reason": {"type": "string"},
                        },
                        "required": ["resource_id", "reason"],
                        "additionalProperties": False,
                    },
                },
            },
        ]
        if bool(self.variant_config.get("enable_python_sandbox", False)):
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": "python_sandbox",
                        "description": (
                            "Run small deterministic Python checks over the visible prompt and candidate response. "
                            "Use for exact counts, regex checks, JSON/list/Markdown structure, required or forbidden terms, "
                            "quote/bracket balance, and simple arithmetic. No network, file access, subprocesses, or hidden metadata are available."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "code": {
                                    "type": "string",
                                    "description": (
                                        "Python code. Available variables: prompt/instruction (str), response (str), "
                                        "system_prompt (str), history (str), sample (dict). Helper functions from "
                                        "if.constraint_tools_source are preloaded. Print compact JSON or set result = {...}."
                                    ),
                                },
                                "reason": {"type": "string", "description": "Which deterministic constraint this checks."},
                                "constraint_id": {"type": "string", "description": "Optional checklist item or short constraint id."},
                            },
                            "required": ["code", "reason"],
                            "additionalProperties": False,
                        },
                    },
                }
            )
        tools.append(final_answer)
        return tools

    def execute_tool_call(
        self,
        tool_call: dict[str, Any],
        *,
        sample: PointwiseSample,
        skill_state: dict[str, Any],
        resources_viewed: list[str],
        resources_run: list[str],
        step: int,
    ) -> dict[str, Any]:
        name = tool_call_name(tool_call)
        args, error = parse_tool_call_arguments(tool_call)
        if error:
            return {"ok": False, "tool": name, "error": error}
        if name == "use_skill":
            if skill_state.get("loaded"):
                return {"ok": True, "tool": name, "already_loaded": True}
            requested = str(args.get("skill_name") or self.skill_name())
            if requested not in {self.skill_name(), "instruction_following_pointwise", "skill-rm-if-pointwise"}:
                return {"ok": False, "tool": name, "error": f"unknown skill: {requested}", "available_skill": self.skill_name()}
            skill_state["loaded"] = True
            skill_state["trigger_step"] = step
            skill_state["trigger_reason"] = str(args.get("reason") or "")
            index = self.build_resource_index(sample)
            return {
                "ok": True,
                "tool": name,
                "skill_name": self.skill_name(),
                "skill_controller": self.skill_md.strip(),
                "resource_index": index,
                "recommended_resources": self.recommended_resources(sample, index),
                "instructions": (
                    "Use resources as visible evidence. For exact visible-text constraints, prefer mounted verifiers "
                    "or python_sandbox when available. Return a calibrated scalar score in [0, 1]."
                ),
            }
        if not skill_state.get("loaded"):
            return {"ok": False, "tool": name, "error": "skill is not loaded; call use_skill first"}
        if name == "list_resources":
            resource_type = args.get("type")
            resources = self.build_resource_index(sample)
            if resource_type:
                resources = [item for item in resources if item.get("type") == resource_type]
            return {"ok": True, "tool": name, "resources": resources}
        if name == "view_resource":
            resource_id = str(args.get("resource_id") or args.get("path") or "")
            return self.view_resource(resource_id, sample=sample, resources_viewed=resources_viewed)
        if name == "run_resource":
            return self.run_resource(str(args.get("resource_id") or ""), sample=sample, resources_run=resources_run)
        if name == "python_sandbox":
            return self.run_python_sandbox(args, sample=sample, skill_state=skill_state, resources_run=resources_run)
        return {"ok": False, "tool": name, "error": f"unknown tool: {name}"}

    def build_resource_index(self, sample: PointwiseSample) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        if bool(self.variant_config.get("mount_sample_resources", False)):
            items.extend(sample_resource_index(sample.sample_resources))
        if bool(self.variant_config.get("enable_python_sandbox", False)):
            items.append(
                {
                    "id": "if.python_sandbox",
                    "type": "tool",
                    "implementation_kind": "runtime_python_sandbox",
                    "subtype": "visible_text_constraint_checker",
                    "hard_or_soft": "hard",
                    "decision_impact": "evidence",
                    "leakage_level": "sample_visible",
                    "cost": "medium",
                    "inputs_required": ["prompt", "candidate_response"],
                    "outputs_produced": ["deterministic_constraint_evidence"],
                    "usage_note": "Use the python_sandbox tool, not run_resource, for short exact checks over visible text.",
                }
            )
        if bool(self.variant_config.get("include_static_resources", False)):
            for entry in self.resources.get("resources") or []:
                items.append(resource_index_item(entry))
        return items

    def recommended_resources(self, sample: PointwiseSample, index: list[dict[str, Any]]) -> list[str]:
        del sample
        ids = [str(item.get("id") or "") for item in index]
        preferred = [
            "sample.verinstruct.checklist",
            "sample.verinstruct.verify_all",
            "if.constraint_toolkit",
            "if.python_sandbox",
            "if.constraint_verification_protocol",
            "if.pointwise_rubric",
            "if.aggregation_policy",
        ]
        return [item for item in preferred if item in ids]

    def view_resource(self, resource_id: str, *, sample: PointwiseSample, resources_viewed: list[str]) -> dict[str, Any]:
        resource_id = normalize_resource_id(resource_id)
        max_resources = int(self.variant_config.get("max_resources_per_sample", 4))
        if resource_id not in resources_viewed and len(set(resources_viewed)) >= max_resources:
            return {"ok": False, "tool": "view_resource", "resource_id": resource_id, "error": f"max resources exceeded: {max_resources}"}
        content = self._resource_content(resource_id, sample=sample)
        if content is None:
            return {"ok": False, "tool": "view_resource", "resource_id": resource_id, "error": "resource not found or not visible"}
        resources_viewed.append(resource_id)
        return {
            "ok": True,
            "tool": "view_resource",
            "resource_id": resource_id,
            "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "truncated": len(content) > int(self.variant_config.get("max_resource_chars", 8000)),
            "content": safe_text(content, int(self.variant_config.get("max_resource_chars", 8000))),
        }

    def run_resource(self, resource_id: str, *, sample: PointwiseSample, resources_run: list[str]) -> dict[str, Any]:
        resource_id = normalize_resource_id(resource_id)
        visible = {normalize_resource_id(str(item.get("id") or "")) for item in self.build_resource_index(sample)}
        if resource_id not in visible:
            return {"ok": False, "tool": "run_resource", "resource_id": resource_id, "error": "resource is not visible"}
        if resource_id == "sample.verinstruct.verify_all":
            resources_run.append(resource_id)
            return {
                "ok": True,
                "tool": "run_resource",
                "resource_id": resource_id,
                "result": run_all_sample_verifiers(sample),
            }
        if resource_id.startswith("sample.verinstruct.verifier."):
            item_id = resource_id.rsplit(".", 1)[-1]
            item = find_sample_item(sample.sample_resources, item_id)
            if not item or not item.verifier_code:
                return {"ok": False, "tool": "run_resource", "resource_id": resource_id, "error": "verifier code unavailable"}
            resources_run.append(resource_id)
            return {
                "ok": True,
                "tool": "run_resource",
                "resource_id": resource_id,
                "result": run_one_sample_verifier(sample, item),
            }
        return {"ok": False, "tool": "run_resource", "resource_id": resource_id, "error": "resource is not runnable"}

    def _resource_content(self, resource_id: str, *, sample: PointwiseSample) -> str | None:
        if resource_id == "sample.verinstruct.checklist":
            resources = sample.sample_resources
            if not resources or not getattr(resources, "checklist_items", None):
                return None
            lines = ["# VerInstruct Sample Checklist", ""]
            for item in resources.checklist_items:
                lines.append(f"- {item.item_id} [{item.tag}]: {item.text}")
            return "\n".join(lines)
        if resource_id.startswith("sample.verinstruct.verifier."):
            item_id = resource_id.rsplit(".", 1)[-1]
            item = find_sample_item(sample.sample_resources, item_id)
            if item is None or not item.verifier_code:
                return None
            title = item.verifier_name or item.item_id
            return f"# {title}\n\nConstraint: {item.text}\n\n```python\n{item.verifier_code}\n```"
        if resource_id == "sample.verinstruct.verify_all":
            return "# Run all VerInstruct rule verifiers\n\nThis runtime resource executes visible sample verifier code against the candidate response."
        if resource_id == "if.python_sandbox":
            return (
                "# Python Sandbox Runtime Tool\n\n"
                "Call the `python_sandbox` tool directly. It receives only visible sample fields: "
                "`prompt`, `instruction`, `response`, `system_prompt`, `history`, and `sample`. "
                "Helper functions from `if.constraint_tools_source` are already loaded."
            )
        for entry in self.resources.get("resources") or []:
            if normalize_resource_id(str(entry.get("id") or "")) != resource_id:
                continue
            path = entry.get("path")
            if not path:
                return None
            full_path = (self.skill_dir / str(path)).resolve()
            if self.skill_dir.resolve() not in full_path.parents and full_path != self.skill_dir.resolve():
                return None
            if not full_path.exists() or not full_path.is_file():
                return None
            return full_path.read_text(encoding="utf-8")
        return None

    def run_python_sandbox(
        self,
        args: dict[str, Any],
        *,
        sample: PointwiseSample,
        skill_state: dict[str, Any],
        resources_run: list[str],
    ) -> dict[str, Any]:
        if not bool(self.variant_config.get("enable_python_sandbox", False)):
            return {"ok": False, "tool": "python_sandbox", "error": "python_sandbox disabled by config"}
        calls = int(skill_state.get("python_sandbox_call_count") or 0)
        max_calls = int(self.variant_config.get("max_python_sandbox_calls", 3))
        if calls >= max_calls:
            return {"ok": False, "tool": "python_sandbox", "error": f"max python_sandbox calls exceeded: {max_calls}"}
        skill_state["python_sandbox_call_count"] = calls + 1
        if "if.python_sandbox" not in resources_run:
            resources_run.append("if.python_sandbox")

        code = str(args.get("code") or "")
        reason = str(args.get("reason") or "")
        constraint_id = str(args.get("constraint_id") or "")
        max_code_chars = int(self.variant_config.get("python_sandbox_max_code_chars", 6000))
        if not code.strip():
            return {"ok": False, "tool": "python_sandbox", "reason": reason, "constraint_id": constraint_id, "error": "empty code"}
        if len(code) > max_code_chars:
            return {
                "ok": False,
                "tool": "python_sandbox",
                "reason": reason,
                "constraint_id": constraint_id,
                "error": f"code too long: {len(code)} > {max_code_chars}",
            }
        validation_error = validate_python_sandbox_code(code)
        if validation_error:
            return {"ok": False, "tool": "python_sandbox", "reason": reason, "constraint_id": constraint_id, "error": validation_error}

        visible_sample = {
            "prompt": sample.prompt,
            "instruction": sample.prompt,
            "response": sample.response,
            "system_prompt": sample.system_prompt,
            "history": sample.history,
        }
        payload = {
            "code": code,
            "sample": visible_sample,
            "helper_source": self._constraint_tools_source(),
        }
        timeout = float(self.variant_config.get("python_sandbox_timeout", 3.0))
        max_output_chars = int(self.variant_config.get("python_sandbox_max_output_chars", 4000))
        started = time.perf_counter()
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
            stdout = exc.stdout if isinstance(exc.stdout, str) else ""
            stderr = exc.stderr if isinstance(exc.stderr, str) else ""
            return {
                "ok": False,
                "tool": "python_sandbox",
                "reason": reason,
                "constraint_id": constraint_id,
                "timeout": True,
                "timeout_sec": timeout,
                "latency_sec": round(time.perf_counter() - started, 4),
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
            "constraint_id": constraint_id,
            "timeout": False,
            "latency_sec": round(time.perf_counter() - started, 4),
            "returncode": completed.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "stdout_truncated": len(completed.stdout) > max_output_chars,
            "stderr_truncated": len(completed.stderr) > max_output_chars,
            "error": None if completed.returncode == 0 else "python_sandbox returned non-zero",
        }

    def _constraint_tools_source(self) -> str:
        path = self.skill_dir / "scripts" / "constraint_tools.py"
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")


def sample_resource_index(resources: SampleVerificationResources | None) -> list[dict[str, Any]]:
    if resources is None or not resources.checklist_items:
        return []
    items: list[dict[str, Any]] = [
        {
            "id": "sample.verinstruct.checklist",
            "type": "checklist",
            "implementation_kind": "reference",
            "subtype": "verinstruct_sample_constraints",
            "hard_or_soft": "mixed",
            "decision_impact": "strong",
            "leakage_level": "sample_visible",
            "cost": "low",
            "outputs_produced": ["visible_constraint_checklist"],
        }
    ]
    runnable = False
    for item in resources.checklist_items:
        if item.verifier_code:
            runnable = True
            items.append(
                {
                    "id": f"sample.verinstruct.verifier.{normalize_resource_id(item.item_id)}",
                    "type": "verifier",
                    "implementation_kind": "runtime_verifier",
                    "subtype": "sample_rule_verifier",
                    "hard_or_soft": "hard",
                    "decision_impact": "evidence",
                    "leakage_level": "sample_visible",
                    "cost": "medium",
                    "inputs_required": ["prompt", "candidate_response"],
                    "outputs_produced": ["constraint_satisfaction_signal"],
                    "usage_note": f"Visible verifier for checklist item {item.item_id}: {item.text}",
                }
            )
    if runnable:
        items.insert(
            1,
            {
                "id": "sample.verinstruct.verify_all",
                "type": "verifier",
                "implementation_kind": "runtime_verifier",
                "subtype": "sample_rule_verifier_batch",
                "hard_or_soft": "hard",
                "decision_impact": "strong",
                "leakage_level": "sample_visible",
                "cost": "medium",
                "inputs_required": ["prompt", "candidate_response"],
                "outputs_produced": ["per_constraint_rule_results", "aggregate_rule_score"],
            },
        )
    return items


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
    "unicodedata",
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
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)

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
    "chr": chr,
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
    "ord": ord,
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
    "system_prompt": sample["system_prompt"],
    "history": sample["history"],
}
helper_source = payload.get("helper_source") or ""
if helper_source:
    exec(compile(helper_source, "<constraint_tools>", "exec"), scope, scope)
exec(compile(payload["code"], "<python_sandbox>", "exec"), scope, scope)
if "result" in scope:
    print(json.dumps(scope["result"], ensure_ascii=False, sort_keys=True))
'''


def resource_index_item(entry: dict[str, Any]) -> dict[str, Any]:
    keep = {
        "id",
        "type",
        "implementation_kind",
        "subtype",
        "hard_or_soft",
        "decision_impact",
        "leakage_level",
        "cost",
        "inputs_required",
        "outputs_produced",
        "usage_note",
    }
    return {key: entry[key] for key in keep if key in entry}


def find_sample_item(resources: SampleVerificationResources | None, item_id: str) -> ChecklistItem | None:
    if resources is None:
        return None
    normalized = normalize_resource_id(item_id)
    for item in resources.checklist_items:
        if normalize_resource_id(item.item_id) == normalized:
            return item
    return None


def run_all_sample_verifiers(sample: PointwiseSample) -> dict[str, Any]:
    resources = sample.sample_resources
    if resources is None:
        return {"verdict": "inconclusive", "reason": "no sample resources"}
    started = time.perf_counter()
    results = [run_one_sample_verifier(sample, item) for item in resources.checklist_items if item.verifier_code]
    decisive = [item for item in results if item.get("satisfied") is not None]
    satisfied = sum(1 for item in decisive if item.get("satisfied") is True)
    total = len(decisive)
    aggregate_score = None if total == 0 else satisfied / total
    return {
        "verdict": "ok" if total else "inconclusive",
        "satisfied_count": satisfied if total else None,
        "total_count": total if total else None,
        "aggregate_rule_score": aggregate_score,
        "per_item": results,
        "latency_sec": round(time.perf_counter() - started, 4),
        "usage_note": "Rule verifier coverage may be partial; score remaining LLM/semantic constraints separately.",
    }


def run_one_sample_verifier(sample: PointwiseSample, item: ChecklistItem) -> dict[str, Any]:
    raw = execute_verifier_code(
        item.verifier_code,
        prompt=sample.prompt,
        response=sample.response,
        verifier_name=item.verifier_name,
    )
    satisfied = verifier_result_to_bool(raw.get("result"))
    return {
        "item_id": item.item_id,
        "constraint": item.text,
        "verifier_name": item.verifier_name,
        "satisfied": satisfied,
        "raw_result": raw.get("result"),
        "stdout": raw.get("stdout", ""),
        "error": raw.get("error", ""),
    }


def execute_verifier_code(code: str, *, prompt: str, response: str, verifier_name: str = "") -> dict[str, Any]:
    stdout = io.StringIO()
    namespace: dict[str, Any] = {
        "__builtins__": _safe_builtins(),
        "json": json,
        "re": re,
        "math": math,
        "string": string,
        "statistics": statistics,
        "textwrap": textwrap,
        "unicodedata": unicodedata,
    }
    try:
        with contextlib.redirect_stdout(stdout):
            exec(code, namespace, namespace)  # noqa: S102 - trusted visible verifier code with restricted builtins.
            fn = choose_verifier_callable(namespace, verifier_name)
            if fn is None:
                return {"result": None, "stdout": stdout.getvalue(), "error": "no callable verifier found"}
            result = call_verifier(fn, prompt=prompt, response=response)
        return {"result": result, "stdout": stdout.getvalue(), "error": ""}
    except Exception as exc:  # noqa: BLE001
        return {"result": None, "stdout": stdout.getvalue(), "error": str(exc)}


def choose_verifier_callable(namespace: dict[str, Any], verifier_name: str) -> Any | None:
    callables = {name: value for name, value in namespace.items() if callable(value) and not name.startswith("_")}
    if not callables:
        return None
    normalized_name = normalize_key(verifier_name)
    if normalized_name:
        for name, value in callables.items():
            if normalize_key(name) == normalized_name:
                return value
    for preferred in ("check", "verify", "judge", "evaluate", "verifier", "checker"):
        if preferred in callables:
            return callables[preferred]
    return next(iter(callables.values()))


def call_verifier(fn: Any, *, prompt: str, response: str) -> Any:
    attempts = [
        lambda: fn(response),
        lambda: fn(prompt, response),
        lambda: fn({"prompt": prompt, "instruction": prompt, "response": response, "output": response}),
        lambda: fn(response=response),
        lambda: fn(output=response),
        lambda: fn(text=response),
        lambda: fn(prompt=prompt, response=response),
        lambda: fn(instruction=prompt, output=response),
    ]
    last_error: Exception | None = None
    for attempt in attempts:
        try:
            return attempt()
        except TypeError as exc:
            last_error = exc
    if last_error:
        raise last_error
    return None


def verifier_result_to_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, dict):
        for key in ("satisfied", "passed", "pass", "ok", "valid", "success", "result"):
            if key in value:
                return verifier_result_to_bool(value[key])
        score = value.get("score")
        if isinstance(score, (int, float)):
            return score >= 0.5
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes", "pass", "passed", "satisfied", "ok", "valid"}:
            return True
        if text in {"false", "no", "fail", "failed", "unsatisfied", "invalid"}:
            return False
    return None


def _safe_builtins() -> dict[str, Any]:
    allowed_imports = {"json", "re", "math", "string", "collections", "itertools", "functools", "textwrap", "unicodedata", "statistics"}

    def safe_import(name: str, globals: Any = None, locals: Any = None, fromlist: tuple[str, ...] = (), level: int = 0) -> Any:
        del globals, locals, fromlist, level
        root = name.split(".", 1)[0]
        if root not in allowed_imports:
            raise ImportError(f"import not allowed: {name}")
        return __import__(name)

    names = [
        "abs",
        "all",
        "any",
        "bool",
        "chr",
        "dict",
        "enumerate",
        "Exception",
        "filter",
        "float",
        "getattr",
        "hasattr",
        "int",
        "isinstance",
        "len",
        "list",
        "map",
        "max",
        "min",
        "ord",
        "range",
        "reversed",
        "round",
        "set",
        "sorted",
        "str",
        "sum",
        "tuple",
        "ValueError",
        "zip",
        "print",
    ]
    builtins = {name: __builtins__[name] if isinstance(__builtins__, dict) else getattr(__builtins__, name) for name in names}
    builtins["__import__"] = safe_import
    return builtins


def normalize_resource_id(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def parse_tool_call_arguments(tool_call: dict[str, Any]) -> tuple[dict[str, Any], str]:
    function = tool_call.get("function") or {}
    args = function.get("arguments")
    if args is None:
        return {}, ""
    if isinstance(args, dict):
        return args, ""
    if not isinstance(args, str):
        return {}, f"tool arguments must be JSON object, got {type(args).__name__}"
    try:
        parsed = json.loads(args or "{}")
    except json.JSONDecodeError as exc:
        return {}, f"invalid tool arguments JSON: {exc}"
    if not isinstance(parsed, dict):
        return {}, "tool arguments JSON must be an object"
    return parsed, ""


def tool_call_name(tool_call: dict[str, Any]) -> str:
    function = tool_call.get("function") or {}
    return str(function.get("name") or tool_call.get("name") or "")


def first_tool_call(tool_calls: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for call in tool_calls:
        if tool_call_name(call) == name:
            return call
    return None


def compact_tool_calls_for_trace(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for call in tool_calls:
        args, _ = parse_tool_call_arguments(call)
        compact.append({"name": tool_call_name(call), "arguments": args})
    return compact


def compact_tool_result_for_trace(result: dict[str, Any]) -> dict[str, Any]:
    output = dict(result)
    if "content" in output:
        output["content"] = safe_text(output["content"], 500)
    if "skill_controller" in output:
        output["skill_controller"] = safe_text(output["skill_controller"], 500)
    return output
