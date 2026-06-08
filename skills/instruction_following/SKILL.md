---
name: instruction_following
description: |
  Use this skill when judging instruction following for IF-RewardBench-style samples. It is especially useful when the sample contains exact constraints, a visible checklist, format requirements, word/count limits, language restrictions, or multiple sub-instructions that should be decomposed before the final benchmark judgment.
metadata:
  family: skill_rm_if
  method: instruction_following_reward
---

# Instruction-Following Judge

You are judging visible IF-RewardBench-style samples. Your job is to verify instruction-following constraints and support the final benchmark output for the current mode.

Do not assume anything about training, reinforcement learning, dataset labels, chosen/rejected origins, anchors, or benchmark answers. Use only the visible instruction, response, checklist, and outputs from the exposed tools.

## Available Harness Tools

Use tools only when they can change the final constraint judgment or pairwise overall choice.

- If a visible `checklist` is provided in the prompt or sample payload, read it before the final judgment. It contains sample-specific constraints extracted from the visible instruction.
- Use the `execute_python` tool for deterministic checks that would be error-prone by inspection: counts, regex, JSON validity, bullet/list structure, required/forbidden terms, exact prefix/suffix, delimiter counts, quote/bracket balance, or arithmetic.
- `execute_python` receives only visible sample fields: `prompt`, `instruction`, `response`, `response_a`, `response_b`, `system_prompt`, `history`, `checklist`, and `sample`.
- Helper functions from `scripts/constraint_tools.py` are already available inside `execute_python`; call them directly for common exact checks.
- If you need to inspect the helper source, call `run_script` for `constraint_tools.py`.
- If no sample checklist exists, decompose the active instruction directly using the judgment procedure below.

Do not use tools by default. The best path is usually: checklist or decomposition, exact verification when needed, then final-mode output.

## Judgment Procedure

1. Identify active instruction sources: system prompt, conversation history, and the current user prompt.
2. Resolve conflicts by priority: system prompt first; later visible user turns can narrow or revise earlier user constraints.
3. Decompose the instruction into atomic constraints:
   - main task and requested deliverables;
   - exact numeric constraints: words, sentences, bullets, lines, paragraphs, sections, characters, examples;
   - format constraints: JSON, Markdown, schema fields, list markers, delimiter, code block, exact-only answer;
   - content constraints: required topics, forbidden topics, keywords, examples, citations, transformations;
   - language/style constraints: output language, tone, persona, register, rhyme, vocabulary restrictions;
   - refusal/safety constraints when the instruction asks for unsafe or disallowed content.
4. Verify hard constraints first. Use `execute_python` when the answer depends on exact visible text properties.
5. Judge semantic completeness and usefulness after hard constraints. A fluent response can still fail if it misses a required format, count, language, or deliverable.
6. For constraint mode, write one verdict block per checklist item. For overall mode, compare which response better follows the instruction after accounting for hard constraints, semantic completeness, and usable final output.

## Final Output

Follow the output format requested by the current benchmark prompt:

- For constraint mode, return one analysis block for each checklist item in the requested `[检查项N-开始] ... [检查项N-结束]` format.
- For overall mode, return the required final label `[[A]]` or `[[B]]` after your comparison.
Do not call any final-output tool unless the harness explicitly exposes it in the tool list.
