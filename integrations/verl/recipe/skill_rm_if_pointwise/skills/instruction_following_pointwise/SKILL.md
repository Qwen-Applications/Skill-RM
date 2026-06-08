---
name: instruction_following_pointwise
description: |
  Use this skill when judging a single candidate response for instruction following and a scalar score is needed. It is especially useful when the sample contains exact constraints, a visible checklist, verifier resources, format requirements, word/count limits, language restrictions, or multiple sub-instructions that should be decomposed before scoring.
metadata:
  family: skill_rm_if
  method: pointwise_scalar
---

# Instruction-Following Pointwise Judge

You are judging one visible sample: an instruction plus one candidate response. Your job is to decide how well the response follows the instruction and return a calibrated score in `[0, 1]`.

Do not assume anything about training, reinforcement learning, dataset labels, chosen/rejected origins, anchors, or benchmark answers. Use only the visible instruction, response, and resources exposed by the tool interface.

## When To Use Resources

Use only resources that can change the score.

- If `sample.verinstruct.checklist` exists, read it before final scoring. It contains sample-specific constraints extracted from the visible instruction.
- If `sample.verinstruct.verify_all` exists and constraints are exact or rule-like, run it. Treat its result as evidence for the constraints it explicitly checks.
- If `if.python_sandbox` is available, use the `python_sandbox` tool for deterministic checks that would be error-prone by inspection: counts, regex, JSON validity, bullet/list structure, required/forbidden terms, exact prefix/suffix, delimiter counts, quote/bracket balance, or arithmetic.
- If no sample checklist exists, use `if.constraint_verification_protocol` and `if.pointwise_rubric` for decomposition and calibration.
- Use `if.constraint_toolkit` before writing sandbox code if you need helper function names or examples.

Do not read every resource by default. The best path is usually: checklist or decomposition, exact verification when needed, then score aggregation.

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
4. Verify hard constraints first. Use mounted verifiers or `python_sandbox` when the answer depends on exact visible text properties.
5. Judge semantic completeness and usefulness after hard constraints. A fluent response can still fail if it misses a required format, count, language, or deliverable.
6. Aggregate evidence into `satisfied_count`, `total_count`, and `score`. If counts are unavailable, estimate them from your decomposed checklist.

## Scoring Contract

- `0.95-1.0`: all mandatory constraints satisfied; only negligible imperfections.
- `0.75-0.9`: main task succeeds with minor non-critical misses.
- `0.45-0.7`: partially useful but misses an important constraint, format, or deliverable.
- `0.15-0.4`: mostly fails the task, gives a generic answer, or violates a critical requirement.
- `0.0-0.15`: empty, unrelated, unusable, unjustified refusal, or direct contradiction of the main task.

When a checklist exists, `satisfied_count / total_count` is the base score. Adjust downward for central/main-task failures or safety problems. Adjust upward only when the checklist is clearly incomplete and the response satisfies important visible requirements not listed there.

## Final Answer

Call `final_answer` with:

```json
{
  "score": 0.0,
  "satisfied_count": 0,
  "total_count": 0,
  "confidence": 0.0,
  "used_resources": [],
  "reason": "short evidence-based reason"
}
```

Include only resources you actually read or ran. Keep `reason` short but name the decisive satisfied and failed requirements.
