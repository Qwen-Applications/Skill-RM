---
name: reward_judge_operational
description: >-
  Use this Skill-RM reward judge when response judging may benefit from resource-rich evidence: benchmark/task metadata, visible references or ground truth, checklists, verifier signals, code/math/factuality tool protocols, same-backbone judging pipeline signals, or bias-control resources. Load it when these resources can materially change the verdict; do not load it merely to be thorough.
metadata:
  family: reward_judge
  short-description: Resource-rich operational reward judge
  method: skill_operational
---

# Reward Judge

Use this skill to organize and access reward-judging resources. The skill is a controller and resource interface, not a per-sample prompt template.

This skill has one clean operational path: inspect the visible resource index, use only resources listed for the current sample, and produce the required verdict JSON.

## Inputs

The host message provides the visible judging task:

- benchmark and task metadata, when the operational setting allows it;
- user prompt or instruction;
- candidate responses and their current labels;
- required final output format.

Hidden chosen/rejected/gold/test labels are not available and must not be inferred.

## Resource Interface

After this skill is loaded, inspect the resource index. Use only resources that can materially affect the verdict.

Common resource types:

- `rubric`: benchmark or task criteria.
- `principle`: high-level judging principles.
- `metadata`: visible benchmark/task metadata.
- `reference`: visible reference answer or ground truth.
- `checklist`: visible constraints, criteria, or checklist items.
- `verifier`: runnable or documented verifier signal over visible content.
- `tool`: runnable tool over visible content, such as `python_sandbox`.
- `tool_protocol`: code, math, or factuality verification protocol.
- `external_pipeline`: same-backbone judging pipeline evidence exposed by the current run.
- `calibration`: position, verbosity, style, and confidence-bias controls.
- `aggregation`: policy for combining conflicting evidence.

Prefer resource reads in this order when relevant:

1. Visible hard evidence: reference, ground truth, checklist, executable verifier, or test-case evidence.
2. Reliable same-backbone external pipeline output.
3. Benchmark/task-specific rubric and principles.
4. Bias-control and aggregation resources for close calls.

Do not read every resource. For ordinary clear cases, direct judgment is acceptable. For close, exact, or correctness-sensitive cases, load only the few resources needed.

## Tool Use

Use `list_resources` only when the index returned at skill load is insufficient.

Use `view_resource` to read reference resources such as rubrics, principles, visible references, checklists, or external outputs.

Use `run_resource` only for resources whose index entry has `implementation_kind` like `runtime_verifier` or `runtime_llm_pipeline`. Do not run `reference` or `tool_protocol` resources; read them with `view_resource` instead. Treat runtime output as evidence, not as a hidden label.

Use `python_sandbox` when exact deterministic checking can change the verdict. It runs short Python over only:

- `prompt`: the visible user prompt;
- `candidates`: the current visible candidate responses keyed by label;
- `sample`: `{"prompt": prompt, "candidates": candidates}`.

It does not receive hidden labels, chosen/rejected origin, benchmark gold fields, files, network, or oracle metadata. Use it for counts, regex/format checks, JSON/list structure, simple arithmetic, supplied examples, small code-behavior checks, and answer extraction. Print compact JSON evidence or assign a JSON-serializable `result`, then use the evidence in `final_answer`.

For RM-Bench / scaled pairwise samples, if `external.openrs_pairwise_qwen` is available, it is a high-value external pipeline signal using the same base model with a preserved OpenRS-style pairwise prompt. Run it when the direct comparison is not already obvious, then reconcile its result with visible hard evidence and the rubric.

For JudgeBench, treat the task as forced choice. If this skill is loaded, use at least one relevant recommended resource before finalizing. For objective-answer, LiveBench, and LiveCodeBench samples, first check visible reference or verifier evidence when available; if that does not decide the winner and `external.openrs_pairwise_qwen` is listed, run it as the normal external-pipeline signal. Use `Tie` only when both responses are genuinely equivalent after resource checks.

For JudgeBench / MMLU-Pro / LiveBench / LiveCodeBench samples, verdict labels `A` and `B` always mean Response A and Response B, not the answer option inside the question. Use `sample.reference_or_ground_truth` and consider `verifier.reference_match` to extract each response's final answer before deciding. If the verifier is inconclusive, both responses match the reference, no reference is available, or a tie is tempting, use the listed external pipeline resources as soft evidence and reconcile them with the visible prompt/reference evidence.

If `verifier.ground_truth_score_pair` is available on an objective-answer sample, run it when the candidates' final answers differ and a visible ground truth or reference can decide correctness. Treat its scores as same-base-model verifier evidence over visible reference content; reconcile it with exact reference matches and the candidate text before finalizing.

OpenRS-style operational runs may expose `sample.task_metadata`, `sample.reference_or_ground_truth`, and `sample.checklist_or_constraints`. Treat `ground_truth`, `constraints`, `check_list`, and sanitized `additional_metadata` as resource evidence only when they are present in the runtime resource index.

## Decision Procedure

1. Identify the user’s actual task and mandatory constraints.
2. Build one shared criterion for the current candidate set.
3. Check visible hard failures before style: wrong final answer, violated required format, unsafe refusal boundary, missing requested deliverable, unsupported factual claim, or failed checklist item.
4. If the resource index contains a visible reference, ground truth, checklist, verifier signal, or reliable external pipeline output that is relevant to the current task, use it before general style preference.
5. Use benchmark/task-specific resources only when they improve the current judgment.
6. Apply bias controls before finalizing: do not prefer position, length, markdown polish, confident tone, or fluent rationale unless it improves task success.
7. Aggregate evidence and return the required JSON.

## Leakage Rules

- Do not use hidden chosen/rejected/gold/label information.
- Do not infer the answer from dataset construction artifacts.
- Do not request or use oracle-only resources.
- Do not treat external pipeline signals as gold labels.
- Record resources actually used in `used_resources`.

## Output

Return JSON only:

```json
{
  "verdict": "A|B|Tie",
  "confidence": 0.0,
  "used_resources": [],
  "reason": "short reason"
}
```
