---
name: reward_judge_fair
description: Use this Skill-RM reward judge to compare candidate responses for a visible user request with generic rubric, principles, bias controls, output contract, and Python sandbox checks over visible text.
metadata:
  family: reward_judge
  short-description: Generic visible-text reward judge
  method: skill_fair
---

# Reward Judge

Use this skill to organize a reward judgment from the current user request and candidate responses. The skill is a controller and resource interface, not a per-sample prompt template.

## Inputs

The host message provides only:

- the visible user prompt or instruction;
- candidate responses and their current labels;
- the required final output format.

Use the current prompt and candidate responses as the full task context.

## Resource Interface

After this skill is loaded, use only resources listed in the current resource index. The resources are generic:

- `rubric`: generic reward judging criteria;
- `principle`: generic correctness, instruction-following, safety, usefulness, and anti-style-bias principles;
- `calibration`: position, verbosity, style, and confidence-bias controls;
- `aggregation`: generic evidence-combination policy;
- `output_contract`: JSON verdict contract;
- `tool`: `python_sandbox`, which can inspect only the visible prompt and candidate responses.

## Tool Use

Use `view_resource` to read generic rubric, principles, bias control, aggregation, or output format resources.

Use `python_sandbox` when deterministic checking over visible text can change the verdict. It runs short Python over only:

- `prompt`: the visible user prompt;
- `candidates`: the current visible candidate responses keyed by label;
- `sample`: `{"prompt": prompt, "candidates": candidates}`.

Use it for counts, regex/format checks, JSON/list structure, simple arithmetic, supplied examples, small code-behavior checks, or answer extraction from visible candidate text.

`run_resource` should normally not be used with this skill. Read generic resources with `view_resource`, use `python_sandbox` for deterministic visible-text checks, then submit `final_answer`.

## Decision Procedure

1. Identify the user's actual task and mandatory constraints from the prompt.
2. Compare candidates under one shared criterion.
3. Prioritize hard correctness, instruction following, safety, factuality, and required output format.
4. Use `python_sandbox` only for checks that can be computed from visible prompt/candidates.
5. Apply bias controls: do not prefer position, length, markdown polish, confidence, or fluent style unless it improves task success.
6. Use `Tie` only when candidates are genuinely equivalent or the visible evidence is insufficient for a reliable preference.
7. Return the required JSON.

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
