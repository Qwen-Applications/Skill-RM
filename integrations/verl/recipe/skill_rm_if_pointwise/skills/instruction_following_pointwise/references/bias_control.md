# Bias Control For Pointwise IF Judging

Before finalizing, check whether the provisional score is being pulled by irrelevant surface features.

## Common Over-Reward Biases

- Verbosity: a longer answer is not better unless the instruction asks for detail.
- Formatting polish: Markdown, tables, and headings do not compensate for missing required content or invalid requested structure.
- Confidence: assertive language can still be wrong or non-compliant.
- Helpfulness spillover: useful extra information is a penalty if the instruction requested a constrained or exact-only output.
- Semantic leniency: near misses on exact counts, required words, schema, or output language are still failures.

## Common Under-Reward Biases

- Minimal compliant answer: short answers can deserve high scores when the instruction asks for concise or exact output.
- Style mismatch over-penalty: awkward wording is minor if the required task and hard constraints are satisfied.
- Resource anchoring: mounted checklist/verifier resources can be incomplete; do not punish a response for failing an unlisted requirement unless that requirement is visible in the prompt.
- Refusal over-penalty: refusal can be correct when the visible instruction asks for unsafe, impossible, or disallowed content.

## Calibration Checks

Ask these before final_answer:

1. Did I verify every exact hard constraint that affects the score?
2. Did I cap the score for any critical failure?
3. Am I rewarding presentation rather than instruction satisfaction?
4. Am I treating a verifier as broader than what it actually checks?
5. Would a user who needed the specified output be able to use this response as-is?

If the answer to question 5 is no, the score should usually be below `0.7`, even if many secondary constraints pass.
