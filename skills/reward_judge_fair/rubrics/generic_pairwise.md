# Generic Reward Rubric

Judge the candidate responses using only the visible user prompt and candidate text.

Use this priority order:

1. Correctness on the actual task.
2. Following explicit user constraints and requested format.
3. Safety and refusal quality when safety is relevant.
4. Factuality and unsupported-claim control.
5. Usefulness, completeness, and practical specificity.
6. Clarity and formatting only as tie-breakers.

Do not reward verbosity, confident tone, markdown polish, or earlier position by itself.

For listwise samples, choose the best visible candidate under one shared criterion. For pairwise samples, choose `A`, `B`, or `Tie`.
