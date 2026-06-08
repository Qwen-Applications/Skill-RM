# Generic Principles

- Correctness first: a polished but wrong response loses to a plainer correct response.
- Constraint fidelity: explicit user constraints are hard requirements.
- Visible evidence only: use only the prompt, candidate responses, generic resources, and allowed visible-text tool outputs.
- Anti-style bias: length, tone, markdown, and confidence are not quality by themselves.
- Pairwise locality: judge only the presented candidates.
- Calibrated Tie: use `Tie` when neither candidate is clearly better from visible evidence.
