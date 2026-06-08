# JudgeBench Rubric

JudgeBench often contains tasks with an objective answer or a clear instruction-following target.

Use visible reference, answer key, checklist, or verifier output before general style preference.
For multiple-choice tasks, identify each response's final selected option and compare it to the visible reference answer when provided.
When no hard answer is visible, choose the response that better follows the instruction and avoids reasoning errors.

JudgeBench-specific calibration:
- The final JSON verdict is the candidate response label `A` or `B`, not the multiple-choice option letter from the question.
- Treat the current A/B labels as only presentation labels; do not infer hidden chosen/rejected origin.
- Extract each candidate's final answer string first, especially repeated-letter outputs such as `AAAAA`.
- If a visible answer key gives one correct option and exactly one response's final option matches it, choose that response even if the other response is more polished.
- If a visible answer key gives multiple acceptable options and both responses choose an acceptable option, use reasoning quality, completeness, and whether the response recognizes the ambiguity to break the tie.
- For LiveBench and LiveCodeBench samples, do not stop at "both final answers match"; use reasoning quality, constraint handling, code correctness, and the OpenRS-style pairwise fallback to break ties.
- Avoid Tie unless the two responses are genuinely indistinguishable after checking answer correctness, reasoning/code quality, and external pairwise evidence.
- A visible reference is strong evidence only when it directly matches the question and answer format. If it is partial, list-valued, or conflicts with clear visible reasoning, use it as evidence rather than an automatic oracle.
- Do not over-reward long rationales after the final answer is wrong.
- Do not over-penalize terse answers if they satisfy the requested answer format and final answer.
- For MMLU-Pro-style prompts, answer-format examples such as `AAAAA` or `KKKKK` are format examples, not constraints to choose that letter.
- When one candidate follows the requested output format and the other gives an unfocused explanation without the requested final answer, format compliance can decide the pair.
