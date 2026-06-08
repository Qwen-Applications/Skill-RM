# Operational Aggregation

Use this priority order:

1. Hard verifier, visible reference answer, visible ground truth, checklist, or executable test-case evidence.
2. Reliable external pipeline output from a resource listed for the current sample.
3. Benchmark-specific rubric, principle, and OpenRS-style visible metadata routing.
4. Skill-guided Qwen3.5-27B judgment using only the visible prompt, candidate responses, and resources the model loaded or ran.
5. If evidence is inconclusive, return `Tie` only when the candidates are genuinely equivalent or the visible evidence is insufficient for a reliable preference.

Do not use previous baseline predictions as fallback evidence in the clean paper path.
Never expose chosen, rejected, gold label, or per-sample oracle routing to the judge.
