# Pointwise Aggregation Policy

Aggregate verified constraints into one scalar reward. The score should reflect task usability, not just the number of boxes checked.

## Default Flow

1. Start with all mandatory visible constraints.
2. Mark each as satisfied, failed, or not applicable.
3. Compute `base = satisfied_count / total_count` when counts are available.
4. Apply severity caps and adjustments.
5. Return score, counts, confidence, used resources, and a short reason naming the decisive evidence.

## Severity Weights

- `critical`: failure makes the response unusable for the requested task. Examples: wrong task, required JSON invalid, answer in wrong language, unjustified refusal, missing the main deliverable.
- `major`: failure materially reduces usability but the answer still addresses the task. Examples: one required section missing, count far outside limit, required keyword absent.
- `minor`: failure is peripheral or stylistic. Examples: slightly verbose when concision requested, one optional example omitted, mild tone mismatch.

## Score Caps

Apply the most restrictive relevant cap:

- unrelated or wrong-task response: `0.25`;
- unjustified refusal for an allowed request: `0.25`;
- missing main deliverable: `0.50`;
- invalid required structured format: `0.55`;
- exact-only answer polluted by explanations: `0.60`;
- mandatory language violated: `0.65`;
- multiple major failures: `0.60`;
- one isolated minor failure: no cap; usually `0.85-0.94`.

These are caps, not automatic scores. A response can be lower if it has additional failures.

## Verifier And Sandbox Evidence

Use deterministic evidence as follows:

- If `sample.verinstruct.verify_all` returns `satisfied_count/total_count`, use that ratio for covered rule constraints.
- If `python_sandbox` confirms a hard constraint failure, mark that item failed even if the response feels close.
- If deterministic evidence is inconclusive, fall back to direct visible-text judgment and lower confidence.

Do not double-count the same constraint from both a checklist item and your own decomposition. If a checklist item combines several requirements, split it mentally only if the final count would otherwise hide an important partial failure.

## Confidence

- `0.8-1.0`: exact evidence covers hard constraints and semantic judgment is straightforward.
- `0.6-0.8`: most constraints are clear, but some style/semantic judgment remains.
- `0.4-0.6`: ambiguous instruction or limited verifier coverage.
- below `0.4`: parsing or backend/tool errors make the score uncertain.
