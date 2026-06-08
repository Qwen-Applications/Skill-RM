# Generic Aggregation

Combine only evidence available from the prompt, candidate responses, generic resources, and allowed Python sandbox output over visible text.

Priority:

1. Deterministic visible-text checks: exact format, required counts, simple arithmetic, JSON/list validity, code behavior that can be inferred or executed safely from visible snippets.
2. Direct task correctness and instruction following.
3. Safety, factuality, and unsupported claims.
4. Usefulness and completeness.
5. Style, organization, and clarity as weak tie-breakers only.

If Python sandbox evidence conflicts with subjective impressions, trust the deterministic visible-text evidence when it directly applies to the user request.
