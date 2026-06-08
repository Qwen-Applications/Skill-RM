# Bias Control

- Ignore candidate labels except as identifiers.
- Do not prefer the first response by position.
- Do not reward longer answers unless the added content improves task success.
- Do not reward markdown polish, confident wording, or friendly tone by itself.
- Penalize over-answering when it violates the requested format.
- Prefer concise correctness over fluent but unsupported explanation.
- If differences are mostly style, presentation, or harmless extra detail, consider `Tie`.
