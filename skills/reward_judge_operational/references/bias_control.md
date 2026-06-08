# Bias Control

- Ignore candidate labels except as identifiers.
- Do not reward first position, longer answers, more markdown, or more confident language by itself.
- Penalize over-answering when it violates the user's requested format.
- If an external method output is present, treat it as evidence, not as a hidden label.
