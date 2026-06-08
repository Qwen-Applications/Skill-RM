# Constraint Verification Protocol

Use this protocol to turn the visible instruction into evidence before scoring.

## Evidence Table

For non-trivial samples, build this table mentally or in the final rationale:

| field | meaning |
| --- | --- |
| `id` | short stable label, such as `format_json` or `word_count_50` |
| `constraint` | the atomic visible requirement |
| `type` | `main_task`, `format`, `count`, `keyword`, `language`, `content`, `style`, `safety` |
| `hardness` | `hard`, `soft`, or `conditional` |
| `evidence` | verifier output, sandbox output, or visible text evidence |
| `status` | `satisfied`, `failed`, or `not_applicable` |
| `severity_if_failed` | `minor`, `major`, or `critical` |

Only count active constraints in `total_count`. Do not count suggestions or examples as mandatory unless the instruction makes them required.

## When To Use Deterministic Tools

Use mounted verifiers or `python_sandbox` for:

- exact word, sentence, bullet, paragraph, line, section, character, or example counts;
- required/forbidden words, phrases, punctuation, prefixes, suffixes, delimiters;
- JSON parseability, required JSON keys, list/table/code block structure;
- regex-like format requirements, bracket/quote balance, exact answer-only constraints;
- simple arithmetic or transformations whose correctness can be checked mechanically.

Use direct semantic judgment for:

- whether the answer actually solves the requested problem;
- paraphrase quality, reasoning quality, style, tone, persona, audience fit;
- nuanced refusal boundaries or safety policy implications;
- factual correctness unless a visible reference or local evidence source is provided.

## Verification Rules

- A deterministic check only proves the property it checked. Do not let it decide unrelated semantic quality.
- Near misses fail exact constraints. "About 50 words" can be approximate; "exactly 50 words" is not.
- Extra prose fails exact-only outputs when the instruction says to return only a label, JSON, number, word, or requested format.
- Invalid required JSON/schema is a format failure even if the content is otherwise useful.
- If the instruction asks for several deliverables, missing any central deliverable is a major failure.
- If the instruction has conditional requirements, verify whether the condition is triggered before counting it.

## Mounted Verifier Interpretation

`sample.verinstruct.verify_all` may cover only rule-like constraints. Treat it as strong evidence for covered items, but still score:

- main task completion;
- semantic correctness;
- style/tone/persona;
- visible constraints not represented in the verifier code.

If a verifier result conflicts with obvious visible evidence, inspect the checklist item and explain the conflict briefly in the final reason.
