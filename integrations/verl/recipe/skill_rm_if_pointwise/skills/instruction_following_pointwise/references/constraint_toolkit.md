# Constraint Toolkit For `python_sandbox`

Use `python_sandbox` only for short deterministic checks over visible text. The sandbox cannot access files, network, subprocesses, hidden labels, or dataset metadata.

Available variables:

- `prompt` and `instruction`: visible user prompt text.
- `response`: candidate response text.
- `system_prompt`: visible system prompt text, if present.
- `history`: visible conversation history text, if present.
- `sample`: dict with the same visible fields.

Helper functions from `scripts/constraint_tools.py` are preloaded, so you can call them directly.

## Common Helpers

- `word_count(text)`, `char_count(text, include_spaces=True)`
- `line_count(text, nonempty=True)`, `paragraph_count(text)`, `sentence_count(text)`
- `contains(text, needle, case_sensitive=False)`
- `count_occurrences(text, needle, case_sensitive=False, overlap=False)`
- `regex_count(text, pattern, flags=0)`, `regex_search(text, pattern, flags=0)`
- `json_parse(text)`, `json_has_keys(text, keys)`
- `bullet_count(text)`, `numbered_list_count(text)`, `markdown_heading_count(text, level=None)`
- `code_block_count(text, language=None)`, `balanced_brackets(text)`
- `mostly_language_script(text)`, `between(value, low=None, high=None, inclusive=True)`
- `emit(value)` prints compact JSON evidence.

## Examples

Exact word count:

```python
n = word_count(response)
result = {"constraint": "exactly_50_words", "word_count": n, "satisfied": n == 50}
```

JSON with required keys:

```python
result = json_has_keys(response, ["answer", "reason"])
result["satisfied"] = result["valid_json"] and not result["missing"]
```

Forbidden phrase and bullet count:

```python
result = {
    "bullet_count": bullet_count(response),
    "has_forbidden_phrase": contains(response, "as an ai language model"),
}
result["satisfied"] = result["bullet_count"] == 3 and not result["has_forbidden_phrase"]
```

Language/script sanity check:

```python
shares = mostly_language_script(response)
result = {"script_share": shares, "mostly_latin": shares["latin"] >= 0.8}
```

Regex format:

```python
ok = regex_search(response.strip(), r"^Answer:\s*[A-D]$")
result = {"constraint": "single_choice_answer_line", "satisfied": ok}
```

## Usage Rules

- Print JSON or set `result = {...}`. Keep output compact.
- Check only visible properties; do not infer hidden labels.
- A sandbox result is evidence for the property tested, not a complete score.
- If code fails, simplify the check rather than guessing.
