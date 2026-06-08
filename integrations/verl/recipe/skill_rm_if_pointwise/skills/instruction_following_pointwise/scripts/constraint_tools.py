"""Pure helper functions preloaded into the pointwise IF python_sandbox.

The sandbox already provides `prompt`, `instruction`, `response`,
`system_prompt`, `history`, and `sample`. These helpers intentionally avoid
file, network, subprocess, or external-package access.
"""

import json
import re
import string
import unicodedata


def normalize_space(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def words(text):
    return re.findall(r"\b[\w'-]+\b", str(text), flags=re.UNICODE)


def word_count(text):
    return len(words(text))


def char_count(text, *, include_spaces=True):
    text = str(text)
    if include_spaces:
        return len(text)
    return len(re.sub(r"\s+", "", text))


def line_count(text, *, nonempty=True):
    lines = str(text).splitlines()
    if nonempty:
        lines = [line for line in lines if line.strip()]
    return len(lines)


def paragraph_count(text):
    blocks = re.split(r"\n\s*\n", str(text).strip())
    return len([block for block in blocks if block.strip()])


def sentence_count(text):
    compact = normalize_space(text)
    if not compact:
        return 0
    matches = re.findall(r"[^.!?。！？]+[.!?。！？]", compact)
    trailing = re.sub(r"[^.!?。！？]+[.!?。！？]", "", compact).strip()
    return len(matches) + (1 if trailing else 0)


def contains(text, needle, *, case_sensitive=False):
    text = str(text)
    needle = str(needle)
    if not case_sensitive:
        text = text.lower()
        needle = needle.lower()
    return needle in text


def count_occurrences(text, needle, *, case_sensitive=False, overlap=False):
    text = str(text)
    needle = str(needle)
    if not case_sensitive:
        text = text.lower()
        needle = needle.lower()
    if not needle:
        return 0
    if not overlap:
        return text.count(needle)
    count = 0
    start = 0
    while True:
        index = text.find(needle, start)
        if index < 0:
            return count
        count += 1
        start = index + 1


def regex_count(text, pattern, *, flags=0):
    return len(re.findall(pattern, str(text), flags=flags))


def regex_search(text, pattern, *, flags=0):
    return re.search(pattern, str(text), flags=flags) is not None


def json_parse(text):
    try:
        return {"valid": True, "value": json.loads(str(text)), "error": ""}
    except Exception as exc:
        return {"valid": False, "value": None, "error": str(exc)}


def json_has_keys(text, keys):
    parsed = json_parse(text)
    if not parsed["valid"] or not isinstance(parsed["value"], dict):
        return {"valid_json": parsed["valid"], "missing": list(keys), "present": []}
    present = [key for key in keys if key in parsed["value"]]
    missing = [key for key in keys if key not in parsed["value"]]
    return {"valid_json": True, "missing": missing, "present": present}


def markdown_heading_count(text, *, level=None):
    pattern = r"(?m)^\s{0,3}#{1,6}\s+\S"
    if level is not None:
        pattern = rf"(?m)^\s{{0,3}}#{{{int(level)}}}\s+\S"
    return regex_count(text, pattern)


def bullet_count(text):
    return regex_count(text, r"(?m)^\s*(?:[-*+]|\d+[.)])\s+\S")


def numbered_list_count(text):
    return regex_count(text, r"(?m)^\s*\d+[.)]\s+\S")


def code_block_count(text, *, language=None):
    if language:
        return regex_count(text, rf"(?ms)^```{re.escape(str(language))}\s.*?^```")
    return regex_count(text, r"(?m)^```")


def balanced_brackets(text):
    pairs = {")": "(", "]": "[", "}": "{"}
    opens = set(pairs.values())
    stack = []
    for char in str(text):
        if char in opens:
            stack.append(char)
        elif char in pairs:
            if not stack or stack[-1] != pairs[char]:
                return False
            stack.pop()
    return not stack


def punctuation_count(text, mark):
    return str(text).count(str(mark))


def unique_word_count(text, *, case_sensitive=False):
    values = words(text)
    if not case_sensitive:
        values = [item.lower() for item in values]
    return len(set(values))


def mostly_language_script(text):
    chars = [char for char in str(text) if char.strip() and char not in string.punctuation]
    if not chars:
        return {"latin": 0.0, "cjk": 0.0, "digit": 0.0, "other": 0.0}
    counts = {"latin": 0, "cjk": 0, "digit": 0, "other": 0}
    for char in chars:
        code = ord(char)
        if char.isdigit():
            counts["digit"] += 1
        elif "LATIN" in unicodedata.name(char, ""):
            counts["latin"] += 1
        elif 0x4E00 <= code <= 0x9FFF or 0x3040 <= code <= 0x30FF or 0xAC00 <= code <= 0xD7AF:
            counts["cjk"] += 1
        else:
            counts["other"] += 1
    total = float(len(chars))
    return {key: value / total for key, value in counts.items()}


def between(value, low=None, high=None, *, inclusive=True):
    if low is not None:
        if inclusive and value < low:
            return False
        if not inclusive and value <= low:
            return False
    if high is not None:
        if inclusive and value > high:
            return False
        if not inclusive and value >= high:
            return False
    return True


def emit(value):
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))
