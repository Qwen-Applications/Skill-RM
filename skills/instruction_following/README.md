# Instruction-Following Skill

This is the local instruction-following skill snapshot used by
`if_rewardbench_experiments`. It is self-contained in this directory and is
loaded through the local `load_skill` tool.

## Purpose

The skill helps a judge evaluate whether one assistant response follows the
visible instruction, system prompt, and conversation history. It supports:

- checklist verification for IF-RewardBench constraint assessment;
- pairwise comparison prompts that first decompose instruction-following
  requirements before choosing `[[A]]` or `[[B]]`.

The judge must use only visible fields: prompt/instruction, response,
system_prompt, history, and any visible `checklist` supplied in the user
prompt or sample payload.

## Files

```text
instruction_following/
  SKILL.md
  scripts/constraint_tools.py
```

`constraint_tools.py` helpers are preloaded into the local `execute_python`
tool. The tool receives only visible IF-RewardBench sample fields and has no
file, network, subprocess, hidden-label, or benchmark-oracle access.
