# Project Architecture

This document summarizes the public Skill-RM code layout. The repository is
organized around reusable reward-evaluation skills, benchmark runners, and
integration recipes for the experiments described in the paper.

## Overview

Skill-RM has three main layers:

- public runners for RewardBench2, JudgeBench, RM-Bench, JETTS, IF-RewardBench,
  and instruction-following RL;
- reusable skill packages that expose rubrics, references, checklists,
  verifiers, and tool surfaces to the judge model;
- shared runtime utilities for endpoint calls, parsing, resource loading,
  sandboxed checks, and metric computation.

Benchmark data and model checkpoints are configured by path. See
[data_preparation.md](data_preparation.md) for source links and expected file
layouts.

## Top-Level Layout

```text
configs/                         Public experiment configuration templates
docs/                            Data setup, reproduction, and architecture
experiments/if_rewardbench/      IF-RewardBench runner and metrics
experiments/jetts_seqko/         JETTS sequential pairwise-KO runner
integrations/verl/               Pointwise IF-RL recipe
scripts/                         Launch and summarization scripts
skillrm/                         Core reward-judging runners and shared helpers
skills/                          Public Skill-RM skill packages
tests/                           Smoke and parity tests
```

## Core Reward-Judging Modules

The public CLI entry points live under `skillrm.runners`:

```text
skillrm/runners/rewardbench2.py  RewardBench2 main runner
skillrm/runners/pairwise.py      JudgeBench/RM-Bench pairwise runner
skillrm/runners/ablation.py      Resource-use ablation runner
```

These runner modules implement resume behavior, endpoint failover, output
manifests, and the agent/tool loops used by Skill-RM settings.

Benchmark-specific helpers live under `skillrm.benchmarks`:

```text
skillrm/benchmarks/rewardbench2/data.py     RewardBench2 data loading
skillrm/benchmarks/rewardbench2/prompts.py  RewardBench2 prompts/tool guidance
skillrm/benchmarks/rewardbench2/parsing.py  RewardBench2 parsers
skillrm/benchmarks/rewardbench2/metrics.py  RewardBench2 official metrics
skillrm/benchmarks/pairwise/data.py         JudgeBench/RM-Bench data loading
skillrm/benchmarks/pairwise/prompts.py      JudgeBench/RM-Bench prompts
skillrm/benchmarks/pairwise/parsing.py      JudgeBench/RM-Bench parsers/tools
skillrm/benchmarks/pairwise/metrics.py      JudgeBench/RM-Bench metrics
skillrm/benchmarks/routing.py               Shared route/resource policy
```

Shared and runtime helpers are kept separate:

```text
skillrm/common/                  Config, IO, OpenAI-compatible client,
                                  parsing, stats, and tool-call utilities
skillrm/runtime/                 Skill package loading, runtime resources,
                                  and Python sandbox helpers
skillrm/ablations/prompts.py     Ablation prompt and tool-only surfaces
```

`skillrm.benchmarks.routing` contains the shared route and resource-selection
utilities used by the RewardBench2 and pairwise runners.

## Skill Packages

Skill packages live under `skills/`:

```text
skills/reward_judge_fair/        Standard-input reward-evaluation skill
skills/reward_judge_operational/ Resource-available reward-evaluation skill
skills/instruction_following/    Instruction-following skill for IF-RewardBench
```

Each package is loaded through `skillrm.runtime.skill_package` and may include
skill instructions, rubrics, references, verifiers, and Python helper tools.

## JETTS-SeqKO Modules

`experiments/jetts_seqko/run_seqko.py` is the CLI entry point and compatibility
facade. The implementation is split by responsibility:

```text
config.py        Config loading and endpoint normalization
data.py          Dataset JSONL loading and response normalization
io_utils.py      JSON/JSONL writing, resume loading, and output path helpers
metrics.py       JETTS metrics and markdown rendering
pairwise.py      Baseline and skill pairwise model calls
prompts.py       Plain baseline and skill prompt rendering
randomization.py Deterministic pair ordering
references.py    Reference metric constants
seqko.py         Per-sample sequential knockout logic
tool_schemas.py  JETTS tool schemas and final-label extraction
```

The public `baseline` setting is the direct judge baseline used for the JETTS
best-of-N experiments.

## IF-RewardBench Modules

`experiments/if_rewardbench/` keeps the IF-RewardBench runner separate from the
main OpenRS-style runners because its data format, skill tools, and metric
surface are different. The runner supports the `overall` and `constraint`
protocols and reports the overall official metric with Kendall-style
aggregation.

## Instruction-Following RL Integration

The pointwise RL recipe is under:

```text
integrations/verl/recipe/skill_rm_if_pointwise/
```

It is installed into a separate `verl` checkout with
`scripts/install_verl_recipe.py`. See [rl_reproduction.md](rl_reproduction.md)
for the end-to-end setup.
