# IF-RewardBench

This folder contains the IF-RewardBench runner used for instruction-following
reward evaluation. The benchmark data and upstream metric scripts are not
included in the repository.

The runner supports the `overall` and `constraint` IF-RewardBench protocols.
The older scalar protocol is intentionally not exposed. For `overall`, the
reported official value is Kendall correlation:

```text
if_rewardbench.overall.kendall
```

## Data

Set:

```bash
export SKILLRM_IF_REWARDBENCH_ROOT=/path/to/IF-RewardBench-main
```

Expected files:

```text
$SKILLRM_IF_REWARDBENCH_ROOT/data/if_rewardbench.json
$SKILLRM_IF_REWARDBENCH_ROOT/inference/position_maps.json
$SKILLRM_IF_REWARDBENCH_ROOT/metrics/analysis_constraint_assessment.py
$SKILLRM_IF_REWARDBENCH_ROOT/metrics/analysis_overall_assessment.py
```

The runner validates these paths before evaluation and reports the missing
files if the environment is not prepared.

For smoke tests, prefer `--limit-rows N` with the IF-RewardBench data root
above. `--dataset-cache-path` may point to a prepared expanded dataset JSON;
official metrics still require the raw data file and upstream metric scripts.
Overall-mode metrics also require the position map.

## Skills

Agentic IF-RewardBench runs expose only the `instruction_following` skill by
default, even though the repository contains other reward-judging skills for
RewardBench2, JudgeBench, and RM-Bench. Override `--allowed-skills` only when
intentionally testing a different skill surface.

## Run

OpenAI-compatible endpoint run:

```bash
python -m experiments.if_rewardbench.runner \
  --mode overall \
  --config-bundle if_rb_overall_agentic \
  --backend vllm \
  --base-urls "$SKILLRM_BASE_URLS" \
  --model "$SKILLRM_MODEL" \
  --workers 32 \
  --output outputs/if_rewardbench/overall_agentic
```

The runner disables Qwen thinking by default by sending
`chat_template_kwargs.enable_thinking=false` when supported. To opt into
thinking-mode evaluation, pass `--enable-thinking`.

Config bundles live under `configs/if_rewardbench/`.
