#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: scripts/run_if_rewardbench.sh <overall|constraint> <config_bundle> [output_dir]" >&2
  exit 2
fi

mode="$1"
bundle="$2"
output="${3:-outputs/if_rewardbench/${mode}_${bundle}}"

: "${SKILLRM_IF_REWARDBENCH_ROOT:?Set SKILLRM_IF_REWARDBENCH_ROOT to the IF-RewardBench directory.}"
: "${SKILLRM_BASE_URLS:?Set SKILLRM_BASE_URLS to comma-separated OpenAI-compatible /v1 endpoints.}"
: "${SKILLRM_MODEL:=Qwen3.5-27B}"
: "${SKILLRM_WORKERS:=32}"

python -m experiments.if_rewardbench.runner \
  --mode "$mode" \
  --config-bundle "$bundle" \
  --backend vllm \
  --base-urls "$SKILLRM_BASE_URLS" \
  --model "$SKILLRM_MODEL" \
  --data-path "$SKILLRM_IF_REWARDBENCH_ROOT/data/if_rewardbench.json" \
  --position-map-path "$SKILLRM_IF_REWARDBENCH_ROOT/inference/position_maps.json" \
  --metrics-dir "$SKILLRM_IF_REWARDBENCH_ROOT/metrics" \
  --skills-dir skills \
  --workers "$SKILLRM_WORKERS" \
  --send-thinking-field \
  --no-enable-thinking \
  --output "$output" \
  --resume

