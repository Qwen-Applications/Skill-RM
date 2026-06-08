#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: scripts/run_ablation_model.sh <qwen35_9b|qwen35_27b|qwen35_35b_a3b|qwen35_122b_a10b> [run_root]" >&2
  exit 2
fi

model_key="$1"
run_root="${2:-runs/ablations/${model_key}}"

: "${SKILLRM_DATA_ROOT:?Set SKILLRM_DATA_ROOT to the benchmark data directory, e.g. /path/to/OpenRS-master/data.}"
: "${SKILLRM_BASE_URLS:?Set SKILLRM_BASE_URLS to comma-separated OpenAI-compatible /v1 endpoints.}"
export SKILLRM_MODEL_KEY="${model_key}"

case "$model_key" in
  qwen35_9b)
    export SKILLRM_MODEL="Qwen3.5-9B"
    workers="${SKILLRM_WORKERS:-64}"
    ;;
  qwen35_27b)
    export SKILLRM_MODEL="Qwen3.5-27B"
    workers="${SKILLRM_WORKERS:-80}"
    ;;
  qwen35_35b_a3b)
    export SKILLRM_MODEL="Qwen3.5-35B-A3B"
    workers="${SKILLRM_WORKERS:-32}"
    ;;
  qwen35_122b_a10b)
    export SKILLRM_MODEL="Qwen3.5-122B-A10B"
    workers="${SKILLRM_WORKERS:-128}"
    ;;
  *)
    echo "unknown model key: ${model_key}" >&2
    exit 2
    ;;
esac

mkdir -p "${run_root}/logs"

echo "run_root=${run_root}"
echo "model=${SKILLRM_MODEL}"
echo "workers=${workers}"
echo "data_root=<set>"
echo "base_url_count=$(python - <<'PY'
import os
print(len([x for x in os.environ["SKILLRM_BASE_URLS"].split(",") if x]))
PY
)"

for bench in rewardbench2 judgebench rmbench; do
  for ablation in fair_flat_prompt flat_prompt tool_only; do
    config="configs/ablations/${bench}/${ablation}.yaml"
    output="${run_root}/${ablation}/${bench}"
    log="${run_root}/logs/${bench}_${ablation}.log"
    echo "[$(date -Is)] start ${model_key} ${ablation}/${bench}" | tee -a "$log"
    python -m skillrm.runners.ablation --config "$config" --output "$output" --workers "$workers" 2>&1 | tee -a "$log"
    echo "[$(date -Is)] done ${model_key} ${ablation}/${bench}" | tee -a "$log"
  done
done
