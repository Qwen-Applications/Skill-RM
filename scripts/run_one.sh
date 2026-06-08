#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: scripts/run_one.sh <benchmark> <setting>" >&2
  echo "benchmarks: rewardbench2 rmbench judgebench" >&2
  echo "settings: baseline skill_fair skill_operational" >&2
  exit 2
fi

bench="$1"
setting="$2"
config="configs/${bench}/${setting}.yaml"

if [[ ! -f "$config" ]]; then
  echo "missing config: $config" >&2
  exit 2
fi

if [[ -z "${SKILLRM_BASE_URLS:-}" ]]; then
  echo "SKILLRM_BASE_URLS is required, e.g. http://<host>:8000/v1,http://<host>:8001/v1" >&2
  exit 2
fi

if [[ -z "${SKILLRM_DATA_ROOT:-}" ]]; then
  echo "SKILLRM_DATA_ROOT is required, e.g. /path/to/OpenRS-master/data" >&2
  exit 2
fi

export SKILLRM_MODEL="${SKILLRM_MODEL:-Qwen3.5-27B}"

if [[ "$bench" == "rewardbench2" ]]; then
  exec python -m skillrm.runners.rewardbench2 --config "$config" --resume
fi

exec python -m skillrm.runners.pairwise --config "$config" --resume
