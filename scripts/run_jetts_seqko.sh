#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 1 ]]; then
  echo "usage: scripts/run_jetts_seqko.sh <config.yaml>" >&2
  exit 2
fi

: "${JETTS_DATA_DIR:?Set JETTS_DATA_DIR to the JETTS reranking_and_refinement directory.}"
: "${SKILLRM_BASE_URLS:?Set SKILLRM_BASE_URLS to comma-separated OpenAI-compatible /v1 endpoints.}"

python experiments/jetts_seqko/run_seqko.py --config "$1"

