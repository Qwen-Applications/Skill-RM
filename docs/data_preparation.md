# Data Preparation

Prepare benchmark data and model checkpoints in a working directory, then point
the runners at those locations with environment variables or command-line
paths. This document provides source links, expected layouts, and validation
commands.

The commands below use a single external data root:

```bash
export SKILLRM_DATA_HOME="$HOME/skillrm_data"
mkdir -p "$SKILLRM_DATA_HOME"
```

The examples below keep downloaded data under `$SKILLRM_DATA_HOME` and pass the
paths to the runners through environment variables.

## RewardBench2 / JudgeBench / RM-Bench

Source:

```text
https://github.com/Qwen-Applications/OpenRS
```

The main reward-judging configs expect an OpenRS-compatible data root. The
smallest reproducible setup is to download the four JSONL files used by the
main experiments:

```bash
export SKILLRM_DATA_ROOT="$SKILLRM_DATA_HOME/openrs_data"
export OPENRS_COMMIT="4c7f22b570753630e91d8450205d33f1580a00df"

mkdir -p \
  "$SKILLRM_DATA_ROOT/rewardbench_v2" \
  "$SKILLRM_DATA_ROOT/judgebench" \
  "$SKILLRM_DATA_ROOT/rmbench"

curl -L \
  -o "$SKILLRM_DATA_ROOT/rewardbench_v2/rewardbench_v2.jsonl" \
  "https://raw.githubusercontent.com/Qwen-Applications/OpenRS/$OPENRS_COMMIT/data/rewardbench_v2/rewardbench_v2.jsonl"

curl -L \
  -o "$SKILLRM_DATA_ROOT/judgebench/gpt.jsonl" \
  "https://raw.githubusercontent.com/Qwen-Applications/OpenRS/$OPENRS_COMMIT/data/judgebench/gpt.jsonl"

curl -L \
  -o "$SKILLRM_DATA_ROOT/judgebench/claude.jsonl" \
  "https://raw.githubusercontent.com/Qwen-Applications/OpenRS/$OPENRS_COMMIT/data/judgebench/claude.jsonl"

curl -L \
  -o "$SKILLRM_DATA_ROOT/rmbench/rmbench.jsonl" \
  "https://raw.githubusercontent.com/Qwen-Applications/OpenRS/$OPENRS_COMMIT/data/rmbench/rmbench.jsonl"
```

Expected files:

```text
$SKILLRM_DATA_ROOT/rewardbench_v2/rewardbench_v2.jsonl
$SKILLRM_DATA_ROOT/judgebench/gpt.jsonl
$SKILLRM_DATA_ROOT/judgebench/claude.jsonl
$SKILLRM_DATA_ROOT/rmbench/rmbench.jsonl
```

Validate:

```bash
test -s "$SKILLRM_DATA_ROOT/rewardbench_v2/rewardbench_v2.jsonl"
test -s "$SKILLRM_DATA_ROOT/judgebench/gpt.jsonl"
test -s "$SKILLRM_DATA_ROOT/judgebench/claude.jsonl"
test -s "$SKILLRM_DATA_ROOT/rmbench/rmbench.jsonl"

wc -l \
  "$SKILLRM_DATA_ROOT/rewardbench_v2/rewardbench_v2.jsonl" \
  "$SKILLRM_DATA_ROOT/judgebench/gpt.jsonl" \
  "$SKILLRM_DATA_ROOT/judgebench/claude.jsonl" \
  "$SKILLRM_DATA_ROOT/rmbench/rmbench.jsonl"
```

Reference line counts for the pinned files are 1865, 350, 270, and 1327.
OpenRS is Apache-2.0. The benchmark rows may originate from their respective
upstream datasets, so users should follow the applicable upstream licenses.

## JETTS Sequential Knockout

Source:

```text
https://github.com/SalesforceAIResearch/jetts-benchmark
```

Download the public reranking/refinement archive:

```bash
export JETTS_DATA_ROOT="$SKILLRM_DATA_HOME/jetts"
export JETTS_DATA_DIR="$JETTS_DATA_ROOT/reranking_and_refinement"

mkdir -p "$JETTS_DATA_ROOT"

python - <<'PY'
import os
import urllib.request
from pathlib import Path

url = "https://storage.googleapis.com/sfr-jetts-benchmark-data/reranking_and_refinement.tar.gz"
data_home = Path(os.environ.get("SKILLRM_DATA_HOME", Path.home() / "skillrm_data"))
out = data_home / "jetts" / "reranking_and_refinement.tar.gz"
out.parent.mkdir(parents=True, exist_ok=True)
urllib.request.urlretrieve(url, out)
print(out)
PY

tar xzf "$JETTS_DATA_ROOT/reranking_and_refinement.tar.gz" -C "$JETTS_DATA_ROOT"
```

The directory should contain JSONL files named as:

```text
<dataset>_<generator>.jsonl
```

The bundled JETTS configuration evaluates these task families:

```text
gsm8k         GSM8K
ifeval        IFEval
humaneval     HumanEval+ task family in the JETTS files
bigcodebench  BigCodeBench
```

Validate:

```bash
test -s "$JETTS_DATA_DIR/gsm8k_Qwen2.5-72B-Instruct.jsonl"
test -s "$JETTS_DATA_DIR/bigcodebench_Qwen2.5-72B-Instruct.jsonl"
test -s "$JETTS_DATA_DIR/ifeval_Qwen2.5-72B-Instruct.jsonl"
test -s "$JETTS_DATA_DIR/humaneval_Qwen2.5-72B-Instruct.jsonl"

wc -l "$JETTS_DATA_DIR/humaneval_Qwen2.5-72B-Instruct.jsonl"
```

The JETTS upstream release is licensed CC BY-NC 4.0. This repository provides
the runner and configuration template; users provide the JETTS data path and
judge endpoint configuration.

## IF-RewardBench

Source:

```text
https://github.com/thu-coai/IF-RewardBench
```

Preferred setup:

```bash
export SKILLRM_IF_REWARDBENCH_ROOT="$SKILLRM_DATA_HOME/IF-RewardBench"

git clone --depth 1 \
  https://github.com/thu-coai/IF-RewardBench.git \
  "$SKILLRM_IF_REWARDBENCH_ROOT"
```

If `git clone` is unavailable or unreliable in your environment, download only
the files needed by the runner:

```bash
export SKILLRM_IF_REWARDBENCH_ROOT="$SKILLRM_DATA_HOME/IF-RewardBench"
export IF_REWARDBENCH_COMMIT="c192fe968c15520352c7e7d9a853a3900f612d06"

mkdir -p \
  "$SKILLRM_IF_REWARDBENCH_ROOT/data" \
  "$SKILLRM_IF_REWARDBENCH_ROOT/inference" \
  "$SKILLRM_IF_REWARDBENCH_ROOT/metrics"

curl -L \
  -o "$SKILLRM_IF_REWARDBENCH_ROOT/data/if_rewardbench.json" \
  "https://raw.githubusercontent.com/thu-coai/IF-RewardBench/$IF_REWARDBENCH_COMMIT/data/if_rewardbench.json"

curl -L \
  -o "$SKILLRM_IF_REWARDBENCH_ROOT/inference/position_maps.json" \
  "https://raw.githubusercontent.com/thu-coai/IF-RewardBench/$IF_REWARDBENCH_COMMIT/inference/position_maps.json"

curl -L \
  -o "$SKILLRM_IF_REWARDBENCH_ROOT/metrics/analysis_constraint_assessment.py" \
  "https://raw.githubusercontent.com/thu-coai/IF-RewardBench/$IF_REWARDBENCH_COMMIT/metrics/analysis_constraint_assessment.py"

curl -L \
  -o "$SKILLRM_IF_REWARDBENCH_ROOT/metrics/analysis_overall_assessment.py" \
  "https://raw.githubusercontent.com/thu-coai/IF-RewardBench/$IF_REWARDBENCH_COMMIT/metrics/analysis_overall_assessment.py"
```

Expected layout:

```text
$SKILLRM_IF_REWARDBENCH_ROOT/data/if_rewardbench.json
$SKILLRM_IF_REWARDBENCH_ROOT/inference/position_maps.json
$SKILLRM_IF_REWARDBENCH_ROOT/metrics/analysis_constraint_assessment.py
$SKILLRM_IF_REWARDBENCH_ROOT/metrics/analysis_overall_assessment.py
```

Validate:

```bash
test -s "$SKILLRM_IF_REWARDBENCH_ROOT/data/if_rewardbench.json"
test -s "$SKILLRM_IF_REWARDBENCH_ROOT/inference/position_maps.json"
test -s "$SKILLRM_IF_REWARDBENCH_ROOT/metrics/analysis_constraint_assessment.py"
test -s "$SKILLRM_IF_REWARDBENCH_ROOT/metrics/analysis_overall_assessment.py"
```

The IF-RewardBench runner supports `overall` and `constraint`; `overall`
reports `if_rewardbench.overall.kendall` as the official value.

For quick smoke runs, use `--limit-rows N` after pointing the runner at the
IF-RewardBench root. If you already have an expanded dataset JSON, pass it with
`--dataset-cache-path`; official metrics still require the raw data file and
upstream metric scripts listed above. Overall-mode metrics also require the
position map.

## Validate All Data Paths

After exporting the three data variables, run:

```bash
python - <<'PY'
import os
from pathlib import Path

required = {
    "SKILLRM_DATA_ROOT": [
        "rewardbench_v2/rewardbench_v2.jsonl",
        "judgebench/gpt.jsonl",
        "judgebench/claude.jsonl",
        "rmbench/rmbench.jsonl",
    ],
    "JETTS_DATA_DIR": [
        "gsm8k_Qwen2.5-72B-Instruct.jsonl",
        "bigcodebench_Qwen2.5-72B-Instruct.jsonl",
        "ifeval_Qwen2.5-72B-Instruct.jsonl",
        "humaneval_Qwen2.5-72B-Instruct.jsonl",
    ],
    "SKILLRM_IF_REWARDBENCH_ROOT": [
        "data/if_rewardbench.json",
        "inference/position_maps.json",
        "metrics/analysis_constraint_assessment.py",
        "metrics/analysis_overall_assessment.py",
    ],
}

missing = []
for env_name, rel_paths in required.items():
    root = os.environ.get(env_name)
    if not root:
        missing.append(f"{env_name} is not set")
        continue
    for rel_path in rel_paths:
        path = Path(root) / rel_path
        if not path.is_file() or path.stat().st_size == 0:
            missing.append(str(path))

if missing:
    print("Missing required data files:")
    for item in missing:
        print(f"- {item}")
    raise SystemExit(1)

print("All required data files are present.")
PY
```
