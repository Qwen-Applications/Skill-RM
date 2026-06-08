# Instruction-Following RL Reproduction

This document covers the Skill-RM instruction-following RL experiment. The
release provides a pointwise Skill-RM reward recipe that can be installed into a
separate `verl` checkout.

## Scope

The run uses:

```text
Trainer:        verl with GRPO
Policy init:    Llama-3.1-Tulu-3-8B-SFT
Training data:  VerInstruct
Reward judge:   Qwen3.5-27B
Reward setting: skill_mounted_verifier_plus_code
```

This recipe focuses on the pointwise instruction-following RL setting described
in the paper.

## Install the verl Recipe

Prepare a compatible external `verl` checkout. The recipe is tested with a
checkout whose `verl/version/version` file contains:

```text
0.8.0.dev
```

Use a separate `verl` checkout and install the Skill-RM recipe into it. The
installer can either copy the recipe into an existing checkout, or clone a
public checkout and then copy the recipe. Always verify the version file before
running a full RL job.

To use an existing checkout:

```bash
export VERL_ROOT="/path/to/verl"
test "$(cat "$VERL_ROOT/verl/version/version")" = "0.8.0.dev"
python scripts/install_verl_recipe.py --verl-root "$VERL_ROOT"
```

To clone from the public `verl` repository, pin a known compatible ref when
possible. The helper validates the version string after checkout:

```bash
export VERL_ROOT="/path/to/verl"
python scripts/install_verl_recipe.py \
  --clone-url https://github.com/verl-project/verl.git \
  --target "$VERL_ROOT" \
  --verl-ref main
```

Replace `main` with a pinned commit or tag for archival reproduction if you
have one; the helper rejects refs whose version file does not match.

If the target recipe already exists, pass `--overwrite` deliberately:

```bash
python scripts/install_verl_recipe.py --verl-root "$VERL_ROOT" --overwrite
```

The installer copies:

```text
integrations/verl/recipe/skill_rm_if_pointwise
```

to:

```text
/path/to/verl/recipe/skill_rm_if_pointwise
```

## Prepare VerInstruct Pointwise Data

The pointwise reward does not require reference responses. It needs a prompt in
`extra_info.query` or `extra_info.prompt`, and can optionally use VerInstruct
checker/function resources.

Install optional data-preparation dependencies:

```bash
pip install -e ".[rl]"
```

Generate train and validation parquet files from VerInstruct:

```bash
python scripts/prepare_verinstruct_pointwise.py \
  --dataset THU-KEG/VerInstruct \
  --output-dir /path/to/verinstruct_pointwise \
  --validation-size 0.02 \
  --seed 42
```

Outputs:

```text
/path/to/verinstruct_pointwise/train.parquet
/path/to/verinstruct_pointwise/validation.parquet
/path/to/verinstruct_pointwise/metadata.json
```

For a smoke subset:

```bash
python scripts/prepare_verinstruct_pointwise.py \
  --dataset THU-KEG/VerInstruct \
  --output-dir /path/to/verinstruct_pointwise_smoke \
  --max-train-samples 128 \
  --max-val-samples 32
```

The script also accepts a local JSONL input for offline checks:

```bash
python scripts/prepare_verinstruct_pointwise.py \
  --input-jsonl /path/to/verinstruct_like.jsonl \
  --output-dir /path/to/verinstruct_pointwise
```

Each JSONL row should contain `prompt`, `query`, or `instruction`; optional
`checkers` and `functions` fields are preserved.

## Prepare the Policy Initialization Model

`MODEL_PATH` points to the policy model trained by `verl`. It is separate from
the reward judge model configured by `MODEL_NAME` and `MODEL_BASE_URLS`.

The RL template uses `allenai/Llama-3.1-Tulu-3-8B-SFT` as the policy
initialization model. Download it to a local path and follow the model license
terms:

```bash
pip install -U "huggingface_hub[cli]"
export MODEL_PATH="$HOME/skillrm_models/Llama-3.1-Tulu-3-8B-SFT"
hf download allenai/Llama-3.1-Tulu-3-8B-SFT --local-dir "$MODEL_PATH"
test -f "$MODEL_PATH/config.json"
```

## Check the Reward Function

Inside the external `verl` checkout:

```bash
cd "$VERL_ROOT"
pip install -e ".[vllm]"
pip install -r recipe/skill_rm_if_pointwise/requirements.txt

export SKILL_RM_BACKEND=mock
python -m compileall recipe/skill_rm_if_pointwise
python - <<'PY'
import asyncio
from recipe.skill_rm_if_pointwise.reward_fn_pointwise import compute_score

async def main():
    result = await compute_score(
        data_source=None,
        solution_str="1. Apples\n2. Bananas\n3. Cherries",
        ground_truth=None,
        extra_info={"query": "List exactly three fruits using a numbered list.", "sample_id": "smoke"},
    )
    print(result)

asyncio.run(main())
PY
```

## Run the GRPO Template

Serve the reward judge as OpenAI-compatible endpoints, then configure the
external `verl` checkout:

```bash
cd "$VERL_ROOT"

export TRAIN_FILE="/path/to/verinstruct_pointwise/train.parquet"
export VAL_FILE="/path/to/verinstruct_pointwise/validation.parquet"
export MODEL_NAME="Qwen3.5-27B"
export MODEL_BASE_URLS="http://127.0.0.1:8000/v1"

export NUM_GPUS=8
export REWARD_NUM_WORKERS=48
export DEFAULT_CONCURRENCY=128
export HTTP_POOL_CONNECTIONS=128
export HTTP_POOL_MAXSIZE=512

test "$(cat verl/version/version)" = "0.8.0.dev"
test -f recipe/skill_rm_if_pointwise/reward_fn_pointwise.py
test -f "$MODEL_PATH/config.json"
test -f "$TRAIN_FILE"
test -f "$VAL_FILE"
test -n "$MODEL_BASE_URLS"

bash recipe/skill_rm_if_pointwise/run_grpo.sh
```

For a small smoke run, override the batch sizes and epoch/checkpoint settings:

```bash
TRAIN_PROMPT_BSZ=4 \
VAL_PROMPT_BSZ=4 \
N_RESP_PER_PROMPT=2 \
PPO_MINI_BSZ=4 \
PPO_MICRO_BSZ_PER_GPU=1 \
REWARD_NUM_WORKERS=4 \
TOTAL_EPOCHS=1 \
SAVE_FREQ=1000 \
TEST_FREQ=1000 \
bash recipe/skill_rm_if_pointwise/run_grpo.sh
```

The template keeps the reward setting:

```text
skill_mounted_verifier_plus_code
```
