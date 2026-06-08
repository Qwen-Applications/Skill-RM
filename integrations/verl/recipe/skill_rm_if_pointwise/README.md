# Pointwise Skill-RM IF Reward Recipe for verl

This recipe contains the pointwise instruction-following reward used for the
Skill-RM GRPO experiment.

It supports the pointwise reward setting:

```text
skill_mounted_verifier_plus_code
```

The reward judges one rollout response against the visible instruction and
returns a scalar score in `[0, 1]`. It can load an instruction-following skill,
mount VerInstruct sample verifier resources from `extra_info.checkers` and
`extra_info.functions`, use static IF resources, and run deterministic
visible-text checks through `python_sandbox`.

## Required Dataset Fields

Each verl parquet row should include `extra_info.query` or `extra_info.prompt`.

Optional fields:

```text
extra_info.sample_id
extra_info.system_prompt
extra_info.history
extra_info.checkers
extra_info.functions
```

`reference_response` is not used by this pointwise reward.

## External verl Checkout

Install this directory under an external `verl` checkout at:

```text
recipe/skill_rm_if_pointwise
```

The tested `verl/version/version` value is:

```text
0.8.0.dev
```

The release repository provides `scripts/install_verl_recipe.py` to copy this
recipe into either an existing checkout or a cloned public checkout.

## Reward Function

Use this reward function in verl:

```bash
reward.custom_reward_function.path=recipe/skill_rm_if_pointwise/reward_fn_pointwise.py
reward.custom_reward_function.name=compute_score
+reward.custom_reward_function.reward_kwargs.variant=skill_mounted_verifier_plus_code
```

The returned metadata includes invalid flags, satisfied/total counts, resources
used by the judge, resource counts, latency, and parse/backend errors.

## Judge Service Environment

Use OpenAI-compatible judge endpoints:

```bash
export SKILL_RM_BACKEND=openai
export MODEL_NAME=Qwen3.5-27B
export MODEL_BASE_URLS="http://127.0.0.1:8000/v1"
export OPENAI_API_KEY=EMPTY
export ENABLE_THINKING=false
export SEND_THINKING_FIELD=true
export MODEL_TEMPERATURE=0.0
export MODEL_TOP_P=1.0
export MAX_TOKENS=4096
export MODEL_TIMEOUT=300
export MODEL_RETRIES=1
```

For multiple endpoints, set `MODEL_BASE_URLS` to a comma-separated list.

## Offline Smoke Test

```bash
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

## Launch GRPO

After installing this recipe into an external verl checkout, download the policy
initialization model and set the data/model paths:

```bash
pip install -U "huggingface_hub[cli]"
export MODEL_PATH="$HOME/skillrm_models/Llama-3.1-Tulu-3-8B-SFT"
hf download allenai/Llama-3.1-Tulu-3-8B-SFT --local-dir "$MODEL_PATH"
test -f "$MODEL_PATH/config.json"

TRAIN_FILE="/path/to/train.parquet" \
VAL_FILE="/path/to/validation.parquet" \
MODEL_BASE_URLS="http://127.0.0.1:8000/v1" \
bash recipe/skill_rm_if_pointwise/run_grpo.sh
```
