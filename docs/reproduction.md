# Reproduction

## Environment

Prepare external benchmark data first:

```bash
# Exports SKILLRM_DATA_ROOT, JETTS_DATA_DIR, and SKILLRM_IF_REWARDBENCH_ROOT.
# See docs/data_preparation.md for copy-paste download commands and checks.
```

```bash
pip install -e .
cp .env.example .env
# Fill in endpoint and data paths.
source .env
```

## Main Matrix

```bash
bash scripts/run_release_model.sh qwen35_27b outputs/paper_main/qwen35_27b
python scripts/summarize_official_acc.py outputs/paper_main/qwen35_27b
```

Supported model keys:

```text
qwen35_9b
qwen35_27b
qwen35_35b_a3b
qwen35_122b_a10b
```

The model key only sets the served model name and default worker count. Endpoint
addresses must be supplied through `SKILLRM_BASE_URLS`.

Qwen thinking is disabled by default for judge runs. Compatible vLLM endpoints
receive `chat_template_kwargs.enable_thinking=false`; endpoints that reject that
field are retried without it.

## Ablations

```bash
bash scripts/run_ablation_model.sh qwen35_27b outputs/ablations/qwen35_27b
python scripts/summarize_official_acc.py outputs/ablations/qwen35_27b
```

Ablations:

```text
fair_flat_prompt
flat_prompt
tool_only
```

## Metrics

This repository reports benchmark official/full-set accuracy:

```text
RewardBench2: official leaderboard average
JudgeBench:   order-swapped aggregation over merged GPT and Claude subsets
RM-Bench:     win / total
```

For JudgeBench, each sample has two order-swapped judgments. A correct vote is
`+1`, an incorrect vote is `-1`, and tie/abstain is `0`. A sample is correct
when the total score is positive.

## JETTS SeqKO

```bash
bash scripts/run_jetts_seqko.sh configs/jetts_seqko/qwen72b.example.yaml
```

## IF-RewardBench

```bash
bash scripts/run_if_rewardbench.sh overall if_rb_overall_agentic
bash scripts/run_if_rewardbench.sh constraint if_rb_constraint_agentic
```
