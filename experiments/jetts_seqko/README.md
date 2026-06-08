# JETTS Sequential Knockout

This folder contains the JETTS best-of-N / sequential pairwise knockout runner.
It evaluates a generator's candidate responses by repeatedly comparing the
current incumbent with the next candidate.

## Data

Set:

```bash
export JETTS_DATA_DIR=/path/to/jetts/reranking_and_refinement
```

The data directory should contain files named:

```text
<dataset>_<generator>.jsonl
```

Each row should include a prompt-like field and a `responses` list. Response
metadata may include scores used for Pass@1, Random@N, and Oracle@N references.

The bundled configuration uses these dataset keys:

```text
gsm8k         GSM8K
ifeval        IFEval
humaneval     HumanEval+ task family in the JETTS files
bigcodebench  BigCodeBench
```

## Configure

Use one of the templates under `configs/jetts_seqko/`:

```bash
export SKILLRM_BASE_URLS="http://<host>:8000/v1,http://<host>:8001/v1"
export SKILLRM_MODEL="Qwen3.5-27B"
```

## Run

The JETTS `baseline` setting is the direct-judge best-of-N baseline.

```bash
python experiments/jetts_seqko/run_seqko.py \
  --config configs/jetts_seqko/qwen72b.example.yaml \
  --limit 2 \
  --settings baseline \
  --datasets gsm8k
```

Full run:

```bash
python experiments/jetts_seqko/run_seqko.py \
  --config configs/jetts_seqko/qwen72b.example.yaml
```

Resume is enabled by default. Completed `sample_id`s in `predictions.jsonl` are
skipped on restart. Use `--force` only to truncate this experiment's own output
files for the requested setting, dataset, and seed.

Use a fresh `output_dir` or pass `--force` when changing the JETTS baseline
configuration; resume keys are based on setting, dataset, and seed.
