# Repository Manifest

This repository contains the Skill-RM reproduction code:

- reward-judging runners for RewardBench2, JudgeBench, and RM-Bench;
- skill packages for the standard-input and resource-available settings;
- resource-use ablation runners;
- JETTS sequential knockout evaluation code;
- IF-RewardBench evaluation code;
- pointwise instruction-following RL recipe for external `verl`;
- configuration templates, documentation, tests, and Apache-2.0 metadata.

Runtime inputs and outputs are prepared separately:

- benchmark data or cached expanded datasets;
- model checkpoints or generated model outputs;
- processed RL parquet files;
- endpoint addresses or local secrets;
- run logs, validation files, or generated experiment outputs.

Prepare benchmark data in a working directory and point the runners at those
paths with environment variables. See `docs/data_preparation.md`.
