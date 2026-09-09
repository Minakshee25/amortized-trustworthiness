# Stage-1 datasets

The Stage-1 datasets (one `.pt` record per prompt) are regenerable and are not committed — the
hidden-state tensors run to tens of gigabytes. This directory is where the code expects them by
default; override with `AMORTIZED_UE_DATA` or the `--data_dir` flag.

## Regenerating

Run from the generation environment (see `environment/`), from the repository root:

```bash
python -m amortized_ue.stage1.build --model_name Llama-2-7b-chat --dataset trivia_qa --num_samples 2000
```

The build is resumable: existing records are skipped. `--smoke --smoke_num_samples 3` runs the
whole path on a few prompts and prints one record. Each dataset is one `<model>` × `<dataset>` ×
`<num_samples>` build with the default `random_seed=10` / `num_few_shot=5` selection; the
`trivia_qa` n1000 sets reuse a shared held-out id list so every model is scored on the same
questions (`--only_ids` / `--selection_num_samples`).

Target models used: Llama-2-7b-chat, Mistral-7B-Instruct-v0.2, Meta-Llama-3-8B-Instruct,
deepseek-llm-7b-chat (the four alignment targets); Qwen3-8B, Qwen3.5-9B, gemma-7b-it,
gemma-2-9b-it (small tier); Qwen3-27B, Qwen3.5-27B, Qwen3.8-27B, gemma-2-27b-it, gemma-3-27b-it
(big tier). Datasets: trivia_qa (in-distribution), squad (out-of-distribution).

## W&B artifacts

Every real build was mirrored to a Weights & Biases artifact in project `amortized_ue_stage1`,
named `stage1_records_<model>_<dataset>_n<N>`. Proxy checkpoints are in the projects referenced
from `docs/EXPERIMENTS.md` (for example `stage2_ckpts_REFERENCE_multipos_p1024_5arm_ckpt`,
`set1_full_eval_ckpts`). `amortized_ue/loaders.py` can load records directly from a W&B artifact
with `load_source="wandb"`.

## Layout

```
data/stage1/<model>_<dataset>_n<N>_full/
  records/<id>.pt
  manifest.json          # written once, at the end of the build
```
