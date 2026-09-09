# Development notes

Design decisions, environment details, and machine-specific setup for the code in this
repository. The chronological experiment log is in [`EXPERIMENTS.md`](EXPERIMENTS.md); the
data-generation and hidden-state internals are in [`methodology.md`](methodology.md).

## Module boundaries

`amortized_ue/` reuses the semantic-entropy pipeline in `semantic_uncertainty/` read-only.
`stage1/sep_bridge.py` puts that directory on `sys.path` and re-exports the functions used:
`HuggingfaceModel.predict(return_latent=True)`, `load_ds`, prompt construction, `get_metric`,
`get_reference`, `split_dataset`, `EntailmentDeberta`, `get_semantic_ids`,
`cluster_assignment_entropy`. Nothing under `semantic_uncertainty/` is modified except
model-loading redirects in `uncertainty/models/huggingface_models.py` (ungated NousResearch
mirrors for the gated Llama-2 / Llama-3 weights, plus the `8b` load branch); the semantic-entropy
and probe logic is untouched.

## Stage-1 record schema (`stage1-v1`)

One `.pt` file per prompt, keyed by `id`:

```
id, question, context, reference
canonical:                              # greedy (temperature 0.1) answer
  response, accuracy, token_log_likelihoods
  hidden_states: { TBG: [L+1, 1, H], SLT: [L+1, 1, H] }    # all layers, native dtype
samples: [ {response, token_log_likelihoods, semantic_id}, ... ]     # N high-temperature
labels: cluster_assignment_entropy (continuous float), semantic_ids, n_clusters, n_samples
meta: { model, dataset, temperatures, entailment settings, git_commit, positions, ... }
```

Records are joined across files **by id**, never by list position — this removes the positional
alignment fragility of the original pipeline (see `methodology.md`, section 7).

### Hidden-state positions

| record key | position | hidden-states index |
|---|---|---|
| `TBG` | token before generation (last prompt token) | `hidden[0]` |
| `SLT` | second-last generated token | `hidden[n_gen - 2]` |

`HuggingfaceModel.predict()` returns `(scalar, second_last, last_token_before_gen)`;
`stage1/build.py` unpacks these into `SLT` and `TBG` respectively. Note that the semantic-entropy
probe baseline stores its own keys the other way round: this repo's `TBG` corresponds to the SEP
code's `emb_tok_before_eos` / `slt_dataset`, and this repo's `SLT` to its
`emb_last_tok_before_gen` / `tbg_dataset`. This matters only when comparing directly against SEP
numbers.

## The proxy

`stage2/model.py`:

- **Projector.** `LayerNorm(H_in) -> Linear(H_in, hidden) -> GELU -> Dropout(0.1) ->
  Linear(hidden, k * d_model)`, reshaped to `k` soft tokens, each unit-normalised and scaled by a
  learnable scalar. `hidden` is `--projector_hidden_dim`: 1024 when stacking positions, where 256
  is a measurable bottleneck. The input interface is `[B, n_layers_in, H]` and flattens, so
  `--z_inputs POS:LAYER,...` stacks positions with no change to `model.py`.
- **One model per arm** (`z`, `z_q`, `z_q_resp`, `q_only`, `q_resp_only`), each trained on its own
  fixed, null-free sequence — no modality dropout. The text-only arms skip the projector.
- **Backbone** frozen (Llama-3.2-3B by default; Qwen2.5-3B in E66), LoRA r16 / alpha32 / dropout
  0.05 on the attention projections, plus a trainable linear head and a REG readout embedding.
  bf16 backbone, fp32 projector and head.
- **Target** z-scored on the training split; metrics reported in the original label space, with
  Spearman as the primary measure and AUROC via a training-split `best_split`. R^2 is not
  meaningful under a dataset shift because the label scale moves; rank metrics are used there.

The `(position, layer)` input is selected with the exact ridge sweep in
`amortized_ue/baselines/ridge.py`, not the proxy's built-in sweep (which is unreliable — see
E9). The current best input is `TBG:22, SLT:15` with `k = 4`; the two positions are
complementary, additional layers within a position are not.

## Environments

Two environments, both Python 3.10, torch 2.13; exact pins in `environment/generation.txt` and
`environment/proxy.txt`.

| environment | transformers | used for |
|---|---|---|
| generation | 5.x | Stage-1 generation, the ridge / ceiling diagnostics, most `experiments/` scripts |
| proxy | 4.52.4 (+ peft) | Stage-2 proxy training (the 3B backbone needs the newer rope config) |

A third environment with transformers 4.44 was used historically for Llama-3-8B Stage-1
generation, where the generation environment's tokenizer was too new and the proxy environment
hit a DeBERTa checkpoint-format wall. Llama-2 and Llama-3 weights load through ungated
NousResearch mirrors.

`amortized_ue/config.py` reads `AMORTIZED_UE_DATA` for the Stage-1 dataset location, falling back
to `<repo>/data/stage1`. Scripts that write results resolve `RESULTS_DIR` from
`amortized_ue/paths.py`, so they can be run from anywhere.

Individual `experiments/` scripts also honour `AMORTIZED_UE_DATA`; a few that were pinned to the
staged copy on this machine fall back to `/data2/mn1025/stage1` when it is unset.

## Checkpoints

Trained proxy checkpoints are large (a single 5-arm run is ~3 GB; all runs together ~40 GB) and
are **not** committed. `amortized_ue/paths.py` resolves them through `ckpt_dir(run_name, …)`,
which looks under `$AMORTIZED_UE_CKPTS/<run_name>/` first, then the in-repo
`amortized_ue/stage2/runs/<run_name>/` — so an eval reads either a pre-staged copy or one this
repo just trained. Every training script writes under `amortized_ue/stage2/runs/` and offers
`--stage push_wandb`; every experiment can also be run from scratch with `--stage all` (train +
eval), which needs no pre-existing checkpoint.

All checkpoints are mirrored to Weights & Biases, project `amortized_ue_stage2`, `type=model`:

| artifact | contents |
|---|---|
| `stage2_ckpts_REFERENCE_multipos_p1024_5arm_ckpt` | the reference Llama-2 proxy (5 arms × 5 seeds), TBG:22 + SLT:15, k=4 |
| `stage2_ckpts_Mistral-7B-Instruct-v0.2_trivia_qa_n2000` | E22 role-swap proxy (Mistral, 5 arms) |
| `stage2_ckpts_deploy_checkpoints` | the shared `q_resp_only` deploy proxy (4 alignment targets); used by `e45`–`e49`, `e61`, `e64`, `se_fidelity` |
| `stage2_ckpts_E37_LOLO_ckpt` | the thesis leave-one-LLM-out proxy (4 folds); used by `correctness_eval_lolo_squad`, `e68` |
| `stage2_ckpts_E40_pooled_multimodel_ridge` | pooled multi-model ridge (`alignment.npz` + `ridges.npz`) for the model-specificity analysis |
| `stage2_ckpts_E63_lto_6model_qresp` | E63 leave-two-out proxy (6 source models) |
| `stage2_ckpts_E65_bigtier_lolo_qresp` | E65 big-tier (5 × 27B) leave-one-LLM-out |
| `stage2_ckpts_E66_qwen25_proxy_lolo` | E66 Qwen2.5-3B backbone swap |
| `stage2_ckpts_E27_resp_only_arm`, `stage2_ckpts_E27_zresp_arm` | E27 alignment-arm proxies |
| `e70_…_bundles`, `e71_…`, `e72_…`, `e73_…`, `set1_full_eval_ckpts` | E70–E74 aligned-ridge bundles and per-model proxies |

To use a staged copy, download the relevant artifact(s) into one directory and point
`AMORTIZED_UE_CKPTS` at it (fallback `/data2/mn1025/stage2_checkpoints`, which holds `REFERENCE`,
`E22`, `deploy_checkpoints`, `E37_LOLO_ckpt` on the project machine).

## Machine-specific setup (Imperial DoC)

The home directory has a small quota that silently breaks large downloads, so model weights,
conda environments, and the Stage-1 dataset all live on `/data2` (or `/vol/bitbucket`):

- `export HF_HOME=/data2/<user>/hf_cache`
- Stage-1 records staged on `/data2` load in a few seconds; the shared NFS is far slower and
  periodically degrades, which is why every real build is also mirrored to a W&B artifact.
- `~/.bashrc` returns early for non-interactive shells, so all `export`s that batch jobs need
  (`WANDB_ENT`, `WANDB_API_KEY`, `HUGGING_FACE_HUB_TOKEN`, `OPENAI_API_KEY`, `HF_HOME`) must sit
  above that guard.
- `uncertainty/utils/openai.py` builds its client at import time, so `OPENAI_API_KEY` must be
  set even for runs that make no OpenAI call (a placeholder value is fine).
- This account's W&B key is 86 characters (institutional SSO). The `wandb login` CLI rejects it,
  but `wandb.init()` accepts it — store it in `WANDB_API_KEY` and `~/.netrc`.

## Reproducing an experiment

Every `experiments/` script is a module entry point and writes its output to `results/`. Each
Stage-1 dataset is (re)built with `python -m amortized_ue.stage1.build --model_name … --dataset …
--num_samples …`; the flags used for each build are recorded in the Stage-1 schema section above
and in the experiment docstrings.
