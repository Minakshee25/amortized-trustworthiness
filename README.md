# Amortized Trustworthiness

Training small language models as real-time uncertainty proxies for large language models.

MSc Individual Project, Department of Computing, Imperial College London.

## Overview

Sampling-based uncertainty estimators such as semantic entropy detect confabulation well but
cost several generations plus a clustering pass per query, which is too slow for
latency-constrained deployment. This project asks whether that computation can be *amortised*:
run it once, offline, over many prompts and target LLMs, and distil the resulting signal into a
small language model (SLM) that predicts it in a single forward pass.

The proxy is trained on a three-input tuple `(x, y*, z)` — the prompt, the greedy answer, and a
decoder hidden state from the target LLM projected into a shared space — and regresses the target
LLM's semantic entropy. Because the text inputs are model-agnostic, the same proxy can in
principle be pointed at a target LLM it was never trained on.

### Research questions

- **RQ1 — Amortised inference efficiency.** Can an SLM proxy predict a large target LLM's
  multi-sample semantic entropy in one forward pass, and what latency reduction does that give
  over sampling?
- **RQ2 — Supervision signal.** Which ground-truth uncertainty metric is the most stable and
  distillable training target?
- **RQ3 — Input modality and access.** Does the proxy need the target LLM's hidden state
  (grey-box), or does it transfer from prompt and response text alone (black-box)? Does it
  generalise across model families and knowledge domains?

### Headline results

| | finding | experiments |
|---|---|---|
| RQ1 | one proxy forward pass is ~90–100x faster per question than 10-sample semantic entropy with DeBERTa clustering; ~14–16x end to end | E61 |
| RQ2 | the semantic clustering carries the signal; binarising the entropy label costs ~0.07 AUROC; the entropy weighting itself adds little over a raw cluster count | E56, E58 |
| RQ3 | raw hidden states do **not** transfer across LLMs; an orthogonal-Procrustes alignment recovers most of the signal; a text-only proxy (`q_resp_only`) transfers directly and, with no target labels and no sampling, is on par with 10-sample semantic entropy in-distribution and beats a supervised hidden-state probe (SEP) out-of-distribution | E20–E42, E45–E73 |

The full chronological account, including retracted and superseded results, is in
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md). A per-experiment index is in
[`docs/experiment_index.md`](docs/experiment_index.md).

## Repository layout

```
amortized_ue/            installable package
  config.py                Stage-1 configuration dataclass
  paths.py                 canonical filesystem locations
  stage1/                  offline construction of the semantic-entropy dataset
    build.py                 CLI: python -m amortized_ue.stage1.build
    records.py, loaders.py, sep_bridge.py, wandb_io.py
  stage2/                  the SLM proxy
    model.py                 architecture: projector + frozen backbone + LoRA + regression head
    train.py                 training loop, (position, layer, k) selection
    run.py                   CLI: python -m amortized_ue.stage2.run
    data.py, config.py, checkpoint.py
  baselines/              ridge probes, SEP bridge, label-noise and linear ceilings
  alignment/             cross-model representational alignment (orthogonal Procrustes)
  eval/                  shared evaluation engines (correctness, SE-fidelity, latency, fusion)

experiments/            one runnable script per experiment, grouped by research question
  rq1_efficiency/  rq2_supervision/  rq3_modality/
  cross_llm_transfer/  lolo_thesis/  big_tier/  qwen_gemma/  model_specificity/
  archive/               retracted / superseded experiments, kept for the record

semantic_uncertainty/    semantic-entropy generation pipeline, adapted from jlko/semantic_uncertainty
semantic_entropy_probes/ SEP probe training (baseline)
results/                 result JSON/CSV for every experiment + fitted baseline objects
docs/                    EXPERIMENTS.md, experiment_index.md, methodology.md, development.md
environment/             pinned dependency lists for the two environments
```

## Installation

Two Python 3.10 environments are used; exact pins are in [`environment/`](environment/).

```bash
# generation + analysis (semantic-entropy pipeline, ridge diagnostics)
python -m venv .venv-gen && . .venv-gen/bin/activate
pip install -r environment/generation.txt
pip install -e .

# proxy training (separate env: newer transformers + peft for the 3B backbone)
python -m venv .venv-proxy && . .venv-proxy/bin/activate
pip install -r environment/proxy.txt
pip install -e .
```

Required environment variables: `HUGGING_FACE_HUB_TOKEN`, `WANDB_ENT` (or run offline),
`OPENAI_API_KEY` (a placeholder is fine unless using GPT entailment). Set `AMORTIZED_UE_DATA`
to the Stage-1 dataset directory if it is not `./data/stage1`. See
[`docs/development.md`](docs/development.md) for machine-specific notes.

## Pipeline

**Stage 1 — build the offline semantic-entropy dataset** (generation env):

```bash
python -m amortized_ue.stage1.build --model_name Llama-2-7b-chat --dataset trivia_qa --num_samples 2000
```

One id-keyed record per prompt: greedy answer and accuracy, all-layer hidden states at the
token-before-generation (TBG) and second-last-token (SLT) positions, N high-temperature samples,
and a continuous `cluster_assignment_entropy` label. Resumable. See
[`docs/methodology.md`](docs/methodology.md) for the extraction internals and
[`data/README.md`](data/README.md) for regeneration and dataset artefacts.

**Stage 2 — train the proxy** (proxy env):

```bash
python -m amortized_ue.stage2.run \
  --ood --ood_dataset squad --ood_num_samples 1000 --seeds 5 --reuse_selection \
  --arms z,z_q,z_q_resp,q_only,q_resp_only \
  --z_inputs TBG:22,SLT:15 --selected_k 4 --projector_hidden_dim 1024 \
  --run_name reference

python -m amortized_ue.stage2.run --eval --eval_datasets squad:1000   # reload, no retraining
```

**Diagnostics and baselines** (generation env, CPU):

```bash
python -m amortized_ue.baselines.ridge          --data_dir <stage1>   # ridge ceiling + layer picker
python -m amortized_ue.baselines.label_noise_ceiling --data_dir <stage1>
```

**Experiments.** Each script under `experiments/` is runnable as a module and writes to
`results/`, for example:

```bash
python -m experiments.rq2_supervision.supervision_signal_compare --data_dir <stage1>
python -m experiments.cross_llm_transfer.procrustes_e26_decomposition
python -m experiments.lolo_thesis.correctness_eval_e37 --data_dir <stage1>
```

## Attribution

The `semantic_uncertainty/` module is adapted from
[jlko/semantic_uncertainty](https://github.com/jlko/semantic_uncertainty) (Kuhn et al.; Farquhar
et al., *Nature* 2024). The semantic-entropy-probe baseline follows Kossen et al.,
*Semantic Entropy Probes* (arXiv:2406.15927). Changes to the vendored pipeline are limited to
model-loading redirects (ungated mirrors) and are documented in `docs/development.md`.
