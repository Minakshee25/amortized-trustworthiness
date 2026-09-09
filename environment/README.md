# Environments

Two Python 3.10 environments, both on torch 2.13. They are kept separate because the proxy
backbone (Llama-3.2-3B / Qwen2.5-3B) needs a newer `transformers` than the semantic-entropy
generation pipeline is pinned against.

| file | environment | used for |
|---|---|---|
| `generation.txt` | generation + analysis | Stage-1 dataset construction, the ridge / ceiling diagnostics, and most scripts under `experiments/` |
| `proxy.txt` | proxy training | Stage-2 proxy training and evaluation (`amortized_ue.stage2`) |
| `generation-conda.yaml` | conda form of the generation environment | alternative to `generation.txt` if using conda |

## Setup

```bash
python -m venv .venv-gen   && . .venv-gen/bin/activate   && pip install -r environment/generation.txt && pip install -e .
python -m venv .venv-proxy && . .venv-proxy/bin/activate && pip install -r environment/proxy.txt      && pip install -e .
```

The `.txt` files are `pip freeze` output from the environments the results were produced in, so
they pin every transitive dependency. Key versions:

- generation: `transformers` 5.x, `tokenizers` 0.22, `scikit-learn` 1.7, `numpy` 2.2
- proxy: `transformers` 4.52.4, `peft` 0.19, `tokenizers` 0.21, `scikit-learn` 1.3, `numpy` 1.26

A third environment on `transformers` 4.44 was used historically for Llama-3-8B Stage-1
generation only; it is not needed to reproduce any downstream result.

## Environment variables

- `HUGGING_FACE_HUB_TOKEN` — required for the gated model configs
- `AMORTIZED_UE_DATA` — Stage-1 dataset directory (default `./data/stage1`)
- `AMORTIZED_UE_CKPTS` — directory holding the saved proxy checkpoints, read by the cross-model
  evaluation scripts (default `/data2/mn1025/stage2_checkpoints`; the checkpoints are also on W&B)
- `WANDB_ENT` — W&B entity, or run with `WANDB_MODE=offline`
- `OPENAI_API_KEY` — needed only for GPT-based entailment; a placeholder satisfies the import

`amortized_ue.stage1.*` and the semantic-entropy pipeline import only in the generation
environment; the proxy environment is for `amortized_ue.stage2` and the evaluation scripts.
