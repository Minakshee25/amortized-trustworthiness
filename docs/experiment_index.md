# Experiment index

Each script under `experiments/` is runnable as a module (`python -m experiments.<folder>.<name>`) and writes to `results/`. Full method and numbers for each experiment are in [`EXPERIMENTS.md`](EXPERIMENTS.md).

## RQ1 — amortised inference efficiency

| script | experiment | result |
|---|---|---|
| `experiments/rq1_efficiency/e61_efficiency.py` | E61 — RQ1 inference-latency benchmark: one amortized proxy forward pass vs. the N=10 sampling + DeBERTa-clustering pipeline it replaces | `results/e61_efficiency_Llama-2-7b-chat.json` |

## RQ2 — supervision signal

| script | experiment | result |
|---|---|---|
| `experiments/rq2_supervision/supervision_signal_compare.py` | E56 — SE vs cheaper wrong-answer signals (cluster count, MC entropy, perplexity, binarised SE) | `results/supervision_signal_compare_deepseek_trivia.json` |
| `experiments/rq2_supervision/two_pos_ceiling_freshn1000.py` | E59 — two-position ridge ceiling, re-scored on the fresh n1000 held-out set | — |

## RQ3 — input modality and access

| script | experiment | result |
|---|---|---|
| `experiments/rq3_modality/correctness_e33_ens_vs_qresp.py` | E33 — label-free ensemble vs the text proxy, correctness metric | — |
| `experiments/rq3_modality/exp2_step1_zarm.py` | E37 precursor — multi-target z-arm ridge | — |
| `experiments/rq3_modality/procrustes_e27_zresp_gate.py` | E27 — z / response gating | `results/procrustes_e27_zresp_gate.json` |
| `experiments/rq3_modality/procrustes_e27b_proxy_vs_ridge.py` | E27b — 3B proxy vs ridge on aligned z | `results/procrustes_e27b_proxy_vs_ridge.json` |
| `experiments/rq3_modality/procrustes_e27c_zresp_arm.py` | E27c — z + response arm | `results/procrustes_e27c_zresp_arm.json` |
| `experiments/rq3_modality/procrustes_e27d_resp_only.py` | E27d — response-only arm | `results/procrustes_e27d_resp_only.json` |
| `experiments/rq3_modality/procrustes_e33_ens_vs_qresp.py` | E33 — label-free ensemble vs the text proxy, SE-fidelity | — |

## Cross-LLM transfer and Procrustes alignment

| script | experiment | result |
|---|---|---|
| `experiments/cross_llm_transfer/build_cross_llm_combined_table.py` | E67 — assemble the cross-LLM combined table | — |
| `experiments/cross_llm_transfer/cross_llm_5arm_eval.py` | E67 — raw frozen cross-LLM transfer of the 5-arm proxies | — |
| `experiments/cross_llm_transfer/eval_samemodel_5arm.py` | E23 — same-model 5-arm proxy evaluation | — |
| `experiments/cross_llm_transfer/mistral_trained_proxy_ood.py` | E42 — Mistral-trained proxy on Mistral squad | — |
| `experiments/cross_llm_transfer/procrustes_e26_decomposition.py` | E26 — decompose the aligned transfer (semi-partial correlation + ensemble) | `results/procrustes_e26_decomposition.json` |
| `experiments/cross_llm_transfer/procrustes_e27_auroc.py` | E27 — does hidden-state alignment help UE beyond text (AUROC) | `results/procrustes_e27_auroc.json` |
| `experiments/cross_llm_transfer/procrustes_e27_labelfree_ensemble.py` | E27 — label-free ensemble estimator | `results/procrustes_e27_labelfree_ensemble.json` |
| `experiments/cross_llm_transfer/procrustes_e27a_robustness.py` | E27a — robustness of the alignment result | `results/procrustes_e27a_robustness.json` |
| `experiments/cross_llm_transfer/procrustes_e27a_vs_text.py` | E27a — aligned-z increment over the text baseline | `results/procrustes_e27a_vs_text.json` |
| `experiments/cross_llm_transfer/procrustes_e29_ensemble_sep.py` | E29 — DeepSeek alignment chain, ensemble vs SEP | `results/procrustes_e29_ensemble_sep_Mistral-7B-Instruct-v0.2.json` |
| `experiments/cross_llm_transfer/procrustes_e29_master_table.py` | E29 — four-model alignment master table | `results/procrustes_e29_master_table.json` |
| `experiments/cross_llm_transfer/procrustes_e30_ensemble_sep.py` | E30 — full-power ensemble vs SEP | `results/procrustes_e30_ensemble_sep_Meta-Llama-3-8B-Instruct.json` |
| `experiments/cross_llm_transfer/procrustes_e30_master_table.py` | E30 — full-power four-model alignment master table | `results/procrustes_e30_master_table.json` |

## Leave-one-LLM-out proxy (the thesis experiment)

| script | experiment | result |
|---|---|---|
| `experiments/lolo_thesis/baseline_table_freshn1000.py` | E51 support — per-target SE / SEP / ridge table, fresh n1000 | `results/baseline_table_freshn1000.json` |
| `experiments/lolo_thesis/baseline_table_squad_ood.py` | E51 support — per-target table, squad OOD | `results/baseline_table_squad_ood_n1000.json` |
| `experiments/lolo_thesis/correctness_eval_e37.py` | E38 — correctness evaluation of the E37 leave-one-LLM-out proxy | `results/correctness_eval_e37.json` |
| `experiments/lolo_thesis/correctness_eval_lolo_squad.py` | E54 — true leave-one-LLM-out proxy correctness on squad OOD | `results/correctness_eval_lolo_squad.json` |
| `experiments/lolo_thesis/correctness_eval_ood.py` | E39 — cross-dataset correctness evaluation (trivia to squad) | `results/correctness_eval_ood.json` |
| `experiments/lolo_thesis/e62_qresp_alone_vs_sep.py` | E62 — q_resp_only alone vs each target's own SEP | `results/e62_qresp_alone_vs_sep.json` |
| `experiments/lolo_thesis/e66_qwen25_proxy_lolo.py` | E66 — swap the proxy BACKBONE (Llama-3.2-3B → Qwen2.5-3B) and re-run the thesis experiment | `results/e66_qwen25_proxy_lolo.json` |
| `experiments/lolo_thesis/e68_lolo_squad_llama3_deepseek.py` | E68 — extend the TRUE LOLO-proxy squad OOD eval from Llama-2/Mistral to Llama-3 + DeepSeek | `results/e68_combined_table.json` |
| `experiments/lolo_thesis/log_e51_wandb.py` | E51 — W&B logging | — |
| `experiments/lolo_thesis/log_e52_wandb.py` | E52 — W&B logging | — |

## Big-tier (27B) targets

| script | experiment | result |
|---|---|---|
| `experiments/big_tier/e65_bigtier_lolo.py` | E65 — leave-one-LLM-out `q_resp_only` proxy over the 5 big-tier 27B targets | `results/e65_bigtier_lolo.json` |
| `experiments/big_tier/e69_bigtier_lolo_squad_ood.py` | E69 — the squad OOD counterpart of E65 (big-tier 5×27B leave-one-LLM-out `q_resp_only` proxy) | `results/e69_bigtier_lolo_squad_ood.json` |
| `experiments/big_tier/e70_bigtier_lolo_aligned_ridge.py` | E70 — big-tier (5×27B) leave-one-LLM-out ALIGNED-RIDGE, label-free, ID + OOD | `results/e70_bigtier_lolo_aligned_ridge.json` |
| `experiments/big_tier/e72_bigtier_individual.py` | E72 — big-tier (5×27B) PER-MODEL (individual, not LOLO) supervised ceiling: proxy vs ridge vs SEP on one common layer, ID + OOD | `results/e72_bigtier_individual.json` |
| `experiments/big_tier/set1_full_eval.py` | E74 — Set-1 per-model supervised ceiling (proxy arms vs ridge vs SEP) | `results/set1_full_eval.csv` |

## Qwen / Gemma target families

| script | experiment | result |
|---|---|---|
| `experiments/qwen_gemma/e45_qwen_gemma_zeroshot.py` | E45 | `results/e45_qwen_gemma_zeroshot.json` |
| `experiments/qwen_gemma/e46_examples.py` | E46 | `results/e46_examples.json` |
| `experiments/qwen_gemma/e46_qwen_gemma_pairwise_disagreement.py` | E46 | `results/e46_examples.json` |
| `experiments/qwen_gemma/e47_examples.py` | E47 | `results/e47_examples.json` |
| `experiments/qwen_gemma/e47_qwen_gemma_se_fidelity.py` | E47 | `results/e47_examples.json` |
| `experiments/qwen_gemma/e48_frozen_backbone_baseline.py` | E48 | `results/e48_frozen_backbone_baseline.json` |
| `experiments/qwen_gemma/e49_qwen35_9b_think_leak_check.py` | E49 | `results/e49_qwen35_9b_think_leak_check.json` |
| `experiments/qwen_gemma/e50_crossproxy_id_pull.py` | E50 | `results/e50_crossproxy_id_pull.json` |
| `experiments/qwen_gemma/e53_eval_on_llama2_mistral.py` | E53 — reverse-E45: a proxy trained ONLY on the 4 Qwen/Gemma small-tier models, zero-shot on Llama-2/Mistral | `results/e53_full_comparison.json` |
| `experiments/qwen_gemma/e53_full_comparison.py` | E53 — reverse-E45: a proxy trained ONLY on the 4 Qwen/Gemma small-tier models, zero-shot on Llama-2/Mistral | `results/e53_full_comparison.json` |
| `experiments/qwen_gemma/e53_train_qwengemma_deploy.py` | E53 — reverse-E45: a proxy trained ONLY on the 4 Qwen/Gemma small-tier models, zero-shot on Llama-2/Mistral | `results/e53_full_comparison.json` |
| `experiments/qwen_gemma/e53b_eval_on_llama2_mistral_squad.py` | E53B | `results/e53_full_comparison.json` |
| `experiments/qwen_gemma/e53b_original_ridge_squad.py` | E53B | `results/e53_full_comparison.json` |
| `experiments/qwen_gemma/e64_gemma_baserate_reanalysis.py` | E64 — is E45's gemma-2-9b-it zero-shot LOSS a base-rate artefact or a real transfer failure? | `results/e64_gemma_baserate_reanalysis.json` |
| `experiments/qwen_gemma/e71_settwo_lolo_aligned_ridge.py` | E71 — the E70 comparison for the small-tier Qwen/Gemma "set 2" (LOLO proxy + aligned-ridge, ID + OOD) | `results/e71_settwo_lolo_aligned_ridge.json` |
| `experiments/qwen_gemma/e73_settwo_individual.py` | E73 — E72 for the small-tier Qwen/Gemma "set 2": per-model INDIVIDUAL proxy vs ridge vs SEP, ID + OOD | `results/e73_settwo_individual.json` |

## Model-specificity of the transferred signal

| script | experiment | result |
|---|---|---|
| `experiments/model_specificity/e40_model_specificity.py` | E40 — Is the pooled multi-model RIDGE model-SPECIFIC, or only a question-difficulty detector? | `results/e40_model_specificity.json` |
| `experiments/model_specificity/e40b_lto_significance.py` | E40B | `results/e40_model_specificity.json` |
| `experiments/model_specificity/e40c_lto_ceiling.py` | E40C | `results/e40_model_specificity.json` |
| `experiments/model_specificity/e63_lto_deepseek_qwen3_8b.py` | E63 — leave-TWO-out cross-model test: does a `q_resp_only` proxy trained on 6 target LLMs reproduce the SE disagreement of the 2 held-out ones... | `results/e63_lto_deepseek_qwen3_8b.json` |
| `experiments/model_specificity/e63_lto_failure_analysis.py` | E63 — failure analysis of the leave-two-out disagreement predictor | `results/e63_lto_deepseek_qwen3_8b.json` |

## Archived — retracted or superseded

| script | experiment | result |
|---|---|---|
| `experiments/archive/e34_cosine_instability.py` | E34 — After alignment, do the models share the same UNCERTAINTY DIRECTION? (readout agreement) | — |
| `experiments/archive/e34_cutoff_sweep.py` | E34 — After alignment, do the models share the same UNCERTAINTY DIRECTION? (readout agreement) | — |
| `experiments/archive/e34_matched_ceiling.py` | E34 — After alignment, do the models share the same UNCERTAINTY DIRECTION? (readout agreement) | — |
| `experiments/archive/e34_principal_angles.py` | E34 — After alignment, do the models share the same UNCERTAINTY DIRECTION? (readout agreement) | — |
| `experiments/archive/e35_pooling_datasize_sweep.py` | E35 — Can we POOL multiple aligned models' data into one ridge that beats a single-model ridge? | — |
| `experiments/archive/e35_pooling_loo_pilot.py` | E35 — Can we POOL multiple aligned models' data into one ridge that beats a single-model ridge? | — |
| `experiments/archive/e35_pooling_matched_partition.py` | E35 — Can we POOL multiple aligned models' data into one ridge that beats a single-model ridge? | — |
| `experiments/archive/e35_pooling_matched_partition_bestlayer.py` | E35 — Can we POOL multiple aligned models' data into one ridge that beats a single-model ridge? | — |
