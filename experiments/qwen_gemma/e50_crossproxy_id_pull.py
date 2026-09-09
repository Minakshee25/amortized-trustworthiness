"""E50 — fill two gaps directly: how does (a) the Mistral-ONLY-trained proxy
and (b) the Deploy (all-4-pooled) proxy score on trivia_qa ID (the SAME N=1000 fresh eval set used
throughout correctness_eval.py), for BOTH Llama-2 and Mistral. Neither had been run on trivia_qa ID
before (only squad/OOD existed for both, in E42 and E39 respectively). Additive: reuses `arm_preds`
(unchanged) with a different `ckpt_dir`, and reuses `load_accuracy` for the incorrect label -- no SEP
refit, no hidden-state loading beyond what arm_preds itself needs.

    python -m experiments.qwen_gemma.e50_crossproxy_id_pull
"""
from __future__ import annotations
import os
from amortized_ue.paths import DATA_DIR, DEPLOY_CKPTS_DIR, RESULTS_DIR, ckpt_dir
import json
import numpy as np
from sklearn.metrics import roc_auc_score

from amortized_ue.config import Stage1Config
from amortized_ue.eval.correctness import load_accuracy

THRESH = 0.5  # matches correctness_eval.py's evaluate_target() binarisation exactly
from amortized_ue.eval.rank_fusion import arm_preds

EVAL_N = 1000
DATASET = "trivia_qa"

CKPTS = {
    "trained_on_llama2_only (REFERENCE)": ckpt_dir("REFERENCE_multipos_p1024_5arm_ckpt", "checkpoints"),
    "trained_on_mistral_only (E22)": ckpt_dir("E22_Mistral_proxy_p1024_5arm_ckpt", "checkpoints"),
    "trained_on_all_4_pooled (deploy)": DEPLOY_CKPTS_DIR,
}

TARGETS = ["Llama-2-7b-chat", "Mistral-7B-Instruct-v0.2"]


def main():
    results = {}
    for target in TARGETS:
        acc_cfg = Stage1Config(model_name=target, dataset=DATASET, num_samples=EVAL_N, output_dir=DATA_DIR)
        acc_map = load_accuracy(acc_cfg)
        eval_ids = sorted(acc_map.keys())
        incorrect = (np.array([acc_map[i] for i in eval_ids]) < THRESH).astype(int)
        print(f"\n=== target={target}  N={len(eval_ids)}  incorrect_rate={incorrect.mean():.3f} ===")

        results[target] = {}
        for label, ckpt in CKPTS.items():
            preds = arm_preds("q_resp_only", target, DATASET, EVAL_N, ckpt_dir=ckpt, data_dir=DATA_DIR)
            s = np.array([preds[i] for i in eval_ids])
            auroc = float(roc_auc_score(incorrect, s))
            results[target][label] = auroc
            print(f"  {label:38s} AUROC_incorrect={auroc:.3f}")

    out = os.path.join(RESULTS_DIR, "e50_crossproxy_id_pull.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
