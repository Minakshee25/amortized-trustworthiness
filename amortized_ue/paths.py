"""Canonical filesystem locations, resolved from the repository root.

Every script writes results and reads shared artefacts through these constants so
that behaviour does not depend on the current working directory.
"""
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
LOGS_DIR = os.path.join(REPO_ROOT, "logs")
DATA_DIR = os.getenv("AMORTIZED_UE_DATA", os.path.join(REPO_ROOT, "data", "stage1"))
RUNS_DIR = os.path.join(REPO_ROOT, "amortized_ue", "stage2", "runs")

# Base directory for saved proxy checkpoints. Trained runs write under RUNS_DIR;
# a staged/off-NFS copy (and the canonical location on this project's machine) is
# given by $AMORTIZED_UE_CKPTS. The checkpoints are also mirrored to Weights & Biases.
CKPTS_DIR = os.getenv("AMORTIZED_UE_CKPTS", RUNS_DIR)

# The shared "deploy" proxy (q_resp_only, trained on the 4 alignment targets) used by
# the Qwen/Gemma zero-shot and RQ1 latency experiments.
DEPLOY_CKPTS_DIR = os.path.join(CKPTS_DIR, "deploy_checkpoints")


def ckpt_dir(run_name, *sub):
    """Resolve a proxy checkpoint directory for ``run_name``.

    Tries ``$AMORTIZED_UE_CKPTS/<run_name>/<sub...>`` then
    ``<repo>/amortized_ue/stage2/runs/<run_name>/<sub...>``; returns the first that
    exists, else the ``$AMORTIZED_UE_CKPTS`` path so error messages point at the
    configured location.
    """
    bases = []
    env = os.getenv("AMORTIZED_UE_CKPTS")
    if env:
        bases.append(env)
    bases.append(RUNS_DIR)
    for base in bases:
        p = os.path.join(base, run_name, *sub)
        if os.path.isdir(p):
            return p
    return os.path.join(bases[0], run_name, *sub)
