"""
Fixed settings for the GRPO group-size ablation (owner: Mehul).

Everything here is held equal across arms. The only thing that varies between
arms is G (and the seed, across replicates). The task/dataset is chosen from the
baseline evals and passed to the launcher on the command line.

Matching rule (team design, as in the G=4 / G=8 runs of Nov 2026): every arm trains
for EPOCHS passes over the same train file, so each prompt is seen the same number of
times. Steps then differ by G (prompts/step = 64/G): with 1,750 prompts and 2 epochs,
G=4 runs 218 steps, G=8 436, G=16 874. Rollouts, weighted tokens and GPU-hours
therefore differ by arm and are reported; the analysis compares arms at equal epochs,
equal rollouts, equal tokens and equal GPU-hours.

`launch.py --max-steps N` switches to the alternative rule (equal steps and rollouts,
which needs N x 64/G unique prompts for the smallest G).
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# --- the knob --------------------------------------------------------------
G_VALUES = (4, 8, 16)
ROLLOUTS_PER_STEP = 64                    # grpo.grpo.ROLLOUTS_PER_STEP (checked in train.py)

# --- replicates, by stage ----------------------------------------------------
# Stage 1: one seed. Stage 2: two more seeds. Stage 3 = stages 1+2 on a second task.
# Cost seen at 2 epochs / 4096 tokens on RG tasks: G=4 ~19-24 GPU-h, G=8 ~27-46 GPU-h;
# G=16 (874 steps) is roughly double G=8 and will need a resubmit past 47 h.
STAGE_SEEDS = {1: (42,), 2: (43, 44)}

# --- held fixed (passed explicitly to grpo.grpo, never left to its defaults) ----
HPARAMS = {
    "epochs": 2.0,          # default stop rule; None when --max-steps is used
    "max_steps": None,
    "lr": 2e-5,
    "completion_tokens": 4096,
    "context_tokens": 5120,
}
# grpo.grpo constants that must not change between arms; train.py asserts them.
EXPECTED_GRPO_CONSTANTS = {
    "ROLLOUTS_PER_STEP": 64,
    "MICRO_BATCH_COMPLETIONS": 8,
    "GRAD_ACCUM_STEPS": 8,
    "BETA": 0.001,
    "TEMPERATURE": 1.0,
}

# --- identical starting checkpoint -------------------------------------------
# One saved LoRA init, loaded by every arm via --init-adapter (its sha256 goes in
# each run's manifest). LoRA B=0, so the start policy is the base model; this
# pins the A matrices too.
INIT_ADAPTER_SEED = 0
INIT_ADAPTER_DIR = REPO_ROOT / "runs" / "ablation" / "init_adapter" / f"seed{INIT_ADAPTER_SEED}"

# --- where runs live ---------------------------------------------------------
RUN_ROOT = REPO_ROOT / "runs" / "ablation"


def run_dir(task: str, G: int, seed: int, completion_tokens: int | None = None) -> Path:
    """Non-default completion caps get their own arm directory (e.g. G4-L2048-s42)."""
    cap = "" if completion_tokens in (None, HPARAMS["completion_tokens"]) else f"-L{completion_tokens}"
    return RUN_ROOT / task / f"G{G}{cap}-s{seed}"


def steps_for(G: int, n_rows: int, epochs: float | None, max_steps: int | None) -> int:
    """Optimizer steps an arm will run (TRL drops the last partial batch of prompts)."""
    if max_steps:
        return max_steps
    return int((n_rows // (ROLLOUTS_PER_STEP // G)) * epochs)


# --- checkpoint evals (queued automatically when a run finishes) ----------------
# Greedy on a fixed 500-prompt val sample at every saved checkpoint (save_steps=50).
EVAL = {"split": "val", "splits": "val:500", "k": 0}

# --- SLURM ------------------------------------------------------------------
SLURM_TIME = "47:00:00"        # CARC limit is 48 h; the job stops itself ~15 min early
STOP_MARGIN_SECONDS = 15 * 60  # save + exit this long before the deadline
MAX_ATTEMPTS = 4               # automatic resubmits after a time-limit stop
EXIT_NEEDS_RESUME = 75         # train.py exit code when it stopped early (EX_TEMPFAIL)
