#!/usr/bin/env bash
# Train KodCode on an Apple Silicon Mac.
# Results: runs/kodcode/<model>/<date>_ep<EP>_lr<LR>_<n|full>[_TAG]/
# Usage: bash scripts/train_mac_kodcode.sh [both|llama|qwen]
# No argument trains both models sequentially, each in its own run folder.
# Example: SAMPLES=50 EP=1 bash scripts/train_mac_kodcode.sh both
# Settings: PYTHON (default python3), EP, LR, MAX_LEN, SAMPLES, TAG, RESUME_FROM.
# Each batch contains one question; the optimizer updates every 16 batches.

set -euo pipefail

# Transformers 5.x otherwise copies weights to MPS from several threads.
# Sequential loading avoids a crash in PyTorch's Metal weight-copy path.
export HF_DEACTIVATE_ASYNC_LOAD=1
export PYTHONFAULTHANDLER=1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
MODEL="${1:-both}"
EP="${EP:-3}"
LR="${LR:-1e-4}"
MAX_LEN="${MAX_LEN:-4096}"
SAMPLES="${SAMPLES:-}"
TAG="${TAG:-}"
RESUME_FROM="${RESUME_FROM:-}"
if [[ -n "$RESUME_FROM" && -z "$TAG" ]]; then TAG="resume_$(basename "$RESUME_FROM" | cut -d_ -f1)"; fi
DATA_DIR="${ROOT_DIR}/data/kodcode"
RUNS_ROOT="${ROOT_DIR}/runs"

if [[ "$MODEL" == "both" ]]; then
    if [[ -n "$RESUME_FROM" ]]; then
        echo "RESUME_FROM applies to one model; choose llama or qwen explicitly." >&2
        exit 2
    fi
    failed=0
    for selected_model in llama qwen; do
        echo "===== Starting ${selected_model} on KodCode ====="
        if bash "${ROOT_DIR}/scripts/train_mac_kodcode.sh" "$selected_model"; then
            echo "===== ${selected_model} finished ====="
        else
            echo "===== ${selected_model} failed; continuing to the next model =====" >&2
            failed=1
        fi
    done
    exit "$failed"
fi

if [[ "$MODEL" != "qwen" && "$MODEL" != "llama" ]]; then
    echo "Usage: bash scripts/train_mac_kodcode.sh [both|llama|qwen]" >&2
    exit 2
fi

for split in train val; do
    if [[ ! -f "${DATA_DIR}/${split}.jsonl" ]]; then
        echo "Missing ${DATA_DIR}/${split}.jsonl" >&2
        exit 1
    fi
done

FREE_KIB="$(df -Pk "$ROOT_DIR" | awk 'NR == 2 { print $4 }')"
if [[ "$FREE_KIB" =~ ^[0-9]+$ ]] && (( FREE_KIB < 5 * 1024 * 1024 )); then
    echo "Less than 5 GiB free on the training disk. Free space before starting so the next checkpoint can finish." >&2
    exit 1
fi

"$PYTHON_BIN" -c 'import torch, transformers, peft, datasets, accelerate' || {
    echo "Install the Mac requirements: $PYTHON_BIN -m pip install -r ${ROOT_DIR}/requirements-mac.txt" >&2
    exit 1
}
"$PYTHON_BIN" -c 'import torch; assert torch.backends.mps.is_available(), "PyTorch MPS is unavailable"' || {
    echo "The selected Python cannot access Apple MPS. Choose a Mac PyTorch environment with PYTHON=..." >&2
    exit 1
}

OPTIONAL_ARGS=()
if [[ -n "$SAMPLES" ]]; then OPTIONAL_ARGS+=(--samples "$SAMPLES"); fi
if [[ -n "$TAG" ]]; then OPTIONAL_ARGS+=(--tag "$TAG"); fi
RESUME_ARGS=()
if [[ -n "$RESUME_FROM" ]]; then RESUME_ARGS+=(--resume-from "$RESUME_FROM"); fi

RUN_DIR="$("$PYTHON_BIN" "${ROOT_DIR}/src/run_paths.py" \
    --runs-root "$RUNS_ROOT" \
    --dataset kodcode \
    --model "$MODEL" \
    --ep "$EP" \
    --lr "$LR" \
    "${OPTIONAL_ARGS[@]+"${OPTIONAL_ARGS[@]}"}")"
mkdir -p "${RUN_DIR}/logs"

# Match the CARC run log layout without /dev/fd process substitution.
: > "${RUN_DIR}/logs/slurm.out"
: > "${RUN_DIR}/logs/slurm.err"
{
    echo "===== Environment Information ====="
    echo "Job ID: none (local run)"
    echo "Conda environment: ${CONDA_DEFAULT_ENV:-none}"
    echo "Python location: $(command -v "$PYTHON_BIN")"
    "$PYTHON_BIN" --version
    echo "GPU: Apple MPS"
    echo "Model: $MODEL | KodCode | MPS"
    echo "One question per batch; optimizer update every 16 batches | max length: $MAX_LEN tokens (long answers truncated)"
    echo "Run folder: $RUN_DIR"
} | tee -a "${RUN_DIR}/logs/slurm.out"

if "$PYTHON_BIN" -u "${ROOT_DIR}/src/model_runner.py" \
    --model "$MODEL" \
    --source custom \
    --train-file "${DATA_DIR}/train.jsonl" \
    --val-file "${DATA_DIR}/val.jsonl" \
    --qcol question \
    --acol solution \
    --ep "$EP" \
    --lr "$LR" \
    --bsize 1 \
    --gstep 16 \
    --max-len "$MAX_LEN" \
    --truncate-long \
    --device mps \
    --dtype bfloat16 \
    --gradient-checkpointing \
    --checkpoint-every 0 \
    --dataset-name kodcode \
    --runs-root "$RUNS_ROOT" \
    --outdir "$RUN_DIR" \
    "${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}" \
    "${OPTIONAL_ARGS[@]+"${OPTIONAL_ARGS[@]}"}" \
    2> "${RUN_DIR}/logs/training.err" \
    | tee "${RUN_DIR}/logs/training.log" \
    | tee -a "${RUN_DIR}/logs/slurm.out"; then
    training_status=0
else
    training_status=$?
fi

tee -a "${RUN_DIR}/logs/slurm.err" < "${RUN_DIR}/logs/training.err" >&2
if [[ "$training_status" -ne 0 ]]; then exit "$training_status"; fi

echo "Training finished successfully. Summary: ${RUN_DIR}/summary.txt" | tee -a "${RUN_DIR}/logs/slurm.out"
