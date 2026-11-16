#!/usr/bin/env bash
# Train on a Thunder Compute GPU instance (or any single-GPU Linux machine with CUDA).
# Results: runs/<dataset>/<model>/<date>_ep<EP>_lr<LR>_<n|full>[_TAG]/
#   logs/slurm.out, logs/slurm.err, logs/training.log, logs/training.err, summary.txt
# and every finished run adds a row to runs/index.csv (same layout as CARC and Mac runs).
#
# Usage:  bash scripts/train_thunder.sh <dataset> [both|llama|qwen]
#   e.g.  bash scripts/train_thunder.sh kodcode qwen
#         EP=1 SAMPLES=50 TAG=debug bash scripts/train_thunder.sh competition_math llama
#         RESUME_FROM=runs/kodcode/qwen/<run>/epoch_2 bash scripts/train_thunder.sh kodcode qwen
# No model argument trains both models one after the other, each in its own run folder.
#
# Datasets: competition_math, kodcode, openmathreasoning, opencode,
#           logic_<task> (combined, countdown, knights_knaves, shortest_path)
# Missing data folders are unzipped from zips/ automatically.
#
# Settings (environment): PYTHON (default python3), EP, LR, BSIZE, GSTEP, MAX_LEN,
#   SAMPLES, TAG, RESUME_FROM, CHECKPOINT_EVERY, TRUNCATE_LONG (1 = on, 0 = skip long samples)
#
# The run keeps going after VS Code disconnects only inside tmux or nohup:
#   tmux new -s train            (then run the command; Ctrl+B then D to detach)
#   nohup bash scripts/train_thunder.sh kodcode qwen > /dev/null 2>&1 &

set -euo pipefail

export HF_DEACTIVATE_ASYNC_LOAD=1
export PYTHONFAULTHANDLER=1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
USAGE="usage: bash scripts/train_thunder.sh <dataset> [both|llama|qwen]"
DATASET="${1:?$USAGE}"
MODEL="${2:-both}"

# =============== Dataset Registry ===============
# Add a dataset here: its folder (with train.jsonl and val.jsonl), its zip, columns, max length
DEFAULT_MAX_LEN=4096
case "$DATASET" in
    competition_math)
        DATA_DIR="${ROOT_DIR}/data/competition_math"
        ZIP_NAME=competition_math
        DATA_COLS=(--qcol problem --acol solution)
        ;;
    kodcode)
        DATA_DIR="${ROOT_DIR}/data/kodcode"
        ZIP_NAME=kodcode
        DATA_COLS=(--qcol question --acol solution)
        ;;
    openmathreasoning)
        DATA_DIR="${ROOT_DIR}/data/openmathreasoning"
        ZIP_NAME=openmathreasoning
        DATA_COLS=(--qcol problem --acol expected_answer)
        DEFAULT_MAX_LEN=6000
        ;;
    opencode)
        DATA_DIR="${ROOT_DIR}/data/opencodeinstruct"
        ZIP_NAME=opencodeinstruct
        DATA_COLS=(--qcol input --acol output)
        ;;
    logic_*)
        DATA_DIR="${ROOT_DIR}/data/sft_logic_rg/${DATASET#logic_}"
        ZIP_NAME=sft_logic_rg
        DATA_COLS=(--scol system --qcol user --acol assistant)
        ;;
    *)
        echo "Unknown dataset: $DATASET" >&2
        echo "$USAGE" >&2
        exit 2
        ;;
esac

EP="${EP:-3}"
LR="${LR:-1e-4}"
BSIZE="${BSIZE:-1}"
GSTEP="${GSTEP:-16}"
MAX_LEN="${MAX_LEN:-$DEFAULT_MAX_LEN}"
SAMPLES="${SAMPLES:-}"
TAG="${TAG:-}"
RESUME_FROM="${RESUME_FROM:-}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-0}"
TRUNCATE_LONG="${TRUNCATE_LONG:-1}"
if [[ -n "$RESUME_FROM" && -z "$TAG" ]]; then TAG="resume_$(basename "$RESUME_FROM" | cut -d_ -f1)"; fi
RUNS_ROOT="${ROOT_DIR}/runs"

if [[ "$MODEL" == "both" ]]; then
    if [[ -n "$RESUME_FROM" ]]; then
        echo "RESUME_FROM applies to one model; choose llama or qwen explicitly." >&2
        exit 2
    fi
    failed=0
    for selected_model in llama qwen; do
        echo "===== Starting ${selected_model} on ${DATASET} ====="
        if bash "${ROOT_DIR}/scripts/train_thunder.sh" "$DATASET" "$selected_model"; then
            echo "===== ${selected_model} finished ====="
        else
            echo "===== ${selected_model} failed; continuing to the next model =====" >&2
            failed=1
        fi
    done
    exit "$failed"
fi

if [[ "$MODEL" != "qwen" && "$MODEL" != "llama" ]]; then
    echo "$USAGE" >&2
    exit 2
fi

# =============== Data ===============
if [[ ! -f "${DATA_DIR}/train.jsonl" || ! -f "${DATA_DIR}/val.jsonl" ]]; then
    ZIP_PATH="${ROOT_DIR}/zips/${ZIP_NAME}.zip"
    if [[ -f "$ZIP_PATH" ]]; then
        echo "Unzipping ${ZIP_PATH} into ${ROOT_DIR}/data"
        mkdir -p "${ROOT_DIR}/data"
        unzip -q -o "$ZIP_PATH" -d "${ROOT_DIR}/data"
    fi
fi
for split in train val; do
    if [[ ! -f "${DATA_DIR}/${split}.jsonl" ]]; then
        echo "Missing ${DATA_DIR}/${split}.jsonl" >&2
        exit 1
    fi
done

# =============== Environment Checks ===============
FREE_KIB="$(df -Pk "$ROOT_DIR" | awk 'NR == 2 { print $4 }')"
if [[ "$FREE_KIB" =~ ^[0-9]+$ ]] && (( FREE_KIB < 5 * 1024 * 1024 )); then
    echo "Less than 5 GiB free on the training disk. Free space before starting so checkpoints can be saved." >&2
    exit 1
fi

"$PYTHON_BIN" -c 'import torch, transformers, peft, datasets, accelerate' || {
    echo "Install the training packages:" >&2
    echo "  $PYTHON_BIN -m pip install transformers==5.18.0 peft==0.21.2 datasets==5.1.0 accelerate==1.15.0" >&2
    exit 1
}
"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"' || {
    echo "The selected Python cannot see a CUDA GPU. Check nvidia-smi, or choose another Python with PYTHON=..." >&2
    exit 1
}

# =============== Run Folder ===============
OPTIONAL_ARGS=()
if [[ -n "$SAMPLES" ]]; then OPTIONAL_ARGS+=(--samples "$SAMPLES"); fi
if [[ -n "$TAG" ]]; then OPTIONAL_ARGS+=(--tag "$TAG"); fi
EXTRA_ARGS=()
if [[ -n "$RESUME_FROM" ]]; then EXTRA_ARGS+=(--resume-from "$RESUME_FROM"); fi
if [[ "$TRUNCATE_LONG" == "1" ]]; then EXTRA_ARGS+=(--truncate-long); fi

RUN_DIR="$("$PYTHON_BIN" "${ROOT_DIR}/src/run_paths.py" \
    --runs-root "$RUNS_ROOT" \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --ep "$EP" \
    --lr "$LR" \
    "${OPTIONAL_ARGS[@]+"${OPTIONAL_ARGS[@]}"}")"
mkdir -p "${RUN_DIR}/logs"

# From here on, stdout -> logs/slurm.out and stderr -> logs/slurm.err (still shown on screen)
exec > >(tee -a "${RUN_DIR}/logs/slurm.out") 2> >(tee -a "${RUN_DIR}/logs/slurm.err" >&2)

echo "===== Environment Information ====="
echo "Job ID: none (Thunder Compute)"
echo "Host: $(hostname)"
echo "Python location: $(command -v "$PYTHON_BIN")"
"$PYTHON_BIN" --version
"$PYTHON_BIN" -c 'import torch, transformers, peft; print(f"torch {torch.__version__} | transformers {transformers.__version__} | peft {peft.__version__}")'
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv || echo "nvidia-smi unavailable"
echo "Model: $MODEL | Dataset: $DATASET ($DATA_DIR)"
echo "Batch size $BSIZE; optimizer update every $GSTEP batches | max length: $MAX_LEN tokens (truncate long: $TRUNCATE_LONG)"
echo "Run folder: $RUN_DIR"

# =============== Training ===============
"$PYTHON_BIN" -u "${ROOT_DIR}/src/model_runner.py" \
    --model "$MODEL" \
    --source custom \
    --train-file "${DATA_DIR}/train.jsonl" \
    --val-file "${DATA_DIR}/val.jsonl" \
    "${DATA_COLS[@]}" \
    --ep "$EP" \
    --lr "$LR" \
    --bsize "$BSIZE" \
    --gstep "$GSTEP" \
    --max-len "$MAX_LEN" \
    --device auto \
    --dtype bfloat16 \
    --gradient-checkpointing \
    --checkpoint-every "$CHECKPOINT_EVERY" \
    --dataset-name "$DATASET" \
    --runs-root "$RUNS_ROOT" \
    --outdir "$RUN_DIR" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" \
    "${OPTIONAL_ARGS[@]+"${OPTIONAL_ARGS[@]}"}" \
    > >(tee "${RUN_DIR}/logs/training.log") \
    2> >(tee "${RUN_DIR}/logs/training.err" >&2)

echo "Training finished successfully. Summary: ${RUN_DIR}/summary.txt"
