#!/bin/bash
# Train one model on one dataset. Results go to
#   runs/<dataset>/<model>/<date>_ep<EP>_lr<LR>_<n|full>[_TAG]_j<jobid>/
# and every finished run adds a row to runs/index.csv.
#
# Usage:  sbatch scripts/train.sh <llama|qwen> <dataset>
#   e.g.  sbatch scripts/train.sh qwen competition_math
#         EP=5 LR=5e-5 sbatch scripts/train.sh llama logic_countdown
#         SAMPLES=50 TAG=debug sbatch -J debug scripts/train.sh qwen competition_math
#
# Datasets: competition_math, opencode,
#           logic_<task> (combined, countdown, knights_knaves, shortest_path)
# Settings (environment, defaults below): EP, LR, BSIZE, GSTEP, MAX_LEN, SAMPLES, TAG

#SBATCH --job-name=sft
#SBATCH --partition=gpu
#SBATCH --account=yzhao010_1531
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00

# =============== GPU Allocation ===============
#SBATCH --gpus-per-task=a100:1
#SBATCH --constraint=a100-80gb

# =============== Slurm Logs (fallback, before RUN_DIR exists) ===============
#SBATCH -o /home1/chunyint/cs566/logs/%x_%j.out
#SBATCH -e /home1/chunyint/cs566/logs/%x_%j.err

# Exit on errors, including failures piped through tee
set -euo pipefail

USAGE="usage: sbatch scripts/train.sh <llama|qwen> <dataset>"
MODEL="${1:?$USAGE}"
DATASET="${2:?$USAGE}"

EP="${EP:-3}"
LR="${LR:-1e-4}"
BSIZE="${BSIZE:-1}"
GSTEP="${GSTEP:-16}"
MAX_LEN="${MAX_LEN:-4096}"
SAMPLES="${SAMPLES:-}"
TAG="${TAG:-}"

ROOT_DIR="/home1/chunyint/cs566"
RUNS_ROOT="${ROOT_DIR}/runs"

# =============== Dataset Registry ===============
# Add a dataset here: its folder (with train.jsonl and val.jsonl) and columns
case "$DATASET" in
    competition_math)
        DATA_DIR="${ROOT_DIR}/data/competition_math"
        COLUMNS=(--qcol problem --acol solution)
        ;;
    logic_*)
        DATA_DIR="${ROOT_DIR}/data/sft_logic_rg/${DATASET#logic_}"
        COLUMNS=(--scol system --qcol user --acol assistant)
        ;;
    opencode)
        DATA_DIR="${ROOT_DIR}/data/opencode/opencodeinstruct"
        COLUMNS=(--qcol input --acol output)
        ;;
    *)
        echo "Unknown dataset: $DATASET" >&2
        exit 1
        ;;
esac

for split in train val; do
    if [ ! -f "${DATA_DIR}/${split}.jsonl" ]; then
        echo "Missing ${DATA_DIR}/${split}.jsonl" >&2
        exit 1
    fi
done

module purge

# =============== Conda Activation ===============
eval "$(conda shell.bash hook)"
conda activate venv_1

export HF_DEACTIVATE_ASYNC_LOAD=1
export PYTHONFAULTHANDLER=1

# =============== Run Folder ===============
OPTIONAL_ARGS=()
if [ -n "$SAMPLES" ]; then OPTIONAL_ARGS+=(--samples "$SAMPLES"); fi
if [ -n "$TAG" ]; then OPTIONAL_ARGS+=(--tag "$TAG"); fi

RUN_DIR="$(python "${ROOT_DIR}/src/run_paths.py" \
    --runs-root "$RUNS_ROOT" \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --ep "$EP" \
    --lr "$LR" \
    "${OPTIONAL_ARGS[@]}")"
mkdir -p "${RUN_DIR}/logs"

# From here on, stdout -> logs/slurm.out and stderr -> logs/slurm.err
# (tee also keeps a copy in the fallback Slurm files above)
exec > >(tee -a "${RUN_DIR}/logs/slurm.out") 2> >(tee -a "${RUN_DIR}/logs/slurm.err" >&2)

echo "===== Environment Information ====="
echo "Job ID: ${SLURM_JOB_ID:-none}"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Python location: $(which python)"
python --version
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

echo "Model: $MODEL | Dataset: $DATASET ($DATA_DIR)"
echo "Run folder: $RUN_DIR"

# =============== Training ===============
python -u "${ROOT_DIR}/src/model_runner.py" \
    --model "$MODEL" \
    --source custom \
    --train-file "${DATA_DIR}/train.jsonl" \
    --val-file "${DATA_DIR}/val.jsonl" \
    "${COLUMNS[@]}" \
    --ep "$EP" \
    --bsize "$BSIZE" \
    --gstep "$GSTEP" \
    --lr "$LR" \
    --max-len "$MAX_LEN" \
    --dataset-name "$DATASET" \
    --runs-root "$RUNS_ROOT" \
    --outdir "$RUN_DIR" \
    "${OPTIONAL_ARGS[@]}" \
    > >(tee "${RUN_DIR}/logs/training.log") \
    2> >(tee "${RUN_DIR}/logs/training.err" >&2)

echo "Training finished successfully. Summary: ${RUN_DIR}/summary.txt"
