#!/bin/bash

#SBATCH --job-name=llama_sft
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
#SBATCH -o /home1/chunyint/cs566/logs/llama_%j.out
#SBATCH -e /home1/chunyint/cs566/logs/llama_%j.err

# Exit on errors, including failures piped through tee
set -euo pipefail

module purge

# =============== Conda Activation ===============
eval "$(conda shell.bash hook)"
conda activate venv_1

# ... (environment / GPU info unchanged) ...

# =============== Dataset / Run Folder ===============
ROOT_DIR="/home1/chunyint/cs566"
RUNS_ROOT="${ROOT_DIR}/runs"
DATASET="competition_math"
DATA_DIR="${ROOT_DIR}/data/${DATASET}"
MODEL="llama"
EP=3
LR=1e-4

# runs/<dataset>/<model>/<date>_ep<EP>_lr<LR>_full_j<jobid>/
RUN_DIR="$(python "${ROOT_DIR}/src/run_paths.py" \
    --runs-root "$RUNS_ROOT" \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --ep "$EP" \
    --lr "$LR")"
mkdir -p "${RUN_DIR}/logs"

# From here on, stdout -> logs/slurm.out and stderr -> logs/slurm.err
# (tee also keeps a copy in the fallback Slurm files above)
exec > >(tee -a "${RUN_DIR}/logs/slurm.out") 2> >(tee -a "${RUN_DIR}/logs/slurm.err" >&2)

echo "Run folder: $RUN_DIR"
echo "Training logs: $RUN_DIR/logs/training.log"
echo "Errors: $RUN_DIR/logs/training.err"
echo "Checkpoints: $RUN_DIR/checkpoints"

# =============== Training ===============
python -u "${ROOT_DIR}/src/model_runner.py" \
    --model "$MODEL" \
    --source custom \
    --train-file "${DATA_DIR}/train.jsonl" \
    --val-file "${DATA_DIR}/val.jsonl" \
    --qcol "problem" \
    --acol "solution" \
    --ep "$EP" \
    --bsize 1 \
    --gstep 16 \
    --lr "$LR" \
    --dataset-name "$DATASET" \
    --runs-root "$RUNS_ROOT" \
    --outdir "$RUN_DIR" \
    > >(tee "${RUN_DIR}/logs/training.log") \
    2> >(tee "${RUN_DIR}/logs/training.err" >&2)

echo "Training finished successfully. Summary: ${RUN_DIR}/summary.txt"
