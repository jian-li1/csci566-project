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

# =============== Slurm Logs ===============
#SBATCH -o /home1/chunyint/cs566/llama_%j.out
#SBATCH -e /home1/chunyint/cs566/llama_%j.err

# Exit on errors, including failures piped through tee
set -euo pipefail

module purge

# =============== Conda Activation ===============
eval "$(.conda shell.bash hook)"
.conda activate venv_1

# =============== Working Directory ===============
cd /home1/chunyint/cs566

echo "===== Environment Information ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Python location: $(which python)"
python --version

# =============== GPU Information ===============
echo "===== GPU Information ====="
nvidia-smi

nvidia-smi \
    --query-gpu=name,memory.total,memory.free \
    --format=csv

# =============== Dataset / Logs ===============
RUN_DIR="./logs/run_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}"
DATA_DIR="./data/competition_math"

mkdir -p "$RUN_DIR"

echo "Training logs: $RUN_DIR/training.log"
echo "Checkpoints: $RUN_DIR/checkpoints"

# =============== Training ===============
python -u model_runner.py \
    --model llama\
    --samples 50 \
    --source custom \
    --train-file "$DATA_DIR/train.jsonl" \
    --val-file "$DATA_DIR/val.jsonl" \
    --qcol "problem" \
    --acol "solution" \
    --ep 5 \
    --bsize 1 \
    --gstep 16 \
    --lr 1e-4 \
    --outdir "$RUN_DIR/checkpoints" \
    2>&1 | tee "$RUN_DIR/training.log"

echo "Training finished successfully."
