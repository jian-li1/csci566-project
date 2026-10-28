#!/bin/bash

#SBATCH --job-name=qwen_sft
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
#SBATCH -o /home1/chunyint/cs566/logs/qwen_%j.out
#SBATCH -e /home1/chunyint/cs566/logs/qwen_%j.err

# Exit on errors, including failures piped through tee
set -euo pipefail

module purge

# =============== Conda Activation ===============
eval "$(conda shell.bash hook)"
conda activate venv_1

# ... (environment / GPU info unchanged) ...

# =============== Dataset / Logs ===============
ROOT_DIR="/home1/chunyint/cs566"
RUN_DIR="${ROOT_DIR}/logs/math/run_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}"
DATA_DIR="${ROOT_DIR}/data/competition_math"

mkdir -p "$RUN_DIR"

# From here on, stdout -> RUN_DIR/slurm.out and stderr -> RUN_DIR/slurm.err
# (tee also keeps a copy in the fallback Slurm files above)
exec > >(tee -a "$RUN_DIR/slurm.out") 2> >(tee -a "$RUN_DIR/slurm.err" >&2)

echo "Training logs: $RUN_DIR/training.log"
echo "Errors: $RUN_DIR/training.err"
echo "Checkpoints: $RUN_DIR/checkpoints"

# =============== Training ===============
python -u $ROOT_DIR/src/model_runner.py \
    --model qwen \
    --source custom \
    --train-file "$DATA_DIR/train.jsonl" \
    --val-file "$DATA_DIR/val.jsonl" \
    --qcol "problem" \
    --acol "solution" \
    --ep 3 \
    --bsize 1 \
    --gstep 16 \
    --lr 1e-4 \
    --outdir "$RUN_DIR/checkpoints" \
    > >(tee "$RUN_DIR/training.log") \
    2> >(tee "$RUN_DIR/training.err" >&2)

echo "Training finished successfully."