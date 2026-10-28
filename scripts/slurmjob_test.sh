#!/bin/bash
# Quick CARC smoke test: train llama and qwen for 1 epoch on a handful of
# competition_math samples to check the pipeline end to end.
# Submit with: sbatch scripts/slurmjob_test.sh

#SBATCH --job-name=sft_smoke_test
#SBATCH --partition=gpu
#SBATCH --account=yzhao010_1531
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=0:45:00

# =============== GPU Allocation ===============
#SBATCH --gpus-per-task=a40:1

# =============== Slurm Logs (fallback, before RUN_DIR exists) ===============
#SBATCH -o /home1/chunyint/cs566/logs/test_%j.out
#SBATCH -e /home1/chunyint/cs566/logs/test_%j.err

# Exit on errors, including failures piped through tee
set -euo pipefail

module purge

# =============== Conda Activation ===============
eval "$(conda shell.bash hook)"
conda activate venv_1

export HF_DEACTIVATE_ASYNC_LOAD=1
export PYTHONFAULTHANDLER=1

# =============== Dataset / Logs ===============
ROOT_DIR="/home1/chunyint/cs566"
RUN_DIR="${ROOT_DIR}/logs/test/run_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}"
DATA_DIR="${ROOT_DIR}/data/competition_math"

# Auto split of the first SAMPLES rows of train.jsonl (80/20 -> 8 train, 2 val)
SAMPLES=10

mkdir -p "$RUN_DIR"

# From here on, stdout -> RUN_DIR/slurm.out and stderr -> RUN_DIR/slurm.err
exec > >(tee -a "$RUN_DIR/slurm.out") 2> >(tee -a "$RUN_DIR/slurm.err" >&2)

echo "===== Environment Information ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Python location: $(which python)"
python --version
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

echo "Logs: $RUN_DIR"

# =============== Training ===============
for MODEL in llama qwen; do
    echo "===== Training $MODEL on $SAMPLES samples ====="
    python -u "$ROOT_DIR/src/model_runner.py" \
        --model "$MODEL" \
        --samples "$SAMPLES" \
        --source custom \
        --dataset-file "$DATA_DIR/train.jsonl" \
        --qcol "problem" \
        --acol "solution" \
        --ep 1 \
        --bsize 1 \
        --gstep 2 \
        --lr 1e-4 \
        --outdir "$RUN_DIR/checkpoints" \
        > >(tee "$RUN_DIR/${MODEL}_training.log") \
        2> >(tee "$RUN_DIR/${MODEL}_training.err" >&2)
    echo "===== $MODEL finished ====="
done

echo "Smoke test finished successfully for llama and qwen."
