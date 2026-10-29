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
#SBATCH --mem=24G
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

# =============== Dataset / Runs ===============
ROOT_DIR="/home1/chunyint/cs566"
# Kept apart from real runs, with its own index.csv
RUNS_ROOT="${ROOT_DIR}/runs/smoke_test"
DATA_DIR="${ROOT_DIR}/data/competition_math"

# Auto split of the first SAMPLES rows of train.jsonl (80/20 -> 8 train, 2 val)
SAMPLES=10
EP=1
LR=1e-4

echo "===== Environment Information ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Python location: $(which python)"
python --version
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# =============== Training ===============
for MODEL in llama qwen; do
    RUN_DIR="$(python "${ROOT_DIR}/src/run_paths.py" \
        --runs-root "$RUNS_ROOT" \
        --dataset competition_math \
        --model "$MODEL" \
        --ep "$EP" \
        --lr "$LR" \
        --samples "$SAMPLES")"
    mkdir -p "${RUN_DIR}/logs"

    echo "===== Training $MODEL on $SAMPLES samples -> $RUN_DIR ====="
    python -u "${ROOT_DIR}/src/model_runner.py" \
        --model "$MODEL" \
        --samples "$SAMPLES" \
        --source custom \
        --dataset-file "${DATA_DIR}/train.jsonl" \
        --qcol "problem" \
        --acol "solution" \
        --ep "$EP" \
        --bsize 1 \
        --gstep 2 \
        --lr "$LR" \
        --dataset-name competition_math \
        --runs-root "$RUNS_ROOT" \
        --outdir "$RUN_DIR" \
        > >(tee "${RUN_DIR}/logs/training.log") \
        2> >(tee "${RUN_DIR}/logs/training.err" >&2)
    echo "===== $MODEL finished: ${RUN_DIR}/summary.txt ====="
done

echo "Smoke test finished successfully for llama and qwen."
