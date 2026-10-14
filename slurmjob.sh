#!/bin/bash

#SBATCH --job-name=gemma4_test
#SBATCH -p gpu
#SBATCH --account=yzhao010_1531
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH -o /home1/chunyint/cs566/gemma_%j.out
#SBATCH -e /home1/chunyint/cs566/gemma_%j.err

module purge

eval "$(conda shell.bash hook)"
conda activate venv_1

# Replace with your dir !!!
cd /home1/chunyint/cs566

echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "Python location:"
which python
python --version

RUN_DIR="./logs/run_$(date +%Y%m%d_%H%M%S)"
DATA_DIR="./data/competition_math"

mkdir -p "$RUN_DIR"

python -u model_runner.py \
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
    2>&1 | tee "$RUN_DIR/training.log"