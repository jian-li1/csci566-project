#!/usr/bin/env bash
# Baseline / checkpoint evaluation on CARC with grpo/evaluate.py (vLLM, same prompts,
# scorer and token caps as training).
#
# Submit from the repo root. Recommended baselines per task (two jobs):
#
#   # 1) solve rates + buckets + G predictions: k=16 on fixed 500-prompt samples
#   TASK=math DATASET=data_splits/competition_math SPLITS=val:500,train:500 K=16 \
#     sbatch experiments/baselines/eval.sbatch
#   # 2) headline greedy accuracy on the full val/test splits
#   TASK=math DATASET=data_splits/competition_math SPLITS=val,test K=0 \
#     sbatch experiments/baselines/eval.sbatch
#
# Checkpoints (one vLLM start-up, weights swapped per adapter):
#   TASK=shortest_path DATASET=... SPLITS=val:500 K=0 \
#     ADAPTERS="runs/X/checkpoint-50 runs/X/checkpoint-100" sbatch experiments/baselines/eval.sbatch
#
# Rough cost: ~(prompts x (K+1) x mean completion length) tokens; at 4096-token caps
# 500 prompts x 17 completions is a few hours on one A100. Raise --time for full splits with K>0.
#
# Env vars: TASK, DATASET (required); SPLITS=val; K=16; ADAPTERS="none";
#   COMPLETION_TOKENS=4096; CONTEXT_TOKENS=5120; SEED=0; SAMPLE_SEED=0;
#   SAVE_COMPLETIONS=greedy; MODEL_ID (default grpo.grpo.MODEL_ID); EXTRA_ARGS; VENV_PYTHON
#SBATCH --job-name=grpo-eval
#SBATCH --partition=gpu
#SBATCH --account=yzhao010_1531
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=a100:1
#SBATCH --constraint=a100-80gb
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
cd "${SLURM_SUBMIT_DIR:?Submit from repo root}"
[[ -f grpo/evaluate.py ]] || { echo 'ERROR: submit from the repo root' >&2; exit 2; }
module purge
module load gcc/13.3.0
module load python/3.11.9

PY="${VENV_PYTHON:-$PWD/.venv/bin/python}"
[[ -x "$PY" ]] || { echo "ERROR: Missing Python venv at $PY" >&2; exit 2; }
: "${TASK:?ERROR: set TASK (countdown|knights_knaves|shortest_path|math|kodcode|opencodeinstruct)}"
: "${DATASET:?ERROR: set DATASET}"
[[ -e "$DATASET" ]] || { echo "ERROR: No local dataset found: $DATASET" >&2; exit 2; }

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PWD/.cache/triton}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR" logs

args=(--task "$TASK" --dataset "$DATASET" --splits "${SPLITS:-val}" --k "${K:-16}"
      --completion-tokens "${COMPLETION_TOKENS:-4096}" --context-tokens "${CONTEXT_TOKENS:-5120}"
      --seed "${SEED:-0}" --sample-seed "${SAMPLE_SEED:-0}"
      --save-completions "${SAVE_COMPLETIONS:-greedy}" --backend vllm)
[[ -n "${MODEL_ID:-}" ]] && args+=(--model-id "$MODEL_ID")
for a in ${ADAPTERS:-none}; do args+=(--adapter "$a"); done
# shellcheck disable=SC2206
[[ -n "${EXTRA_ARGS:-}" ]] && args+=(${EXTRA_ARGS})

echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-local} SHA=$(git rev-parse HEAD 2>/dev/null || true)"
echo "ARGS: ${args[*]}"
nvidia-smi
"$PY" -u -m grpo.evaluate "${args[@]}"
