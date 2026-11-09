#!/usr/bin/env bash
# Rollout validation on CARC: real grpo.main() (vLLM colocate) for G in {4,8,16},
# each once with real env rewards and once with the planted reward.
#
#   array index  0 1 2  -> G=4,8,16 real rewards
#                3 4 5  -> G=4,8,16 planted reward (1 for first completion of each group)
#
# Submit from the repo root:
#   DATASET=/path/to/shortest_path TASK=shortest_path sbatch experiments/rollout_validation/validate.sbatch
#   # one arm only:
#   DATASET=... TASK=... sbatch --array=1 experiments/rollout_validation/validate.sbatch
#
# Each job writes runs/validation/<task>-G<G>-<real|planted>-<jobid>/rollout_validation*.json
# and exits non-zero if any check fails. Summarize all arms afterwards with:
#   python -m grpo.rollout_validation summary runs/validation
#
# Env vars (all optional except DATASET/TASK):
#   MAX_STEPS=2  COMPLETION_TOKENS=4096  CONTEXT_TOKENS=5120  USE_VLLM=1  VENV_PYTHON=.venv/bin/python
#SBATCH --job-name=rollout-validate
#SBATCH --partition=gpu
#SBATCH --account=yzhao010_1531
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=a100:1
#SBATCH --constraint=a100-80gb
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --array=0-5
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
cd "${SLURM_SUBMIT_DIR:?Submit from repo root}"
[[ -f grpo/rollout_validation.py ]] || { echo 'ERROR: submit from the repo root' >&2; exit 2; }
module purge
module load gcc/13.3.0
module load python/3.11.9

PY="${VENV_PYTHON:-$PWD/.venv/bin/python}"
[[ -x "$PY" ]] || { echo "ERROR: Missing Python venv at $PY" >&2; exit 2; }
: "${DATASET:?ERROR: set DATASET to a local file or directory}"
: "${TASK:?ERROR: set TASK (countdown|knights_knaves|shortest_path|math|kodcode|opencodeinstruct)}"
[[ -e "$DATASET" ]] || { echo "ERROR: No local dataset found: $DATASET" >&2; exit 2; }

G_LIST=(4 8 16)
IDX="${SLURM_ARRAY_TASK_ID:-0}"
G="${G_LIST[$((IDX % 3))]}"
if (( IDX >= 3 )); then MODE=planted; MODE_FLAG=(--planted); else MODE=real; MODE_FLAG=(); fi
MAX_STEPS="${MAX_STEPS:-2}"
COMPLETION_TOKENS="${COMPLETION_TOKENS:-4096}"
CONTEXT_TOKENS="${CONTEXT_TOKENS:-5120}"
USE_VLLM="${USE_VLLM:-1}"
JOB="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
OUT="$PWD/runs/validation/${TASK}-G${G}-${MODE}-${JOB}"
[[ ! -e "$OUT" ]] || { echo "ERROR: Output directory already exists: $OUT" >&2; exit 2; }

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PWD/.cache/triton}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR" "$(dirname "$OUT")" logs

echo "HOST=$(hostname) JOB=${JOB}_${IDX} SHA=$(git rev-parse HEAD 2>/dev/null || true)"
echo "TASK=$TASK G=$G MODE=$MODE MAX_STEPS=$MAX_STEPS USE_VLLM=$USE_VLLM"
echo "COMPLETION_TOKENS=$COMPLETION_TOKENS CONTEXT_TOKENS=$CONTEXT_TOKENS"
echo "DATASET=$DATASET OUT=$OUT"
nvidia-smi
"$PY" - <<'PY'
import torch, trl, transformers
assert torch.cuda.is_available(), 'CUDA unavailable: stop before loading model'
p = torch.cuda.get_device_properties(0)
print(f'torch={torch.__version__} trl={trl.__version__} transformers={transformers.__version__} '
      f'GPU={p.name} VRAM_GiB={p.total_memory/2**30:.1f}')
PY

args=(--task "$TASK" --G "$G" --dataset "$DATASET" --max-steps "$MAX_STEPS"
      --completion-tokens "$COMPLETION_TOKENS" --context-tokens "$CONTEXT_TOKENS"
      --report-to none --output-dir "$OUT")
if [[ "$USE_VLLM" == 0 ]]; then
  args+=(--no-vllm)
elif [[ "$USE_VLLM" != 1 ]]; then
  echo 'ERROR: USE_VLLM must be 0 or 1' >&2; exit 2
fi

# Live checks inside grpo.main(), then the same offline checks on what landed on disk.
"$PY" -u -m grpo.rollout_validation live ${MODE_FLAG[@]+"${MODE_FLAG[@]}"} -- "${args[@]}"

# Offline re-check from disk (+ re-score every rollout against the dataset by prompt).
OFF=(check "$OUT" --G "$G" --task "$TASK" --dataset "$DATASET" --max-completion-length "$COMPLETION_TOKENS")
[[ "$MODE" == planted ]] && OFF+=(--planted)
"$PY" -u -m grpo.rollout_validation "${OFF[@]}"
