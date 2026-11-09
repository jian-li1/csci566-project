#!/usr/bin/env bash
#SBATCH --job-name=grpo-qwen35
#SBATCH --partition=gpu
#SBATCH --account=yzhao010_1531
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=a100:1
#SBATCH --constraint=a100-80gb
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT,INVALID_DEPEND
#SBATCH --mail-user=mehulsin@usc.edu
# Uses the account shown by `myaccount` as default; explicitly add --account=YOUR_ID if needed.
# Slurm reads #SBATCH before the shell. Override --time on sbatch CLI for long runs.

set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
cd "${SLURM_SUBMIT_DIR:?Submit from repo root}"
[[ -f grpo/grpo.py ]] || { echo 'ERROR: wrong submit directory' >&2; exit 2; }
module purge
module load gcc/13.3.0
module load python/3.11.9

PY="${VENV_PYTHON:-$PWD/.venv/bin/python}"
[[ -x "$PY" ]] || { echo "ERROR: Missing Python venv at $PY" >&2; exit 2; }
: "${DATASET:?ERROR: DATASET must point to an existing local file or directory}"
[[ -e "$DATASET" ]] || { echo "ERROR: No local dataset found: $DATASET" >&2; exit 2; }
TASK="${TASK:-countdown}"
G="${G:-4}"
MAX_STEPS="${MAX_STEPS:-1}"
USE_VLLM="${USE_VLLM:-0}"
RUN_TAG="${RUN_TAG:-smoke}"
OUT="${OUT:-$PWD/runs/${TASK}-G${G}-${RUN_TAG}-${SLURM_JOB_ID}}"
[[ ! -e "$OUT" ]] || { echo "ERROR: Output directory already exists: $OUT" >&2; exit 2; }
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PWD/.cache/triton}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR" "$(dirname "$OUT")"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID} SHA=$(git rev-parse HEAD 2>/dev/null || true)"
echo "TASK=$TASK G=$G MAX_STEPS=$MAX_STEPS USE_VLLM=$USE_VLLM"
echo "DATASET=$DATASET OUT=$OUT"
nvidia-smi
"$PY" - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA unavailable: stop before loading model'
p=torch.cuda.get_device_properties(0)
print(f'PyTorch={torch.__version__}; CUDA={torch.version.cuda}; GPU={p.name}; VRAM_GiB={p.total_memory/2**30:.1f}')
assert p.total_memory/2**30 > 70, 'Expected one 80GB GPU allocation'
PY
"$PY" -m pip check
args=(--task "$TASK" --G "$G" --dataset "$DATASET" --max-steps "$MAX_STEPS" --report-to none --output-dir "$OUT")
if [[ "$USE_VLLM" == 0 ]]; then
  args+=(--no-vllm)
elif [[ "$USE_VLLM" == 1 ]]; then
  "$PY" -c 'import vllm, bitsandbytes; print("vLLM and bitsandbytes imports PASS")'
else
  echo 'ERROR: USE_VLLM must be 0 or 1' >&2; exit 2
fi
"$PY" -u -m grpo.grpo "${args[@]}"
"$PY" - "$OUT" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); roll=p/'rollouts.jsonl'
assert roll.is_file() and roll.stat().st_size, f'Missing or empty {roll}'
rows=[json.loads(s) for s in roll.read_text().splitlines() if s.strip()]
assert rows and all(row.get('reward') is not None for row in rows), 'unscored rollouts detected'
assert (p/'adapter_config.json').is_file(), 'LoRA adapter missing'
print('POSTFLIGHT PASS:',len(rows),'rollouts; adapter present; output:',p)
for extra in ('completion_tokens.json','compute.json'):
    print(extra, 'present' if (p/extra).is_file() else 'not found (inspect metrics/compute callback)')
PY
"$PY" -m grpo.metrics "$OUT"
