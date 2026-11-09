#!/usr/bin/env bash
# One ablation arm (task, G, seed) on CARC. Submitted by experiments/rollout_ablation/launch.py;
# do not call by hand unless you set every variable below.
#
# Flow:  create shared init adapter if missing (flock) -> train.py (resumes from the
# newest checkpoint, stops ~15 min before the time limit) ->
#   exit 75: resubmit this script (ATTEMPT+1, up to MAX_ATTEMPTS)
#   exit 0 : offline rollout validation -> queue checkpoint evals (experiments/baselines/eval.sbatch)
#
# Required env: RUN_DIR TASK G SEED DATASET INIT_ADAPTER LR COMPLETION_TOKENS CONTEXT_TOKENS
#               and exactly one of EPOCHS (e.g. 2) or MAX_STEPS
# Optional env: ATTEMPT=1 MAX_ATTEMPTS=4 JOB_TIME=47:00:00 USE_VLLM=1 STOP_MARGIN=900
#               EVAL_DATASET=$DATASET EVAL_SPLITS=val:500 EVAL_K=0 NO_EVAL=0 VENV_PYTHON
#SBATCH --job-name=grpo-ablation
#SBATCH --partition=gpu
#SBATCH --account=yzhao010_1531
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=a100:1
#SBATCH --constraint=a100-80gb
#SBATCH --mem=64G
#SBATCH --time=47:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
cd "${SLURM_SUBMIT_DIR:?Submit from repo root}"
[[ -f experiments/rollout_ablation/train.py ]] || { echo 'ERROR: submit from the repo root' >&2; exit 2; }
module purge
module load gcc/13.3.0
module load python/3.11.9

PY="${VENV_PYTHON:-$PWD/.venv/bin/python}"
[[ -x "$PY" ]] || { echo "ERROR: Missing Python venv at $PY" >&2; exit 2; }
for v in RUN_DIR TASK G SEED DATASET INIT_ADAPTER LR COMPLETION_TOKENS CONTEXT_TOKENS; do
  [[ -n "${!v:-}" ]] || { echo "ERROR: $v is not set" >&2; exit 2; }
done
if [[ -n "${EPOCHS:-}" && -n "${MAX_STEPS:-}" ]] || [[ -z "${EPOCHS:-}" && -z "${MAX_STEPS:-}" ]]; then
  echo "ERROR: set exactly one of EPOCHS or MAX_STEPS" >&2; exit 2
fi
ATTEMPT="${ATTEMPT:-1}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-4}"
JOB_TIME="${JOB_TIME:-47:00:00}"
USE_VLLM="${USE_VLLM:-1}"

export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PWD/.cache/triton}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR" "$RUN_DIR" logs
echo "{\"attempt\": $ATTEMPT, \"slurm_job_id\": \"${SLURM_JOB_ID:-}\", \"host\": \"$(hostname)\", \"start\": \"$(date -Is)\"}" >> "$RUN_DIR/jobs.jsonl"

echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-} ATTEMPT=$ATTEMPT/$MAX_ATTEMPTS SHA=$(git rev-parse HEAD 2>/dev/null || true)"
echo "RUN_DIR=$RUN_DIR TASK=$TASK G=$G SEED=$SEED EPOCHS=${EPOCHS:-} MAX_STEPS=${MAX_STEPS:-}"
nvidia-smi

# --- identical starting checkpoint: one shared init adapter (first job creates it) ---
mkdir -p "$(dirname "$INIT_ADAPTER")"
flock "$INIT_ADAPTER.lock" "$PY" -u -m experiments.rollout_ablation.make_init_adapter \
  --out "$INIT_ADAPTER" --if-missing

# --- deadline = this job's end time, so train.py can save and stop before it ---
DEADLINE=""
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  END="$(squeue -h -j "$SLURM_JOB_ID" -o %e 2>/dev/null || true)"
  [[ -n "$END" && "$END" != "N/A" ]] && DEADLINE="$(date -d "$END" +%s 2>/dev/null || true)"
fi
DL_ARGS=()
[[ -n "$DEADLINE" ]] && DL_ARGS=(--deadline-epoch "$DEADLINE" --margin-seconds "${STOP_MARGIN:-900}")
echo "DEADLINE=${DEADLINE:-none} ($(date -d "@${DEADLINE:-0}" 2>/dev/null || true))"

args=(--task "$TASK" --G "$G" --dataset "$DATASET" --seed "$SEED"
      --lr "$LR"
      --completion-tokens "$COMPLETION_TOKENS" --context-tokens "$CONTEXT_TOKENS"
      --init-adapter "$INIT_ADAPTER" --report-to none --output-dir "$RUN_DIR")
if [[ -n "${EPOCHS:-}" ]]; then args+=(--epochs "$EPOCHS"); else args+=(--max-steps "$MAX_STEPS"); fi
[[ "$USE_VLLM" == 0 ]] && args+=(--no-vllm)

set +e
"$PY" -u -m experiments.rollout_ablation.train ${DL_ARGS[@]+"${DL_ARGS[@]}"} -- "${args[@]}"
RC=$?
set -e

if [[ $RC -eq 75 ]]; then
  if (( ATTEMPT < MAX_ATTEMPTS )); then
    NEXT=$(sbatch --parsable --job-name="${SLURM_JOB_NAME:-grpo-ablation}" --time="$JOB_TIME" \
      --export=ALL,ATTEMPT=$((ATTEMPT + 1)) experiments/rollout_ablation/train.sbatch)
    echo "Stopped early; resubmitted as job $NEXT (attempt $((ATTEMPT + 1)))"
    echo "{\"attempt\": $ATTEMPT, \"resubmitted_as\": \"$NEXT\"}" >> "$RUN_DIR/jobs.jsonl"
    exit 0
  fi
  echo "ERROR: still unfinished after $MAX_ATTEMPTS attempts; rerun launch.py submit --resume" >&2
  exit 75
elif [[ $RC -ne 0 ]]; then
  echo "ERROR: training failed (exit $RC); fix and rerun launch.py submit --resume" >&2
  exit $RC
fi

# --- finished: validate rollouts on disk (non-fatal; result lands in the run dir) ---
"$PY" -u -m grpo.rollout_validation check "$RUN_DIR" --G "$G" \
  --max-completion-length "$COMPLETION_TOKENS" || echo "WARNING: rollout validation FAILED"

# --- queue checkpoint evals (every save_steps checkpoint + final step) ---
if [[ "${NO_EVAL:-0}" != 1 ]]; then
  ADAPTERS="$("$PY" - "$RUN_DIR" <<'PY'
import json, sys
from pathlib import Path
run = Path(sys.argv[1])
max_steps = json.loads((run / "status.json").read_text())["global_step"]
steps = sorted(int(p.name.split("-")[1]) for p in run.glob("checkpoint-*") if p.name.split("-")[1].isdigit())
keep = [s for s in steps if s % 50 == 0 or s == max_steps]   # skip time-limit checkpoints
print(" ".join(str(run / f"checkpoint-{s}") for s in keep))
PY
)"
  EVAL_JOB=$(TASK="$TASK" DATASET="${EVAL_DATASET:-$DATASET}" SPLITS="${EVAL_SPLITS:-val:500}" \
    K="${EVAL_K:-0}" ADAPTERS="$ADAPTERS" COMPLETION_TOKENS="$COMPLETION_TOKENS" \
    CONTEXT_TOKENS="$CONTEXT_TOKENS" \
    sbatch --parsable --job-name="eval-$(basename "$RUN_DIR")" experiments/baselines/eval.sbatch)
  echo "Queued checkpoint evals: job $EVAL_JOB"
  echo "{\"eval_job\": \"$EVAL_JOB\"}" >> "$RUN_DIR/jobs.jsonl"
fi
