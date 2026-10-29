#!/bin/bash
# Local smoke test: train llama and qwen for 1 epoch on 5 Hugging Face math samples.
# Results go to runs/smoke_test/hf_math/<model>/<run name>/.

set -euo pipefail

export HF_DEACTIVATE_ASYNC_LOAD=1
export PYTHONFAULTHANDLER=1

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

for MODEL in llama qwen; do
    echo "===== Training $MODEL on 5 samples ====="
    python -u "${REPO_DIR}/src/model_runner.py" \
        --model "$MODEL" \
        --samples 5 \
        --source hf \
        --ep 1 \
        --gstep 2 \
        --runs-root "${REPO_DIR}/runs/smoke_test"
done

echo "Both models trained successfully."
