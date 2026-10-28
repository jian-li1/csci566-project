#!/bin/bash

set -euo pipefail

export HF_DEACTIVATE_ASYNC_LOAD=1
export PYTHONFAULTHANDLER=1

for MODEL in llama qwen; do
    echo "===== Training $MODEL on 20 samples ====="
    python -u ../src/model_runner.py \
        --model "$MODEL" \
        --samples 5 \
        --source hf \
        --ep 1 \
        --gstep 2 \
        --outdir "./checkpoints/test_$MODEL"
done

echo "Both models trained successfully."
