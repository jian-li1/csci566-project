#!/bin/bash

export HF_DEACTIVATE_ASYNC_LOAD=1
export PYTHONFAULTHANDLER=1

python -u model_runner.py \
    --model llama\
    --samples 20 \
    --source hf\
    --ep 1 \
    --gstep 2