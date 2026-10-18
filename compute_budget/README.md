# Compute budget (FLOP-matched)

`compute_budget.py` is one shared file for SFT and GRPO that enforces the
compute budget and keeps the running total. It works with any HF `Trainer`
(TRL `GRPOTrainer` / `SFTTrainer`) or a custom training loop.

## What we agreed

- **Each model has its own budget.** SFT, GRPO and SFT→GRPO on the same model
  all spend exactly that budget. Comparing across models just means separate
  rows in the results table.
- **The budget is in FLOPs.** N (params) is the same for every run on a model,
  so we drop it and count **weighted tokens** (FLOPs / N):

  | | weight per token |
  |---|---|
  | SFT | 6 × tokens processed (prompt tokens count even if loss-masked) |
  | GRPO, per rollout token (prompt + completion) | generation 2 + policy update 6 × μ, plus 2 more if μ > 1 (old logprobs) and 2 more if β > 0 (reference model) |
  | LoRA | use 4 instead of 6 for the training part |

  With our settings (LoRA, μ = 1, β = 0.001), that's **4 per SFT token** and
  **8 per GRPO rollout token**.
- **SFT→GRPO uses a fixed split: 25% SFT / 75% GRPO**, for every model. The SFT
  stage is its own short run, not a mid-checkpoint of the full SFT run.
- **GPU-hours are logged for reporting only.** We don't stop on them, because
  wall-clock time would be unfair to GRPO (generation speed depends on setup).

## Hook it up

**GRPO / HF-Trainer SFT.** Mode, μ (`num_iterations`) and β are read from the
trainer args, so the hookup is three lines:
```python
from compute_budget import ComputeBudgetCallback
budget_cb = ComputeBudgetCallback(output_dir, budget=B)   # B = this model's budget, in weighted tokens
trainer = GRPOTrainer(..., callbacks=[token_cb, budget_cb])
```
For SFT with the HF Trainer, also set `include_num_input_tokens_seen=True` in
`TrainingArguments`, so token counts are available.

**SFT→GRPO arm.** Pass the same `B`, and the stage takes its share:
```python
ComputeBudgetCallback(sft_dir,  budget=B, stage="sft_then_grpo:sft")    # stops at 25% of B
ComputeBudgetCallback(grpo_dir, budget=B, stage="sft_then_grpo:grpo")   # stops at 75% of B
```

**Custom loop (e.g. `training_loops.py`):**
```python
from compute_budget import ComputeBudget
tracker = ComputeBudget(output_dir, mode="sft", budget=B, lora=True)
# after each optimizer step, count every token in the batch, prompt included:
if tracker.step(tokens=batch["input_ids"].numel()):
    save_checkpoint(); break
```

**Full fine-tuning:** pass `lora=False`.

**Table of all runs:**
```
python compute_budget.py GRPO/* SFT/*
```

## What it writes

`<output_dir>/compute.json` is updated every step. It holds the weighted tokens,
raw tokens, steps, and whether the run is over budget, plus GPU type, GPU-hours,
wall-clock time and peak memory for reporting. A run resumed from the same
`output_dir` continues its total. When the budget is reached, it saves a
checkpoint and stops. W&B gets `budget/*` metrics.

`budget=None` tracks without stopping.

## Open

- The value of B for each model (Qwen3.5-4B, Llama-3.2-3B).

## Status

Tested with simulated SFT and GRPO runs. **Not yet tested on a GPU.** A short
Qwen3.5-0.8B GRPO test on CARC is next.
