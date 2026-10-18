"""
Shared compute budget for SFT and GRPO runs (FLOP-matched, per model).

Agreed rules (team, Oct 13):
  * Each model gets its own budget. SFT, GRPO and SFT->GRPO on that model all
    spend exactly that budget.
  * Budget is measured in FLOPs. N (params) is the same for every run on one
    model, so we drop it and count *weighted tokens* (FLOPs / N):
        SFT   : 6 per token processed (prompt tokens count even if loss-masked)
        GRPO  : per rollout token (prompt + completion)
                  generation              2
                  policy update           6 x mu      (mu = num_iterations)
                  old logprobs (mu > 1)  +2
                  reference model (beta>0)+2
        LoRA  : use 4 instead of 6 for training tokens (no weight grads on frozen weights)
  * SFT->GRPO uses a fixed 25% SFT / 75% GRPO split of the same budget,
    each stage as its own run.
  * GPU-hours are logged for reporting only, never used to stop.

With the team defaults (LoRA, mu=1, beta=0.001) that is 4 per SFT token and
2 + 4 + 2 = 8 per GRPO rollout token.

Hook-up (any HF Trainer, incl. TRL GRPOTrainer / SFTTrainer). Mode, mu and
beta are read from the trainer args automatically:

    from compute_budget import ComputeBudgetCallback
    budget_cb = ComputeBudgetCallback(output_dir, budget=B)        # B in weighted tokens
    trainer = GRPOTrainer(..., callbacks=[token_cb, budget_cb])

SFT->GRPO stages:

    ComputeBudgetCallback(sft_dir,  budget=B, stage="sft_then_grpo:sft")    # spends 25% of B
    ComputeBudgetCallback(grpo_dir, budget=B, stage="sft_then_grpo:grpo")   # spends 75% of B

Custom training loop (e.g. SFT training_loops.py):

    tracker = ComputeBudget(output_dir, mode="sft", budget=B, lora=True)
    for batch in loader:
        ...optimizer.step()
        if tracker.step(tokens=batch["input_ids"].numel()):   # all tokens, incl. prompt
            save_checkpoint(); break

Table of runs:

    python compute_budget.py GRPO/* SFT/*
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

try:
    from transformers import TrainerCallback
except ImportError:  # custom loops / CLI work without transformers
    TrainerCallback = object

COMPUTE_FILE = "compute.json"
SFT_FRACTION = 0.25  # fixed SFT share of the budget in the SFT->GRPO arm
STAGE_FRACTIONS = {
    None: 1.0,
    "sft_then_grpo:sft": SFT_FRACTION,
    "sft_then_grpo:grpo": 1.0 - SFT_FRACTION,
}


def tokens_weight(
    mode: str, lora: bool = True, num_iterations: int = 1, beta: float = 0.0
) -> float:
    """FLOPs per token divided by N, per the team's agreed accounting."""
    train = 4.0 if lora else 6.0
    if mode == "sft":
        return train
    if mode == "grpo":
        w = 2.0 + train * num_iterations  # generation + policy update(s)
        if num_iterations > 1:
            w += 2.0  # extra forward for old logprobs
        if beta > 0:
            w += 2.0  # reference model forward
        return w
    raise ValueError(f"mode must be 'sft' or 'grpo', got {mode!r}")


def _gpu_info() -> tuple[str, int]:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0), torch.cuda.device_count()
    except ImportError:
        pass
    return "cpu", 0


def _peak_mem_gb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024**3
    except ImportError:
        pass
    return None


class ComputeBudget:
    """Framework-free running total. ComputeBudgetCallback wraps this.

    budget: total budget for this model in weighted tokens (FLOPs / N). The
    stage fraction is applied on top, so pass the same B to every arm.
    """

    def __init__(
        self,
        output_dir: str | Path,
        mode: str,
        budget: float | None = None,
        lora: bool = True,
        num_iterations: int = 1,
        beta: float = 0.0,
        stage: str | None = None,
        n_params: float | None = None,
        num_gpus: int | None = None,
    ):
        if stage not in STAGE_FRACTIONS:
            raise ValueError(f"stage must be one of {list(STAGE_FRACTIONS)}")
        self.output_dir = Path(output_dir)
        self.path = self.output_dir / COMPUTE_FILE
        self.mode = mode
        self.stage = stage
        self.weight = tokens_weight(mode, lora, num_iterations, beta)
        self.total_budget = budget
        self.budget = None if budget is None else budget * STAGE_FRACTIONS[stage]
        self.config = {
            "mode": mode,
            "lora": lora,
            "num_iterations": num_iterations,
            "beta": beta,
            "stage": stage,
            "weight_per_token": self.weight,
            "n_params": n_params,
        }
        self.n_params = n_params

        gpu_type, visible = _gpu_info()
        self.num_gpus = num_gpus or int(os.environ.get("WORLD_SIZE", 0)) or max(visible, 1)
        self.gpu_type = gpu_type

        prev = self._load()
        self.tokens = prev.get("tokens", 0.0)
        self.weighted_tokens = prev.get("weighted_tokens", 0.0)
        self.steps = prev.get("steps", 0)
        self.prev_gpu_hours = prev.get("gpu_hours", 0.0)
        self.prev_wall_s = prev.get("wall_clock_s", 0.0)
        self.peak_mem_gb = prev.get("peak_mem_gb") or 0.0
        self.segments = prev.get("segments", [])
        if prev:
            old = prev.get("config", {})
            if old.get("weight_per_token") not in (None, self.weight):
                print(
                    f"[compute_budget] WARNING: weight/token changed on resume "
                    f"({old.get('weight_per_token')} -> {self.weight})"
                )
            print(
                f"[compute_budget] resuming: {self.weighted_tokens:.3e} weighted tokens, "
                f"{self.steps} steps, {self.prev_gpu_hours:.3f} GPU-h so far"
            )

        self.t0 = time.time()
        self.segments.append(
            {"start": self.t0, "gpu_type": self.gpu_type, "num_gpus": self.num_gpus}
        )

    def _load(self) -> dict:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text())
            except json.JSONDecodeError:
                print(f"[compute_budget] WARNING: unreadable {self.path}, starting fresh")
        return {}

    @property
    def wall_clock_s(self) -> float:
        return self.prev_wall_s + (time.time() - self.t0)

    @property
    def gpu_hours(self) -> float:
        return self.prev_gpu_hours + (time.time() - self.t0) * self.num_gpus / 3600

    def over_budget(self) -> bool:
        return self.budget is not None and self.weighted_tokens >= self.budget

    def step(self, tokens: float) -> bool:
        """Record one optimizer step that processed `tokens` tokens.

        SFT: every token in the batch (prompt included). GRPO: every rollout
        token (prompt + completion). Returns True once the budget is spent.
        """
        self.steps += 1
        self.tokens += float(tokens)
        self.weighted_tokens += float(tokens) * self.weight
        mem = _peak_mem_gb()
        if mem is not None:
            self.peak_mem_gb = max(self.peak_mem_gb, mem)
        self.save()
        return self.over_budget()

    def metrics(self) -> dict:
        m = {
            "budget/weighted_tokens": self.weighted_tokens,
            "budget/tokens": self.tokens,
            "budget/gpu_hours": self.gpu_hours,
            "budget/steps": self.steps,
            "budget/peak_mem_gb": self.peak_mem_gb,
        }
        if self.budget:
            m["budget/frac_used"] = self.weighted_tokens / self.budget
        if self.n_params:
            m["budget/flops"] = self.weighted_tokens * self.n_params
        return m

    def save(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.segments[-1]["end"] = time.time()
        payload = {
            "config": self.config,
            "budget_total": self.total_budget,
            "budget_this_run": self.budget,
            "weighted_tokens": self.weighted_tokens,
            "tokens": self.tokens,
            "flops": self.weighted_tokens * self.n_params if self.n_params else None,
            "steps": self.steps,
            "over_budget": self.over_budget(),
            # reporting only:
            "gpu_type": self.gpu_type,
            "num_gpus": self.num_gpus,
            "gpu_hours": self.gpu_hours,
            "wall_clock_s": self.wall_clock_s,
            "peak_mem_gb": self.peak_mem_gb,
            "segments": self.segments,
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n")
        tmp.replace(self.path)  # atomic: a crash never leaves a half-written file


class ComputeBudgetCallback(TrainerCallback):
    """HF Trainer callback. Reads mode / mu / beta from the trainer args.

    Tokens per step come from state.num_input_tokens_seen. TRL's GRPOTrainer
    counts prompt + completion tokens there. For SFT with the HF Trainer, set
    include_num_input_tokens_seen=True in TrainingArguments.
    """

    def __init__(
        self,
        output_dir: str | Path,
        budget: float | None = None,
        stage: str | None = None,
        lora: bool = True,
        mode: str | None = None,
        n_params: float | None = None,
        num_gpus: int | None = None,
    ):
        self._kw = dict(
            output_dir=output_dir,
            budget=budget,
            stage=stage,
            lora=lora,
            n_params=n_params,
            num_gpus=num_gpus,
        )
        self._mode = mode
        self.tracker: ComputeBudget | None = None
        self._last_seen = 0
        self._warned = False

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        is_grpo = hasattr(args, "num_generations")
        mode = self._mode or ("grpo" if is_grpo else "sft")
        self.tracker = ComputeBudget(
            mode=mode,
            num_iterations=int(getattr(args, "num_iterations", 1) or 1),
            beta=float(getattr(args, "beta", 0.0) or 0.0),
            **self._kw,
        )
        # On resume the Trainer restores num_input_tokens_seen from the checkpoint.
        self._last_seen = getattr(state, "num_input_tokens_seen", 0) or 0
        c = self.tracker.config
        print(
            f"[compute_budget] mode={mode} weight/token={c['weight_per_token']} "
            f"(lora={c['lora']}, mu={c['num_iterations']}, beta={c['beta']}) "
            f"budget_this_run={self.tracker.budget}"
        )

    def on_step_end(self, args, state, control, **kwargs):
        if self.tracker is None:
            return
        seen = getattr(state, "num_input_tokens_seen", 0) or 0
        delta = seen - self._last_seen
        self._last_seen = seen
        # delta is 0 on reused-rollout steps when mu > 1 (already counted via mu);
        # only a counter that never moves means tokens aren't being tracked.
        if seen == 0 and not self._warned:
            self._warned = True
            print(
                "[compute_budget] WARNING: state.num_input_tokens_seen is not increasing. "
                "For SFT set include_num_input_tokens_seen=True in TrainingArguments; "
                "the budget cannot be enforced without it."
            )
        done = self.tracker.step(tokens=max(delta, 0))
        _wandb_log(self.tracker.metrics(), state.global_step)
        if done:
            print(
                f"[compute_budget] budget reached at step {state.global_step} "
                f"({self.tracker.weighted_tokens:.3e} / {self.tracker.budget:.3e} weighted tokens, "
                f"{self.tracker.gpu_hours:.3f} GPU-h); saving and stopping."
            )
            control.should_save = True
            control.should_training_stop = True

    def on_train_end(self, args, state, control, **kwargs):
        if self.tracker is not None:
            self.tracker.save()


def _wandb_log(metrics: dict, global_step: int) -> None:
    """Log next to the Trainer's own W&B rows.

    The HF WandbCallback plots against "train/global_step"; wandb.log(step=...)
    after the Trainer already advanced W&B's internal step gets dropped, so the
    step goes in as a metric instead.
    """
    try:
        import wandb

        if wandb.run is not None:
            wandb.log({**metrics, "train/global_step": global_step})
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# CLI: python compute_budget.py GRPO/* SFT/*
# ---------------------------------------------------------------------------
def _table(dirs: list[str]) -> None:
    rows = []
    for d in dirs:
        p = Path(d) / COMPUTE_FILE
        if not p.exists():
            continue
        c = json.loads(p.read_text())
        cfg = c.get("config", {})
        budget = c.get("budget_this_run")
        used = c.get("weighted_tokens", 0.0)
        rows.append(
            (
                Path(d).name,
                cfg.get("mode", "?"),
                c.get("steps", 0),
                used,
                f"{used / budget:.0%}" if budget else "-",
                c.get("gpu_hours", 0.0),
                c.get("gpu_type", "?"),
            )
        )
    if not rows:
        print("No compute.json found in the given directories.")
        return
    print(f"{'run':<40} {'mode':<5} {'steps':>6} {'weighted_tok':>13} {'used':>5} {'gpu_h':>7}  gpu")
    for r in sorted(rows):
        print(f"{r[0]:<40} {r[1]:<5} {r[2]:>6} {r[3]:>13.3e} {r[4]:>5} {r[5]:>7.3f}  {r[6]}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    _table(sys.argv[1:])
