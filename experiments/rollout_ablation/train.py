"""
Resumable, time-limit-aware wrapper around grpo.grpo.main() for ablation runs.

    python -m experiments.rollout_ablation.train --deadline-epoch 1767225600 -- \
        --task shortest_path --G 8 --dataset data/shortest_path --seed 42 \
        --max-steps 400 --lr 2e-5 --completion-tokens 4096 --context-tokens 5120 \
        --init-adapter runs/ablation/init_adapter/seed0 --report-to none \
        --output-dir runs/ablation/shortest_path/G8-s42

Runs the unmodified training code (grpo.main()) and adds only what long runs on
SLURM need. No shared file is edited: GRPOTrainer is swapped for a subclass on
the grpo.grpo module, as in grpo/rollout_validation.py.

  resume       picks the newest complete checkpoint-* in --output-dir and passes it
               to trainer.train(resume_from_checkpoint=...). Before resuming it
               rewinds the run's own logs to that checkpoint so nothing is counted
               twice: rollouts.jsonl is cut to step*64 rows (original kept as
               .bak), compute.json token totals are reset from the checkpoint's
               num_input_tokens_seen (GPU-hours are kept: that time was spent),
               and the metrics step label continues from the checkpoint.
  time limit   a callback saves a checkpoint and stops cleanly when the next step
               would not finish before --deadline-epoch minus --margin-seconds.
  status       writes status.json and appends to attempts.jsonl. Exit 0 = run
               complete (global_step reached the trainer's max_steps, whether set by
               --max-steps or by --epochs); exit 75 = stopped early, resubmit.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.rollout_ablation import config as C  # noqa: E402


# ---------------------------------------------------------------------------
# Resume helpers (pure file operations; unit-tested)
# ---------------------------------------------------------------------------
def _ckpt_step(p: Path) -> int:
    return int(p.name.rsplit("-", 1)[-1])


def latest_checkpoint(run_dir: Path) -> Path | None:
    """Newest checkpoint-N with a trainer_state.json and adapter weights (i.e. fully saved)."""
    ckpts = []
    for p in run_dir.glob("checkpoint-*"):
        try:
            _ckpt_step(p)
        except ValueError:
            continue
        if (p / "trainer_state.json").is_file() and any(
                (p / f).is_file() for f in ("adapter_model.safetensors", "adapter_model.bin")):
            ckpts.append(p)
    return max(ckpts, key=_ckpt_step) if ckpts else None


def rewind_rollouts(run_dir: Path, step: int, rollouts_per_step: int, tag: str) -> dict[str, Any]:
    """Keep exactly the rows of steps 1..step (labels 0..step-1); back up the rest."""
    path = run_dir / "rollouts.jsonl"
    if not path.is_file():
        return {"rollouts": "missing"}
    lines = path.read_text().splitlines(keepends=True)
    keep = step * rollouts_per_step
    if len(lines) < keep:
        raise RuntimeError(f"{path} has {len(lines)} rows, checkpoint-{step} needs {keep}")
    for i in range(keep):
        lab = json.loads(lines[i]).get("step")
        if lab != i // rollouts_per_step:
            raise RuntimeError(f"{path}: row {i} has step label {lab}, expected "
                               f"{i // rollouts_per_step}; rollouts are misaligned")
    dropped = len(lines) - keep
    if dropped:
        shutil.copy2(path, run_dir / f"rollouts.jsonl.{tag}.bak")
        path.write_text("".join(lines[:keep]))
    return {"rollouts_kept": keep, "rollouts_dropped": dropped}


def rewind_compute(run_dir: Path, ckpt: Path) -> dict[str, Any]:
    """Reset compute.json token totals to the checkpoint (GPU-hours stay: they were spent)."""
    path = run_dir / "compute.json"
    if not path.is_file():
        return {"compute": "missing"}
    d = json.loads(path.read_text())
    st = json.loads((ckpt / "trainer_state.json").read_text())
    tokens = float(st.get("num_input_tokens_seen") or 0)
    weight = float(d.get("config", {}).get("weight_per_token") or 0)
    before = {k: d.get(k) for k in ("tokens", "weighted_tokens", "steps")}
    d.update(tokens=tokens, weighted_tokens=tokens * weight, steps=int(st["global_step"]))
    if d.get("budget_this_run") is not None:
        d["over_budget"] = d["weighted_tokens"] >= d["budget_this_run"]
    d.setdefault("rewinds", []).append({"to_step": int(st["global_step"]), "before": before,
                                        "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
    path.write_text(json.dumps(d, indent=2) + "\n")
    return {"compute_before": before, "compute_tokens": tokens}


def prepare_resume(run_dir: Path, ckpt: Path, rollouts_per_step: int) -> dict[str, Any]:
    step = _ckpt_step(ckpt)
    tag = f"pre-resume-{step}-{int(time.time())}"
    info: dict[str, Any] = {"resume_from": str(ckpt), "resume_step": step}
    info.update(rewind_rollouts(run_dir, step, rollouts_per_step, tag))
    info.update(rewind_compute(run_dir, ckpt))
    # CompletionTokenCallback keeps its total in memory and rewrites this file at the
    # end of each attempt; keep the earlier attempt's copy.
    ct = run_dir / "completion_tokens.json"
    if ct.is_file():
        ct.rename(run_dir / f"completion_tokens.{tag}.json")
    return info


# ---------------------------------------------------------------------------
# Time-limit callback
# ---------------------------------------------------------------------------
def should_stop(now: float, deadline: float | None, step_seconds: float | None,
                margin: float) -> bool:
    """True if another step (plus a save margin) would run past the deadline."""
    if deadline is None:
        return False
    need = (step_seconds or 0.0) * 1.25 + margin
    return now + need >= deadline


def make_time_limit_callback(deadline: float | None, margin: float):
    from transformers import TrainerCallback

    class TimeLimitCallback(TrainerCallback):
        def __init__(self):
            self.stopped_at: int | None = None
            self._last = None
            self._step_s: float | None = None

        def on_step_begin(self, args, state, control, **kw):
            if self._last is None:
                self._last = time.time()

        def on_step_end(self, args, state, control, **kw):
            now = time.time()
            dt = now - self._last
            self._last = now
            self._step_s = dt if self._step_s is None else max(dt, 0.7 * self._step_s + 0.3 * dt)
            if state.global_step < state.max_steps and should_stop(now, deadline, self._step_s, margin):
                print(f"[ablation] step {state.global_step}: deadline in "
                      f"{(deadline - now) / 60:.1f} min, step ~{self._step_s / 60:.1f} min; "
                      "saving checkpoint and stopping for resubmit")
                control.should_save = True
                control.should_training_stop = True
                self.stopped_at = state.global_step

    return TimeLimitCallback()


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def _git() -> dict[str, Any]:
    def run(*a):
        try:
            return subprocess.check_output(["git", *a], cwd=C.REPO_ROOT, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None
    return {"git_sha": run("rev-parse", "HEAD"),
            "git_dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}


def check_constants(g) -> None:
    bad = {k: (getattr(g, k, None), v) for k, v in C.EXPECTED_GRPO_CONSTANTS.items()
           if getattr(g, k, None) != v}
    if bad:
        raise SystemExit(f"grpo.grpo constants changed since the ablation was configured "
                         f"(got, expected): {bad}")


def check_init_adapter(path: str | None, model_id: str) -> dict[str, Any] | None:
    if not path:
        return None
    info_path = Path(path) / "init_adapter.json"
    if not info_path.is_file():
        raise SystemExit(f"{info_path} missing; create it with make_init_adapter")
    info = json.loads(info_path.read_text())
    if info.get("model_id") != model_id:
        raise SystemExit(f"init adapter is for {info.get('model_id')}, training {model_id}")
    return info


def run(grpo_argv: list[str], *, deadline: float | None, margin: float,
        model_id: str | None = None) -> int:
    import grpo.grpo as g

    sys.argv = ["grpo.grpo", *grpo_argv]
    args = g.parse_args()
    if not args.output_dir:
        raise SystemExit("--output-dir is required (resume looks for checkpoints there)")
    if model_id:
        g.MODEL_ID = model_id
    check_constants(g)
    init_info = check_init_adapter(args.init_adapter, g.MODEL_ID)

    run_dir = Path(args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    attempt: dict[str, Any] = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "host": os.uname().nodename, "argv": grpo_argv, "model_id": g.MODEL_ID,
        "deadline_epoch": deadline, **_git(),
    }
    if init_info:
        attempt["init_adapter_sha256"] = init_info["adapter_sha256"]

    status_path = run_dir / "status.json"
    if status_path.is_file() and json.loads(status_path.read_text()).get("done"):
        print(f"[ablation] {run_dir} already complete; nothing to do")
        return 0

    ckpt = latest_checkpoint(run_dir)
    start_step = 0
    if ckpt is not None:
        attempt.update(prepare_resume(run_dir, ckpt, g.ROLLOUTS_PER_STEP))
        start_step = _ckpt_step(ckpt)
        print(f"[ablation] resuming {run_dir} from {ckpt.name}")
    elif (run_dir / "rollouts.jsonl").exists():
        # Crashed before the first checkpoint: start clean.
        tag = f"pre-restart-{int(time.time())}"
        for name in ("rollouts.jsonl", "compute.json", "completion_tokens.json"):
            if (run_dir / name).exists():
                (run_dir / name).rename(run_dir / f"{name}.{tag}.bak")
        attempt["restart_from_scratch"] = True

    time_cb = make_time_limit_callback(deadline, margin)
    holder: dict[str, Any] = {}
    base = g.GRPOTrainer

    class AblationGRPOTrainer(base):
        def __init__(self, *a, **kw):
            cbs = list(kw.get("callbacks") or [])
            for cb in cbs:   # continue rollouts.jsonl step labels from the checkpoint
                if type(cb).__name__ == "GRPOMetricsCallback":
                    cb.metrics.step = start_step
            kw["callbacks"] = [*cbs, time_cb]
            super().__init__(*a, **kw)
            holder["trainer"] = self

        def train(self, *a, **kw):
            if ckpt is not None:
                kw["resume_from_checkpoint"] = str(ckpt)
            return super().train(*a, **kw)

    g.GRPOTrainer = AblationGRPOTrainer
    try:
        g.main()
    finally:
        g.GRPOTrainer = base

    st = holder["trainer"].state
    max_steps = int(st.max_steps)       # set by the Trainer for both --max-steps and --epochs
    done = int(st.global_step) >= max_steps
    attempt.update(ended=time.strftime("%Y-%m-%dT%H:%M:%S"), start_step=start_step,
                   end_step=int(st.global_step), stopped_for_time=time_cb.stopped_at is not None)
    with open(run_dir / "attempts.jsonl", "a") as f:
        f.write(json.dumps(attempt) + "\n")
    status = {"done": done, "global_step": int(st.global_step), "max_steps": max_steps,
              "epochs": args.epochs,
              "num_input_tokens_seen": getattr(st, "num_input_tokens_seen", None),
              "updated": attempt["ended"]}
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    print(f"[ablation] {run_dir}: step {st.global_step}/{max_steps} "
          f"({'done' if done else 'needs resume'})")
    return 0 if done else C.EXIT_NEEDS_RESUME


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    grpo_argv: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, grpo_argv = argv[:i], argv[i + 1:]
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--deadline-epoch", type=float, default=None,
                   help="unix time the SLURM job ends (stop + save before it)")
    p.add_argument("--margin-seconds", type=float, default=C.STOP_MARGIN_SECONDS)
    p.add_argument("--model-id", default=None, help="override grpo.grpo.MODEL_ID (tests)")
    a = p.parse_args(argv)
    return run(grpo_argv, deadline=a.deadline_epoch, margin=a.margin_seconds,
               model_id=a.model_id)


if __name__ == "__main__":
    sys.exit(main())
