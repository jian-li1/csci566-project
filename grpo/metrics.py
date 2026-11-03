"""
GRPO analysis & stability metrics (owner: Manasa).

Note: no `import math` here on purpose: grpo/math/ is a package and would shadow
the stdlib module when grpo/ is on sys.path.

Tracks, per step and per task:
  - reward distribution        reward/{task}/mean|std|min|max|hist
  - accept rate                reward/{task}/accept_rate   (env acceptor flag)
  - group signal               group/{task}/frac_zero_std | frac_all_correct | frac_all_wrong
                               (reward variance; groups with zero reward std give GRPO no gradient)
                               group/{task}/frac_none_accepted | frac_all_accepted | frac_mixed_acceptance
                               (verifier acceptance; partial rewards like 0.05 are not "accepted")
  - advantages                 adv/{task}/abs_mean | std | abs_max   ((r - mean_g) / (std_g + eps))
  - failure cases              fail/{task}/<type>_rate     (missing_answer, empty_answer,
                                                            empty_response, verifier_crash,
                                                            wrong, unscored)
  - rollout length             len/{task}/mean | p95 | trunc_rate  (needs completion_ids;
                               trunc_rate is a length-limit proxy: n_tokens >= max_completion_length,
                               not confirmed EOS-absence truncation)
  - stability alerts           alert/kl_spike | alert/entropy_collapse | alert/reward_std_collapse
                               (read from TRL's own logged kl / entropy)

Every rollout is also appended to <output_dir>/rollouts.jsonl, so failure cases can be
inspected offline:   python grpo/metrics.py <output_dir>

Hookup (TRL GRPOTrainer — grpo.py, or grpo/math/train.py / grpo/mbpp/train.py next to token_cb):

    from grpo.metrics import GRPOMetrics, GRPOMetricsCallback, make_env_reward_fn
    metrics = GRPOMetrics(output_dir)
    reward_fn = make_env_reward_fn(envs, metrics)          # envs = {"countdown": CountdownEnv(...), ...}
    trainer = GRPOTrainer(..., reward_funcs=[reward_fn],
                          callbacks=[token_cb, budget_cb, GRPOMetricsCallback(metrics)])

The dataset must have a "task" column (env name) and an "entry" column (the dict the
env's load_problem() takes). TRL passes extra dataset columns to reward functions as kwargs.
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict, deque
from typing import Any, Callable

try:
    from transformers import TrainerCallback
except ImportError:  # lets the offline summary run without transformers
    TrainerCallback = object

ADV_EPS = 1e-4
FAIL_TYPES = ("missing_answer", "empty_answer", "empty_response", "verifier_crash",
              "syntax_error", "timeout", "partial", "wrong", "unscored")


def _failure_type(out: dict[str, Any]) -> str | None:
    """
    Map an env.step() result to a failure bucket (None = accepted).
    Works for math, mbpp and Reasoning Gym envs (same step() shape; error strings
    taken from grpo/math/environment.py, grpo/mbpp/environment.py, reasoning_gym_env.py).
    """
    if out.get("acceptor"):
        return None
    check = out.get("check") or {}
    err = str(check.get("error") or "")
    reward = out.get("reward")
    if err == "empty response":
        return "empty_response"
    if err.startswith("missing <answer>"):
        return "missing_answer"
    if err in ("empty <answer>", "empty answer"):
        return "empty_answer"
    if reward is None:
        # envs set reward=None + severity end_rollout/stop_training when the verifier raises
        if err and err not in ("step before reset", "step after terminate"):
            return "verifier_crash"
        return "unscored"
    # mbpp: per-test results carry SyntaxError / timeout messages
    test_errs = [str(t.get("error") or "") for t in (check.get("tests") or []) if isinstance(t, dict)]
    if any(e.startswith("SyntaxError") for e in test_errs):
        return "syntax_error"
    if any(e.startswith("timeout") for e in test_errs):
        return "timeout"
    if reward > 0:
        return "partial"   # e.g. mbpp "3/5 tests passed", RG partial credit
    return "wrong"


def _pct(xs: list[float], q: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * (len(s) - 1) + 0.5))]


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def _completion_n_tokens(token_ids: Any) -> int | None:
    """
    Count tokens in one `completion_ids[i]` entry from TRL GRPOTrainer.

    Verified against current TRL (`trl.trainer.grpo_trainer`): reward functions receive
    a Python list of per-completion ID lists. Those lists are already cut with the
    generation `completion_mask` (keep tokens through the first EOS); TRL itself logs
    lengths as `len(ids)`. Padding is applied only later when building train tensors.

    Therefore `len(token_ids)` is the valid generation length. Do **not** strip
    `pad_token_id` here: train scripts often set `pad_token = eos_token`, so filtering
    pad would also drop a legitimate trailing EOS.

    If a 1-D tensor is passed, use its numel (already a single sequence). Refuse 2-D /
    batched tensors — we are not given `completion_mask` in the reward kwargs.
    """
    if token_ids is None:
        return None
    # torch.Tensor / numpy-like without importing torch (optional dep for offline summary).
    if hasattr(token_ids, "detach"):
        t = token_ids.detach().cpu()
        if getattr(t, "ndim", None) != 1:
            return None
        return int(t.numel())
    if hasattr(token_ids, "ndim") and getattr(token_ids, "ndim") != 1:
        return None
    try:
        return len(token_ids)
    except TypeError:
        return None


class GRPOMetrics:
    """Buffers rollouts between logging steps and turns them into metrics."""

    def __init__(
        self,
        output_dir: str,
        max_completion_length: int | None = None,
        kl_spike: float = 0.5,          # alert if kl exceeds this
        entropy_floor: float = 0.05,    # alert if entropy drops below this
        window: int = 20,               # steps for the reward-std collapse check
    ) -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.path = os.path.join(output_dir, "rollouts.jsonl")
        self.max_completion_length = max_completion_length
        self.kl_spike, self.entropy_floor = kl_spike, entropy_floor
        self._buf: list[dict[str, Any]] = []
        self._reward_std_hist: deque[float] = deque(maxlen=window)
        self.step = 0

    # ---- collection -------------------------------------------------------
    def record(self, task: str, prompt: str, completion: str, out: dict[str, Any],
               n_tokens: int | None = None) -> None:
        # Length-limit proxy only (n >= max_completion_length). Not TRL's EOS-absence
        # truncation check (`ids[-1] not in eos_and_pad`).
        hit_length_limit = (
            n_tokens is not None
            and self.max_completion_length is not None
            and n_tokens >= self.max_completion_length
        )
        row = {
            "step": self.step, "task": task, "prompt": prompt, "completion": completion,
            "reward": out.get("reward"), "acceptor": bool(out.get("acceptor")),
            "severity": out.get("severity"), "fail": _failure_type(out),
            "error": (out.get("check") or {}).get("error"),
            "n_tokens": n_tokens,
            "truncated": hit_length_limit,  # legacy key; means hit_length_limit
            "hit_length_limit": hit_length_limit,
        }
        self._buf.append(row)
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")

    # ---- aggregation ------------------------------------------------------
    def flush(self, trainer_logs: dict[str, float] | None = None) -> dict[str, float]:
        """Compute metrics over everything recorded since the last flush."""
        rows, self._buf = self._buf, []
        m: dict[str, float] = {}
        by_task: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            by_task[r["task"]].append(r)
            by_task["all"].append(r)

        for task, rs in by_task.items():
            n = len(rs)
            rewards = [r["reward"] for r in rs if r["reward"] is not None]
            if rewards:
                m[f"reward/{task}/mean"] = sum(rewards) / len(rewards)
                m[f"reward/{task}/std"] = _std(rewards)
                m[f"reward/{task}/min"] = min(rewards)
                m[f"reward/{task}/max"] = max(rewards)
            m[f"reward/{task}/accept_rate"] = sum(r["acceptor"] for r in rs) / n
            for ft in FAIL_TYPES:
                m[f"fail/{task}/{ft}_rate"] = sum(r["fail"] == ft for r in rs) / n

            # groups = completions that share a prompt
            groups: dict[str, list[float]] = defaultdict(list)
            for r in rs:
                groups[r["prompt"]].append(0.0 if r["reward"] is None else r["reward"])
            gs = [g for g in groups.values() if len(g) > 1]
            if gs:
                m[f"group/{task}/frac_zero_std"] = sum(_std(g) == 0 for g in gs) / len(gs)
                m[f"group/{task}/frac_all_correct"] = sum(min(g) >= 0.99 for g in gs) / len(gs)
                m[f"group/{task}/frac_all_wrong"] = sum(max(g) <= 0.0 for g in gs) / len(gs)
                advs = []
                for g in gs:
                    mu, sd = sum(g) / len(g), _std(g)
                    advs += [(x - mu) / (sd + ADV_EPS) for x in g]
                m[f"adv/{task}/abs_mean"] = sum(abs(a) for a in advs) / len(advs)
                m[f"adv/{task}/std"] = _std(advs)
                m[f"adv/{task}/abs_max"] = max(abs(a) for a in advs)

            # Group by the same prompt, using verifier acceptance.
            accepted_groups: dict[str, list[bool]] = defaultdict(list)
            for r in rs:
                accepted_groups[r["prompt"]].append(bool(r["acceptor"]))
            ags = [g for g in accepted_groups.values() if len(g) > 1]
            if ags:
                m[f"group/{task}/frac_none_accepted"] = (
                    sum(not any(g) for g in ags) / len(ags)
                )
                m[f"group/{task}/frac_all_accepted"] = (
                    sum(all(g) for g in ags) / len(ags)
                )
                m[f"group/{task}/frac_mixed_acceptance"] = (
                    sum(any(g) and not all(g) for g in ags) / len(ags)
                )

            lens = [r["n_tokens"] for r in rs if r["n_tokens"] is not None]
            if lens:
                m[f"len/{task}/mean"] = sum(lens) / len(lens)
                m[f"len/{task}/p95"] = _pct(lens, 0.95)
                # Proxy: fraction with n_tokens >= max_completion_length (not EOS-trunc).
                hit = sum(r.get("hit_length_limit", r["truncated"]) for r in rs) / n
                m[f"len/{task}/trunc_rate"] = hit  # legacy name
                m[f"len/{task}/hit_length_limit_rate"] = hit

            if any(r["severity"] == "stop_training" for r in rs):
                m["alert/stop_training"] = 1.0

        # stability alerts from TRL's own logs
        logs = trainer_logs or {}
        kl, ent = logs.get("kl"), logs.get("entropy")
        m["alert/kl_spike"] = float(kl is not None and kl > self.kl_spike)
        m["alert/entropy_collapse"] = float(ent is not None and ent < self.entropy_floor)
        if "reward/all/std" in m:
            self._reward_std_hist.append(m["reward/all/std"])
        h = self._reward_std_hist
        m["alert/reward_std_collapse"] = float(len(h) == h.maxlen and max(h) < 1e-3)
        return m


def make_env_reward_fn(envs: dict[str, Any], metrics: GRPOMetrics,
                       unscored_reward: float = 0.0) -> Callable[..., list[float]]:
    """TRL reward function that scores with our envs and records every rollout."""

    def env_reward(prompts, completions, task, entry, completion_ids=None, **_):
        rewards = []
        for i, (p, c, t, e) in enumerate(zip(prompts, completions, task, entry)):
            text = c if isinstance(c, str) else c[-1]["content"]   # plain or chat format
            ptxt = p if isinstance(p, str) else p[-1]["content"]
            env = envs[t]
            env.load_problem(e)
            out = env.step(text)
            # TRL passes EOS-trimmed unpadded id lists; see _completion_n_tokens.
            n_tok = (
                _completion_n_tokens(completion_ids[i])
                if completion_ids is not None
                else None
            )
            metrics.record(t, ptxt, text, out, n_tok)
            if out.get("severity") == "stop_training":
                raise RuntimeError(f"[{t}] fatal verifier error: {out['check'].get('error')}")
            rewards.append(unscored_reward if out["reward"] is None else float(out["reward"]))
        return rewards

    env_reward.__name__ = "env_reward"
    return env_reward


class GRPOMetricsCallback(TrainerCallback):
    """Adds our metrics to every Trainer log (W&B picks them up automatically)."""

    def __init__(self, metrics: GRPOMetrics) -> None:
        self.metrics = metrics

    def on_step_end(self, args, state, control, **kwargs):
        self.metrics.step = state.global_step

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or "loss" not in logs and "reward" not in logs:
            return  # skip eval / final summary logs
        extra = self.metrics.flush(logs)
        logs.update(extra)
        if extra.get("alert/kl_spike") or extra.get("alert/entropy_collapse") \
                or extra.get("alert/reward_std_collapse"):
            print(f"[grpo-metrics] step {state.global_step} stability alert: "
                  f"{ {k: v for k, v in extra.items() if k.startswith('alert/') and v} }")


# ---- offline failure-case summary --------------------------------------------
def summarize(output_dir: str, n_examples: int = 3) -> None:
    path = os.path.join(output_dir, "rollouts.jsonl")
    rows = [json.loads(l) for l in open(path)]
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
    print(f"{'task':<16}{'n':>7}{'accept':>8}" + "".join(f"{f[:11]:>12}" for f in FAIL_TYPES))
    for t, rs in sorted(by_task.items()):
        n = len(rs)
        print(f"{t:<16}{n:>7}{sum(r['acceptor'] for r in rs) / n:>8.2f}"
              + "".join(f"{sum(r['fail'] == f for r in rs) / n:>12.2f}" for f in FAIL_TYPES))
    for t, rs in sorted(by_task.items()):
        fails = [r for r in rs if r["fail"] and r["fail"] != "wrong"][-n_examples:]
        for r in fails:
            print(f"\n--- {t} | step {r['step']} | {r['fail']} ---\n{r['completion'][-400:]}")


if __name__ == "__main__":
    summarize(sys.argv[1])
