"""
Analysis for the GRPO group-size ablation (owner: Mehul).

Compares the G arms on the five questions of the rollout study:

  reward diversity      fraction of groups with zero reward std (no GRPO gradient),
                        i.e. the share of rollouts that carry learning signal
  all-correct / -wrong  observed fractions of all-accepted / none-accepted groups over
                        training, next to what the base model predicts for each G
                        (evaluate.py group_predictions, ~ p^G and (1-p)^G)
  generation cost       completion tokens, weighted tokens (FLOPs/N), step time,
                        GPU-hours, share of completions hitting the length cap
  learning efficiency   val accuracy vs steps / weighted tokens / GPU-hours; gain per
                        GPU-hour; accuracy at the largest GPU-hour budget all arms reach
  final val accuracy    mean +- std over seeds, gain over base, by difficulty bucket

Inputs (whatever exists; missing pieces are reported, not fatal)
  run dirs              rollouts.jsonl, checkpoint-*/trainer_state.json (TRL log_history),
                        compute.json, manifest.json, checkpoint-*/eval/<task>/<stem>.json
  --baseline-dir        evaluate.py output for the base model (baselines/<model>/<task>):
                        <eval-stem>.json (+ _per_prompt.jsonl) and <train-stem>.json

Usage
    python -m analysis.rollout_ablation runs/ablation/math \
        --baseline-dir baselines/Qwen3.5-4B-Base/math --out analysis/out/math
    # pilot runs (no rollouts / evals): TRL-log metrics only
    python -m analysis.rollout_ablation runs/shortest_path_13383839_analysis --out /tmp/pilot

Outputs in --out: report.md, runs.csv, arms.csv, steps.csv, evals.csv, buckets.csv,
fig_*.png (+ the CSV behind every figure). Needs only the stdlib; figures need matplotlib.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

ROLLOUTS_PER_STEP = 64
ACCEPT_THRESHOLD = 0.99
TASK_NAMES = ("countdown", "knights_knaves", "shortest_path", "math", "kodcode",
              "opencodeinstruct")
EARLY_STEPS = 10          # "start of training" window for predicted-vs-observed
PHASE_STEPS = 50          # first/last window for early vs late comparisons


# ---------------------------------------------------------------------------
# Small stats helpers
# ---------------------------------------------------------------------------
def mean(xs: Iterable[float | None]) -> float | None:
    v = [x for x in xs if x is not None]
    return sum(v) / len(v) if v else None


def std(xs: Iterable[float | None]) -> float | None:
    v = [x for x in xs if x is not None]
    return statistics.stdev(v) if len(v) > 1 else None


def rolling(xs: Sequence[float | None], w: int) -> list[float | None]:
    """Centered rolling mean that ignores None."""
    h = w // 2
    return [mean(xs[max(0, i - h):i + h + 1]) for i in range(len(xs))]


def interp(xs: Sequence[float], ys: Sequence[float], x: float) -> float | None:
    """Linear interpolation of y at x on an increasing xs; None outside the range."""
    pts = sorted(zip(xs, ys))
    if not pts or x < pts[0][0] or x > pts[-1][0]:
        return None
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            return y0 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return pts[-1][1]


def trapz_mean(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Area under y(x) divided by the x range (= time-averaged y)."""
    pts = sorted(zip(xs, ys))
    if len(pts) < 2 or pts[-1][0] == pts[0][0]:
        return None
    area = sum((x1 - x0) * (y0 + y1) / 2 for (x0, y0), (x1, y1) in zip(pts, pts[1:]))
    return area / (pts[-1][0] - pts[0][0])


def _sd(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


# ---------------------------------------------------------------------------
# Per-run loading
# ---------------------------------------------------------------------------
@dataclass
class Run:
    path: Path
    task: str | None
    G: int | None
    seed: int | None
    cap: int | None = None                # completion-token cap (its own arm if it differs)
    arm: str = ""                         # display label, set per task in analyze()
    n_prompts: int | None = None          # distinct training prompts (for the epoch axis)
    steps: dict[int, dict[str, Any]] = field(default_factory=dict)   # step -> metrics
    evals: dict[int, dict[str, Any]] = field(default_factory=dict)   # ckpt step -> eval summary
    eval_dirs: dict[int, Path] = field(default_factory=dict)
    compute: dict[str, Any] | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        if self.G is not None and self.seed is not None:
            cap = f"-L{self.cap}" if self.cap else ""
            return f"G{self.G}{cap}-s{self.seed}"
        return self.path.name


def find_run_dirs(paths: Iterable[Path]) -> list[Path]:
    """A run dir has rollouts.jsonl, compute.json or checkpoint-*; search below each path."""
    def is_run(p: Path) -> bool:
        return ((p / "rollouts.jsonl").is_file() or (p / "compute.json").is_file()
                or any(p.glob("checkpoint-*/trainer_state.json")))
    found: list[Path] = []
    for p in paths:
        if is_run(p):
            found.append(p)
            continue
        cands = {q.parent for q in p.rglob("compute.json")}
        cands |= {q.parent.parent for q in p.rglob("checkpoint-*/trainer_state.json")}
        cands |= {q.parent for q in p.rglob("rollouts.jsonl")}
        found += sorted(c for c in cands if is_run(c) and not c.name.startswith("checkpoint-"))
    return sorted(set(found))


def infer_identity(path: Path, default_G: int | None) -> tuple[str | None, int | None, int | None]:
    m = _read_json(path / "manifest.json") or {}
    task, G, seed = m.get("task"), m.get("G"), m.get("seed")
    text = str(path)
    if G is None:
        g = re.search(r"(?:^|[-_/])G(\d+)(?:[-_/]|$)", text)
        G = int(g.group(1)) if g else default_G
    if seed is None:
        s = re.search(r"-s(\d+)(?:[-_/]|$)", path.name)
        seed = int(s.group(1)) if s else None
    if task is None:
        for t in sorted(TASK_NAMES, key=len, reverse=True):
            if t in text:
                task = t
                break
    return task, G, seed


def infer_cap(path: Path) -> int | None:
    """Completion-token cap from the manifest, else an -L<n>- tag in the run name."""
    m = _read_json(path / "manifest.json") or {}
    cap = (m.get("hparams") or {}).get("completion_tokens")
    if cap:
        return int(cap)
    tag = re.search(r"(?:^|[-_])L(\d+)(?:[-_]|$)", path.name)
    return int(tag.group(1)) if tag else None


def _read_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _ckpt_step(p: Path) -> int | None:
    tail = p.name.rsplit("-", 1)[-1]
    return int(tail) if tail.isdigit() else None


def load_log_history(run_dir: Path) -> list[dict[str, Any]]:
    cands = [run_dir / "trainer_state.json"]
    cands += sorted(run_dir.glob("checkpoint-*/trainer_state.json"),
                    key=lambda p: _ckpt_step(p.parent) or -1, reverse=True)
    for p in cands:
        d = _read_json(p)
        if d and d.get("log_history"):
            return d["log_history"]
    return []


TRL_KEYS = {
    "reward": "trl_reward", "reward_std": "trl_reward_std",
    "frac_reward_zero_std": "trl_frac_zero_std",
    "completions/mean_length": "mean_length", "completions/clipped_ratio": "clipped_ratio",
    "step_time": "step_time", "kl": "kl", "entropy": "entropy", "num_tokens": "num_tokens",
}


def group_metrics(block: list[tuple[str, float, bool, int | None, bool]], G: int) -> dict[str, Any]:
    """Metrics for one step's 64 rollouts: (prompt, reward, accepted, n_tokens, hit_length)."""
    groups: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    for p, r, a, _, _ in block:
        groups[p].append((r, a))
    gs = [g for g in groups.values() if len(g) > 1]
    out: dict[str, Any] = {"n_rollouts": len(block), "n_groups": len(gs)}
    if gs:
        rewards = [[r for r, _ in g] for g in gs]
        acc = [[a for _, a in g] for g in gs]
        zero = [_sd(r) < 1e-8 for r in rewards]
        out.update(
            frac_zero_std=sum(zero) / len(gs),
            effective_rollout_frac=sum(len(g) for g, z in zip(gs, zero) if not z) / sum(map(len, gs)),
            mean_group_reward_std=mean(_sd(r) for r in rewards),
            frac_all_accepted=sum(all(a) for a in acc) / len(gs),
            frac_none_accepted=sum(not any(a) for a in acc) / len(gs),
            frac_all_reward_max=sum(min(r) >= ACCEPT_THRESHOLD for r in rewards) / len(gs),
            frac_all_reward_zero=sum(max(r) <= 0.0 for r in rewards) / len(gs),
        )
        out["frac_mixed_acceptance"] = 1 - out["frac_all_accepted"] - out["frac_none_accepted"]
    out["accept_rate"] = mean(float(a) for _, _, a, _, _ in block)
    out["mean_reward"] = mean(r for _, r, _, _, _ in block)
    lens = [n for *_, n, _ in block if n is not None]
    out["rollout_mean_tokens"] = mean(lens)
    out["hit_length_rate"] = mean(float(h) for *_, h in block)
    return out


def load_rollout_steps(run_dir: Path, G: int) -> tuple[dict[int, dict[str, Any]], int]:
    """Stream rollouts.jsonl (can be 100s of MB); label s -> optimizer step s+1.
    Also returns the number of distinct prompts seen."""
    path = run_dir / "rollouts.jsonl"
    if not path.is_file():
        return {}, 0
    by_label: dict[int, list] = defaultdict(list)
    prompts: set[int] = set()
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            rew = r.get("reward")
            prompts.add(hash(r["prompt"]))
            by_label[int(r.get("step", 0))].append((
                r["prompt"], 0.0 if rew is None else float(rew), bool(r.get("acceptor")),
                r.get("n_tokens"), bool(r.get("hit_length_limit", r.get("truncated", False)))))
    return {lab + 1: group_metrics(rows, G) for lab, rows in by_label.items()}, len(prompts)


def load_run(path: Path, *, default_G: int | None, eval_stem: str) -> Run:
    task, G, seed = infer_identity(path, default_G)
    run = Run(path, task, G, seed, cap=infer_cap(path))
    if G is None:
        run.notes.append("G unknown (no manifest, no G<n> in path; pass --default-G)")

    hist = load_log_history(path)
    if not hist:
        run.notes.append("no trainer_state.json log_history")
    for e in hist:
        if "step" in e and "reward" in e:
            run.steps.setdefault(int(e["step"]), {}).update(
                {v: e[k] for k, v in TRL_KEYS.items() if k in e})
    if run.steps and G:
        cum_t, prev_tokens = 0.0, 0.0
        for s in sorted(run.steps):
            m = run.steps[s]
            if m.get("step_time") is not None:
                cum_t += m["step_time"]
                m["cum_gpu_hours"] = cum_t / 3600
            if m.get("mean_length") is not None:
                m["completion_tokens"] = m["mean_length"] * ROLLOUTS_PER_STEP
            if m.get("num_tokens") is not None:
                m["step_tokens"] = m["num_tokens"] - prev_tokens
                prev_tokens = m["num_tokens"]

    if G:
        roll, n_prompts = load_rollout_steps(path, G)
        if not roll:
            run.notes.append("no rollouts.jsonl: acceptance-based group metrics unavailable")
        for s, m in roll.items():
            run.steps.setdefault(s, {}).update(m)
        run.n_prompts = n_prompts or (_read_json(path / "manifest.json") or {}).get("n_rows")

    run.compute = _read_json(path / "compute.json")
    if run.compute is None:
        run.notes.append("no compute.json")
    weight = ((run.compute or {}).get("config") or {}).get("weight_per_token") or 8.0
    cum_completion = 0.0
    for s in sorted(run.steps):
        m = run.steps[s]
        if m.get("num_tokens") is not None:
            m["weighted_tokens"] = m["num_tokens"] * weight
        # Comparable x-axes for arms that run different numbers of steps.
        m["cum_rollouts"] = s * ROLLOUTS_PER_STEP
        if m.get("completion_tokens") is not None:
            cum_completion += m["completion_tokens"]
            m["cum_completion_tokens"] = cum_completion
        if G and run.n_prompts:
            m["epoch"] = s * (ROLLOUTS_PER_STEP // G) / run.n_prompts
        # Training correctness: verifier acceptance if rollouts exist, else TRL's mean reward.
        m["train_correct"] = m["accept_rate"] if m.get("accept_rate") is not None else m.get("trl_reward")

    for ck in path.glob("checkpoint-*"):
        step = _ckpt_step(ck)
        if step is None:
            continue
        for summ in ck.glob(f"eval/*/{eval_stem}.json"):
            d = _read_json(summ)
            if d:
                run.evals[step] = d
                run.eval_dirs[step] = summ.parent
    if not run.evals:
        run.notes.append(f"no checkpoint evals ({eval_stem}.json)")
    return run


# ---------------------------------------------------------------------------
# Baselines (evaluate.py output for the base model)
# ---------------------------------------------------------------------------
@dataclass
class Baseline:
    eval_summary: dict[str, Any] | None = None       # e.g. val_n500.json
    eval_per_prompt: dict[str, dict] = field(default_factory=dict)   # prompt_hash -> row
    train_summary: dict[str, Any] | None = None      # e.g. train_n500.json (k samples)

    def accuracy(self) -> float | None:
        m = (self.eval_summary or {}).get("metrics", {})
        return (m.get("greedy") or {}).get("accuracy")

    def predictions(self, G: int) -> dict[str, float] | None:
        for summ in (self.train_summary, self.eval_summary):
            pred = ((summ or {}).get("metrics", {}).get("sampled") or {}).get("group_predictions")
            if pred and str(G) in pred:
                return pred[str(G)]
        return None


def load_baseline(d: Path | None, eval_stem: str, train_stem: str) -> Baseline:
    b = Baseline()
    if d is None:
        return b
    b.eval_summary = _read_json(d / f"{eval_stem}.json")
    b.train_summary = _read_json(d / f"{train_stem}.json")
    pp = d / f"{eval_stem}_per_prompt.jsonl"
    if pp.is_file():
        for line in open(pp):
            r = json.loads(line)
            b.eval_per_prompt[r["prompt_hash"]] = r
    return b


# ---------------------------------------------------------------------------
# Per-run scalars
# ---------------------------------------------------------------------------
def _series(run: Run, key: str) -> list[tuple[int, float]]:
    return [(s, run.steps[s][key]) for s in sorted(run.steps) if run.steps[s].get(key) is not None]


def _window_mean(run: Run, key: str, lo: int, hi: int) -> float | None:
    return mean(v for s, v in _series(run, key) if lo <= s <= hi)


def eval_curve(run: Run, base_acc: float | None) -> list[dict[str, Any]]:
    """Accuracy per checkpoint with its cumulative cost; step 0 = base model."""
    pts = []
    if base_acc is not None:
        pts.append({"step": 0, "accuracy": base_acc, "cum_gpu_hours": 0.0, "weighted_tokens": 0.0})
    for step in sorted(run.evals):
        m = run.evals[step].get("metrics", {})
        acc = (m.get("greedy") or {}).get("accuracy")
        if acc is None:
            acc = ((m.get("sampled") or {}).get("pass_at") or {}).get("1")
        near = run.steps.get(step) or run.steps.get(max([s for s in run.steps if s <= step], default=-1), {})
        pts.append({"step": step, "accuracy": acc,
                    "cum_gpu_hours": near.get("cum_gpu_hours"),
                    "weighted_tokens": near.get("weighted_tokens")})
    return pts


def bucket_accuracy(run: Run, base: Baseline) -> dict[str, dict[str, Any]]:
    """Final-checkpoint greedy accuracy split by the base model's difficulty bucket."""
    if not run.evals or not base.eval_per_prompt:
        return {}
    final = max(run.evals)
    path = run.eval_dirs[final] / (_stem_of(run.evals[final]) + "_per_prompt.jsonl")
    if not path.is_file():
        return {}
    acc: dict[str, list[float]] = defaultdict(list)
    base_acc: dict[str, list[float]] = defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        b = base.eval_per_prompt.get(r["prompt_hash"])
        if not b or not b.get("bucket") or "greedy_accepted" not in r:
            continue
        acc[b["bucket"]].append(float(r["greedy_accepted"]))
        if "greedy_accepted" in b:
            base_acc[b["bucket"]].append(float(b["greedy_accepted"]))
    return {k: {"n": len(v), "accuracy": mean(v), "base_accuracy": mean(base_acc[k])}
            for k, v in acc.items()}


def _stem_of(summary: dict[str, Any]) -> str:
    lim = summary.get("limit")
    return f"{summary.get('split')}_n{lim}" if lim else str(summary.get("split"))


def run_scalars(run: Run, base: Baseline) -> dict[str, Any]:
    last = max(run.steps) if run.steps else 0
    lo_late = max(1, last - PHASE_STEPS + 1)
    sc: dict[str, Any] = {"run": run.name, "task": run.task, "arm": run.arm, "G": run.G,
                          "cap": run.cap, "seed": run.seed, "path": str(run.path), "steps": last,
                          "rollouts_total": last * ROLLOUTS_PER_STEP,
                          "epochs": (run.steps.get(last) or {}).get("epoch"),
                          "n_prompts": run.n_prompts}
    # reward diversity
    for key in ("trl_frac_zero_std", "frac_zero_std", "effective_rollout_frac",
                "mean_group_reward_std"):
        sc[f"{key}_mean"] = _window_mean(run, key, 1, last)
        sc[f"{key}_first{PHASE_STEPS}"] = _window_mean(run, key, 1, PHASE_STEPS)
        sc[f"{key}_last{PHASE_STEPS}"] = _window_mean(run, key, lo_late, last)
    if sc["effective_rollout_frac_mean"] is None and sc["trl_frac_zero_std_mean"] is not None:
        sc["effective_rollout_frac_mean"] = 1 - sc["trl_frac_zero_std_mean"]   # equal-size groups
    # all-correct / all-wrong
    for key in ("frac_all_accepted", "frac_none_accepted", "frac_mixed_acceptance"):
        sc[f"{key}_mean"] = _window_mean(run, key, 1, last)
        sc[f"{key}_early"] = _window_mean(run, key, 1, EARLY_STEPS)
        sc[f"{key}_last{PHASE_STEPS}"] = _window_mean(run, key, lo_late, last)
    pred = base.predictions(run.G) if run.G else None
    if pred:
        sc["pred_frac_all_accepted"] = pred.get("frac_all_accepted")
        sc["pred_frac_none_accepted"] = pred.get("frac_none_accepted")
        sc["pred_frac_zero_std"] = pred.get("frac_zero_std")
    # generation cost
    steps = [run.steps[s] for s in sorted(run.steps)]
    sc["completion_tokens_total"] = sum(m.get("completion_tokens") or 0 for m in steps) or None
    sc["completion_tokens_per_step"] = mean(m.get("completion_tokens") for m in steps)
    sc["weighted_tokens_total"] = steps[-1].get("weighted_tokens") if steps else None
    sc["train_gpu_hours"] = steps[-1].get("cum_gpu_hours") if steps else None
    sc["compute_gpu_hours"] = (run.compute or {}).get("gpu_hours")
    sc["step_time_mean_s"] = mean(m.get("step_time") for m in steps)
    sc["clipped_ratio_mean"] = mean(m.get("clipped_ratio") for m in steps)
    sc["train_reward_first"] = _window_mean(run, "trl_reward", 1, PHASE_STEPS)
    sc["train_reward_last"] = _window_mean(run, "trl_reward", lo_late, last)
    sc["train_correct_first"] = _window_mean(run, "train_correct", 1, PHASE_STEPS)
    sc["train_correct_last"] = _window_mean(run, "train_correct", lo_late, last)
    sc["hit_length_rate_mean"] = _window_mean(run, "hit_length_rate", 1, last)
    # learning efficiency + final accuracy
    curve = eval_curve(run, base.accuracy())
    accs = [p for p in curve if p["accuracy"] is not None]
    sc["base_accuracy"] = base.accuracy()
    if accs:
        final = max(accs, key=lambda p: p["step"])
        sc["final_step"] = final["step"]
        sc["final_accuracy"] = final["accuracy"]
        sc["best_accuracy"] = max(p["accuracy"] for p in accs)
        if sc["base_accuracy"] is not None:
            sc["gain"] = final["accuracy"] - sc["base_accuracy"]
            gh = final.get("cum_gpu_hours")
            sc["gain_per_gpu_hour"] = sc["gain"] / gh if gh else None
            wt = final.get("weighted_tokens")
            sc["gain_per_1e9_weighted_tokens"] = sc["gain"] / (wt / 1e9) if wt else None
        sc["accuracy_auc_steps"] = trapz_mean([p["step"] for p in accs], [p["accuracy"] for p in accs])
    for b, v in bucket_accuracy(run, base).items():
        sc[f"final_accuracy_{b}"] = v["accuracy"]
        sc[f"n_{b}"] = v["n"]
        sc[f"base_accuracy_{b}"] = v["base_accuracy"]
    sc["notes"] = "; ".join(run.notes)
    return sc


COMMON_AXES = (("epoch", "epochs"), ("cum_rollouts", "rollouts"),
               ("cum_completion_tokens", "completion tokens"), ("cum_gpu_hours", "GPU-hours"))


def train_correct_at_common(runs: list[Run], xkey: str, smooth: int) -> dict[str, Any]:
    """Each run's smoothed training correctness at the largest `xkey` every run reached."""
    curves = {}
    for r in runs:
        pts = [(m[xkey], m["train_correct"]) for _, m in sorted(r.steps.items())
               if m.get(xkey) is not None and m.get("train_correct") is not None]
        if len(pts) >= 2:
            ys = rolling([y for _, y in pts], smooth)
            curves[r.name] = ([x for x, _ in pts], ys)
    if len(curves) < 2:
        return {}
    budget = min(xs[-1] for xs, _ in curves.values())
    return {"budget": budget, **{n: interp(xs, ys, budget) for n, (xs, ys) in curves.items()}}


def accuracy_at_common_compute(runs: list[Run], base: Baseline, key: str) -> dict[str, float | None]:
    """Each run's accuracy interpolated at the largest `key` budget every run reached."""
    curves = {r.name: [p for p in eval_curve(r, base.accuracy())
                       if p["accuracy"] is not None and p[key] is not None] for r in runs}
    reach = [max(p[key] for p in c) for c in curves.values() if len(c) >= 2]
    if not reach:
        return {}
    budget = min(reach)
    return {"budget": budget, **{name: interp([p[key] for p in c], [p["accuracy"] for p in c], budget)
                                 for name, c in curves.items()}}


# ---------------------------------------------------------------------------
# Arm aggregation
# ---------------------------------------------------------------------------
ARM_METRICS = [
    ("trl_frac_zero_std_mean", "zero-std groups (TRL)"),
    ("effective_rollout_frac_mean", "rollouts with signal"),
    ("frac_all_accepted_mean", "all-accepted groups"),
    ("frac_none_accepted_mean", "none-accepted groups"),
    ("pred_frac_all_accepted", "predicted all-accepted (base)"),
    ("pred_frac_none_accepted", "predicted none-accepted (base)"),
    ("completion_tokens_per_step", "completion tokens / step"),
    ("weighted_tokens_total", "weighted tokens total"),
    ("train_gpu_hours", "GPU-hours (train steps)"),
    ("compute_gpu_hours", "GPU-hours (job)"),
    ("step_time_mean_s", "step time (s)"),
    ("clipped_ratio_mean", "length-capped share"),
    ("steps", "optimizer steps"),
    ("epochs", "epochs over train prompts"),
    ("rollouts_total", "rollouts"),
    ("train_correct_first", f"train correctness, first {PHASE_STEPS} steps"),
    ("train_correct_last", f"train correctness, last {PHASE_STEPS} steps"),
    ("train_correct_at_common_epoch", "train correctness at equal epochs"),
    ("train_correct_at_common_cum_rollouts", "train correctness at equal rollouts"),
    ("train_correct_at_common_cum_completion_tokens", "train correctness at equal tokens"),
    ("train_correct_at_common_cum_gpu_hours", "train correctness at equal GPU-hours"),
    ("final_accuracy", "final val accuracy"),
    ("gain", "gain over base"),
    ("gain_per_gpu_hour", "gain / GPU-hour"),
    ("accuracy_at_common_gpu_hours", "accuracy at common GPU-h"),
    ("accuracy_auc_steps", "mean accuracy over training"),
    ("final_accuracy_hard", "final acc, Hard"),
    ("final_accuracy_intermediate", "final acc, Intermediate"),
    ("final_accuracy_easy", "final acc, Easy"),
]


def aggregate_arms(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    arms: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        arms[(r["task"], r["G"], r.get("cap"))].append(r)
    out = []
    for (task, G, cap), rs in sorted(arms.items(),
                                     key=lambda kv: (str(kv[0][0]), kv[0][1] or 0, -(kv[0][2] or 0))):
        a: dict[str, Any] = {"task": task, "G": G, "cap": cap, "arm": rs[0].get("arm") or f"G={G}",
                             "n_runs": len(rs),
                             "seeds": ",".join(str(r["seed"]) for r in rs)}
        for key, _ in ARM_METRICS:
            vals = [r.get(key) for r in rs if r.get(key) is not None]
            a[f"{key}_mean"] = mean(vals)
            a[f"{key}_std"] = std(vals)
            a[f"{key}_n"] = len(vals)
        out.append(a)
    return out


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _fmt(v: Any, sd: Any = None) -> str:
    if v is None:
        return "–"
    if isinstance(v, float) and v == int(v) and abs(v) < 1e15 and sd is None:
        return f"{int(v):,}"
    if isinstance(v, float):
        big = abs(v) >= 1000
        s = f"{v:.3g}" if big else f"{v:.3f}"
        if sd is not None:
            s += f" ± {sd:.3g}" if big else f" ± {sd:.3f}"
        return s
    return str(v)


def markdown_report(arms: list[dict], rows: list[dict], figures: list[str],
                    common: dict[str, Any], base: Baseline, missing: list[str]) -> str:
    lines = ["# GRPO group-size ablation", ""]
    tasks = sorted({str(a["task"]) for a in arms})
    lines.append(f"Tasks: {', '.join(tasks)}. Runs: {len(rows)}. "
                 f"Base-model val accuracy: {_fmt(base.accuracy())}.")
    lines.append("")
    lines.append("Values are mean ± std over seeds (std blank with one seed). Every step is "
                 f"{ROLLOUTS_PER_STEP} rollouts; G sets prompts per step (64/G). Arms that train "
                 "for the same number of epochs therefore run different numbers of steps, "
                 "rollouts, tokens and GPU-hours. The 'at equal ...' rows compare arms at the "
                 "largest value of that axis that every arm reached.")
    lines.append("")
    for task in tasks:
        ta = [a for a in arms if str(a["task"]) == task]
        lines.append(f"## {task}")
        lines.append("")
        lines.append("| metric | " + " | ".join(f"{a['arm']} (n={a['n_runs']})" for a in ta) + " |")
        lines.append("|---|" + "---|" * len(ta))
        for key, label in ARM_METRICS:
            if all(a[f"{key}_n"] == 0 for a in ta):
                continue
            lines.append(f"| {label} | " + " | ".join(
                _fmt(a[f"{key}_mean"], a[f"{key}_std"]) for a in ta) + " |")
        lines.append("")
    for task, axes in sorted((common.get("train") or {}).items()):
        lines.append(f"{task}: equal-budget points are " + ", ".join(
            f"{label} = {v:.3g}" for label, v in axes.items()) + ".")
    if common.get("train"):
        lines.append("")
    if common.get("gpu_hours"):
        lines.append(f"Accuracy at common GPU-hours = each run's val accuracy interpolated at "
                     f"{common['gpu_hours']:.2f} GPU-h, the most every run reached.")
        lines.append("")
    if figures:
        lines.append("## Figures")
        lines.append("")
        for f in figures:
            lines.append(f"![{f}]({f})")
        lines.append("")
    lines.append("## Runs")
    lines.append("")
    lines.append("| run | task | arm | seed | steps | final acc | GPU-h | notes |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(f"| {r['run']} | {r['task']} | {r.get('arm') or r['G']} | {r['seed']} | {r['steps']} | "
                     f"{_fmt(r.get('final_accuracy'))} | {_fmt(r.get('train_gpu_hours'))} | "
                     f"{r.get('notes') or ''} |")
    if missing:
        lines += ["", "## Missing inputs", ""] + [f"- {m}" for m in missing]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Figures (matplotlib, static PNG; light surface)
# ---------------------------------------------------------------------------
# Categorical slots 1-3 of the validated reference palette (G=4, 8, 16 fixed order):
# all-pairs CVD-safe for three series; aqua is < 3:1 on the surface, so every line
# carries a direct end label and every figure has its CSV next to it.
G_COLORS = {4: "#2a78d6", 8: "#eb6834", 16: "#1baf7a", 2: "#e87ba4"}
SURFACE, INK, INK2, GRID, BASE_GRAY = "#fcfcfb", "#0b0b0b", "#52514e", "#e7e6e2", "#8a8984"


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=8)
    ax.set_xlabel(xlabel, fontsize=9, color=INK2)
    ax.set_ylabel(ylabel, fontsize=9, color=INK2)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def _end_label(ax, xs, ys, text, color) -> None:
    """Queue a direct label at the series' last point (drawn by _flush_end_labels)."""
    pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
    if pts:
        x, y = pts[-1]
        ax.plot([x], [y], "o", ms=4, color=color, mec=SURFACE, mew=1.5, zorder=3)
        if not hasattr(ax, "_end_labels"):
            ax._end_labels = []
        ax._end_labels.append((x, y, text))


def _flush_end_labels(ax) -> None:
    """Draw queued end labels, nudged apart vertically so they never overlap."""
    items = sorted(getattr(ax, "_end_labels", []), key=lambda t: t[1])
    if not items:
        return
    lo, hi = ax.get_ylim()
    x0, x1 = ax.get_xlim()
    gap, near = 0.07 * (hi - lo), 0.12 * (x1 - x0)
    placed: list[float] = []
    for i, (x, y, _) in enumerate(items):          # items sorted by y
        below = [placed[j] for j in range(i) if abs(items[j][0] - x) < near]
        placed.append(max([y] + [b + gap for b in below]))
    top = hi - 0.02 * (hi - lo)
    for i in sorted(range(len(items)), key=lambda i: -placed[i]):   # pull back under the top
        above = [placed[j] for j in range(len(items))
                 if placed[j] > placed[i] and abs(items[j][0] - items[i][0]) < near]
        placed[i] = min(placed[i], top, *(a - gap for a in above)) if above else min(placed[i], top)
    for (x, _, text), ly in zip(items, placed):
        # leading spaces clear the end dot; clip_on=False lets labels sit past the last tick
        ax.text(x, ly, "  " + text, fontsize=8, color=INK2, va="center", ha="left",
                clip_on=False)
    ax._end_labels = []


EXTRA_COLORS = ("#eda100", "#e87ba4", "#6250d6", "#e34948")   # arms beyond one-per-G


def arm_colors(runs: list[Run]) -> dict[str, str]:
    """Fixed color per arm. With one cap per task an arm is a G and takes G's color;
    extra-cap arms (e.g. G=4 L2048 next to G=4 L4096) take the next palette slots."""
    arms = sorted({(r.G or 0, -(r.cap or 0), r.arm) for r in runs})
    colors, used_G, extra = {}, set(), iter(EXTRA_COLORS)
    for G, _, arm in arms:
        if G not in used_G and G in G_COLORS:
            colors[arm] = G_COLORS[G]
            used_G.add(G)
        else:
            colors[arm] = next(extra, INK2)
    return colors


def arm_series(runs: list[Run], ykey: str, xkey: str, smooth: int) -> dict[str, tuple[list, list]]:
    """Per arm: (x, seed-mean of the rolling-mean y), aligned by optimizer step."""
    by_arm: dict[str, list[Run]] = defaultdict(list)
    for r in runs:
        by_arm[r.arm].append(r)
    out = {}
    for arm, rs in by_arm.items():
        per = []
        for r in rs:
            pts = [(s, m[xkey] if xkey != "step" else s, m[ykey]) for s, m in sorted(r.steps.items())
                   if m.get(ykey) is not None and (xkey == "step" or m.get(xkey) is not None)]
            if pts:
                ys = rolling([y for *_, y in pts], smooth)
                per.append({s: (x, y) for (s, x, _), y in zip(pts, ys)})
        if not per:
            continue
        steps = sorted(set().union(*per))
        xs = [mean(p[s][0] for p in per if s in p) for s in steps]
        ys = [mean(p[s][1] for p in per if s in p) for s in steps]
        out[arm] = (xs, ys)
    return out


X_LABELS = {"step": "optimizer step", "epoch": "epochs over training prompts",
            "cum_completion_tokens": "generated completion tokens (millions)",
            "cum_gpu_hours": "GPU-hours (training steps)", "cum_rollouts": "rollouts"}
X_SCALE = {"cum_completion_tokens": 1e6}


def make_figures(runs: list[Run], rows: list[dict], base: Baseline, out: Path,
                 smooth: int) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: skipping figures (CSVs still written)")
        return []
    figs: list[str] = []
    tasks = sorted({r.task for r in runs if r.task}) or [None]

    def save(fig, name, data_rows):
        for a in fig.axes:
            _flush_end_labels(a)
        fig.patch.set_facecolor(SURFACE)
        fig.tight_layout()
        fig.savefig(out / name, dpi=160, facecolor=SURFACE)
        plt.close(fig)
        write_csv(out / name.replace(".png", ".csv"), data_rows)
        figs.append(name)

    for task in tasks:
        tr = [r for r in runs if r.task == task]
        sfx = f"_{task}" if task else ""
        colors = arm_colors(tr)
        arm_G = {r.arm: r.G for r in tr}
        # Arms can run different numbers of steps: compare on epochs when we know them.
        xdef = "epoch" if all(any(m.get("epoch") is not None for m in r.steps.values())
                              for r in tr) else "step"

        def line_panel(ax, key, title, ylabel, pred_key=None, ylim=None, xkey=None):
            xkey = xkey or xdef
            scale = X_SCALE.get(xkey, 1)
            data = []
            ax._has_pred = False
            series = arm_series(tr, key, xkey, smooth)
            for arm in sorted(series, key=lambda a: (arm_G[a] or 0, a)):
                xs, ys = series[arm]
                xs = [x / scale for x in xs]
                c = colors[arm]
                ax.plot(xs, ys, color=c, lw=2, label=arm)
                _end_label(ax, xs, ys, arm, c)
                data += [{"arm": arm, xkey: x, key: y} for x, y in zip(xs, ys)]
                if pred_key:
                    pv = (base.predictions(arm_G[arm]) or {}).get(pred_key)
                    if pv is not None:
                        ax.axhline(pv, color=c, lw=1.2, ls=(0, (4, 3)), alpha=0.9)
                        ax._has_pred = True
            _style(ax, title, X_LABELS[xkey], ylabel)
            if ylim:
                ax.set_ylim(*ylim)
            if len(series) >= 2:
                ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
            return data, bool(series)

        def caption(ax, extra=""):
            ax.text(0, -0.28, f"rolling mean over {smooth} steps"
                    + ("; dashed = base-model prediction for that G" if ax._has_pred else "") + extra,
                    transform=ax.transAxes, fontsize=7.5, color=INK2)

        # 1. reward diversity
        key = "frac_zero_std" if any(_series(r, "frac_zero_std") for r in tr) else "trl_frac_zero_std"
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        data, ok = line_panel(ax, key, f"{task}: groups with zero reward std (no gradient)",
                              "fraction of groups", pred_key="frac_zero_std", ylim=(0, 1))
        if ok:
            caption(ax)
            save(fig, f"fig_reward_diversity{sfx}.png", data)
        else:
            plt.close(fig)
        # 2. all-correct / all-wrong (needs rollouts.jsonl)
        if any(_series(r, "frac_all_accepted") for r in tr):
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
            d1, _ = line_panel(axes[0], "frac_all_accepted", f"{task}: all-accepted groups",
                               "fraction of groups", "frac_all_accepted", (0, 1))
            d2, _ = line_panel(axes[1], "frac_none_accepted", f"{task}: none-accepted groups",
                               "", "frac_none_accepted", (0, 1))
            caption(axes[0])
            save(fig, f"fig_group_outcomes{sfx}.png", d1 + d2)
        # 3. generation cost
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        d1, ok1 = line_panel(axes[0], "completion_tokens", f"{task}: completion tokens per step",
                             "tokens (64 rollouts)")
        d2, ok2 = line_panel(axes[1], "step_time", f"{task}: time per optimizer step", "seconds")
        if ok1 or ok2:
            save(fig, f"fig_generation_cost{sfx}.png", d1 + d2)
        else:
            plt.close(fig)
        # 4. training efficiency: rollout correctness vs epochs / tokens / GPU-hours
        xkeys = [k for k in ("epoch", "cum_completion_tokens", "cum_gpu_hours")
                 if any(m.get(k) is not None for r in tr for m in r.steps.values())]
        if xkeys and any(_series(r, "train_correct") for r in tr):
            fig, axes = plt.subplots(1, len(xkeys), figsize=(4.4 * len(xkeys), 3.6), sharey=True,
                                     squeeze=False)
            data = []
            for ax, xk in zip(axes[0], xkeys):
                short = {"epoch": "epochs", "cum_completion_tokens": "generated tokens",
                         "cum_gpu_hours": "GPU-hours"}[xk]
                d, _ = line_panel(ax, "train_correct",
                                  (f"{task}: training correctness vs {short}" if ax is axes[0][0]
                                   else f"vs {short}"),
                                  "share of rollouts correct" if ax is axes[0][0] else "",
                                  ylim=(0, 1), xkey=xk)
                data += d
            caption(axes[0][0], "; correct = accepted by the verifier")
            save(fig, f"fig_training_efficiency{sfx}.png", data)
        # 5. learning efficiency: val accuracy vs steps / weighted tokens / GPU-hours
        curves = {r.name: (r.arm, eval_curve(r, base.accuracy())) for r in tr}
        if any(len([p for p in c if p["accuracy"] is not None]) >= 2 for _, c in curves.values()):
            fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
            data = []
            for ax, (xkey, xlabel, scale) in zip(axes, (
                    ("step", "optimizer step", 1), ("weighted_tokens", "weighted tokens (1e9)", 1e9),
                    ("cum_gpu_hours", "GPU-hours", 1))):
                by_arm: dict[str, list] = defaultdict(list)
                for name, (arm, c) in curves.items():
                    pts = [p for p in c if p["accuracy"] is not None and p[xkey] is not None]
                    if pts:
                        by_arm[arm].append(pts)
                        data += [{"run": name, "arm": arm, **p} for p in pts]
                for arm in sorted(by_arm, key=lambda a: (arm_G[a] or 0, a)):
                    plist, c = by_arm[arm], colors[arm]
                    for pts in plist:   # each seed: thin line; arm mean: thick
                        ax.plot([p[xkey] / scale for p in pts], [p["accuracy"] for p in pts],
                                color=c, lw=1, alpha=0.35)
                    steps = sorted({p["step"] for pts in plist for p in pts})
                    mx = [mean(next((p[xkey] for p in pts if p["step"] == s), None) for pts in plist)
                          for s in steps]
                    my = [mean(next((p["accuracy"] for p in pts if p["step"] == s), None)
                               for pts in plist) for s in steps]
                    xs = [x / scale for x in mx if x is not None]
                    ys = [y for x, y in zip(mx, my) if x is not None]
                    ax.plot(xs, ys, color=c, lw=2, marker="o", ms=4, mec=SURFACE, mew=1.5, label=arm)
                    _end_label(ax, xs, ys, arm, c)
                if base.accuracy() is not None:
                    ax.axhline(base.accuracy(), color=BASE_GRAY, lw=1.2, ls=(0, (4, 3)))
                _style(ax, f"{task}: val accuracy vs {xlabel.split(' (')[0]}", xlabel,
                       "val accuracy" if ax is axes[0] else "")
            if len(by_arm) >= 2:
                axes[0].legend(frameon=False, fontsize=8, labelcolor=INK2)
            axes[0].text(0, -0.28, "thin = single seed, thick = seed mean, gray dashed = base model",
                         transform=axes[0].transAxes, fontsize=7.5, color=INK2)
            save(fig, f"fig_learning_efficiency{sfx}.png", data)
        # 6. final accuracy by arm (+ by bucket)
        trows = [r for r in rows if r["task"] == task and r.get("final_accuracy") is not None]
        if trows:
            arms = sorted({r["arm"] for r in trows}, key=lambda a: (arm_G.get(a) or 0, a))
            buckets = [b for b in ("hard", "intermediate", "easy")
                       if any(r.get(f"final_accuracy_{b}") is not None for r in trows)]
            groups = ["all"] + buckets
            fig, ax = plt.subplots(figsize=(6.4 if not buckets else 8.5, 3.8))
            width = 0.8 / max(len(arms), 1)
            data = []
            for i, arm in enumerate(arms):
                c = colors[arm]
                for j, grp in enumerate(groups):
                    key = "final_accuracy" if grp == "all" else f"final_accuracy_{grp}"
                    vals = [r[key] for r in trows if r["arm"] == arm and r.get(key) is not None]
                    if not vals:
                        continue
                    x = j + (i - (len(arms) - 1) / 2) * width
                    m, sd = mean(vals), std(vals)
                    ax.bar(x, m, width * 0.92, color=c, label=arm if j == 0 else None,
                           edgecolor=SURFACE, linewidth=2)
                    if sd:
                        ax.errorbar(x, m, yerr=sd, color=INK2, lw=1, capsize=3)
                    ax.plot([x] * len(vals), vals, "o", ms=3.5, color=SURFACE, mec=INK2, mew=1)
                    top = max([m + (sd or 0)] + vals)
                    ax.annotate(f"{m:.2f}", (x, top), xytext=(0, 4), textcoords="offset points",
                                ha="center", fontsize=7, color=INK2)
                    data.append({"arm": arm, "group": grp, "mean": m, "std": sd, "values": vals})
            for j, grp in enumerate(groups):
                bkey = "base_accuracy" if grp == "all" else f"base_accuracy_{grp}"
                bv = mean(r.get(bkey) for r in trows)
                if bv is not None:
                    ax.hlines(bv, j - 0.45, j + 0.45, color=BASE_GRAY, lw=1.2, ls=(0, (4, 3)))
                    data.append({"group": grp, "base_accuracy": bv})
            ax.set_xticks(range(len(groups)))
            ax.set_xticklabels([g.capitalize() if g != "all" else "All prompts" for g in groups])
            _style(ax, f"{task}: final val accuracy by G", "", "val accuracy")
            ax.set_ylim(0, 1.08)
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            if len(arms) >= 2:
                ax.legend(frameon=False, fontsize=8, labelcolor=INK2, ncol=len(arms),
                          loc="lower right", bbox_to_anchor=(1, 1.0))
            ax.text(0, -0.2, "bars = seed mean, dots = seeds, gray dashed = base model; buckets "
                    "from base-model solve rate", transform=ax.transAxes, fontsize=7.5, color=INK2)
            save(fig, f"fig_final_accuracy{sfx}.png", data)
    return figs


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def analyze(paths: list[Path], out: Path, *, baseline_dir: Path | None = None,
            eval_stem: str = "val_n500", train_stem: str = "train_n500",
            default_G: int | None = None, smooth: int = 10, figures: bool = True,
            min_steps: int = 20) -> dict[str, Any]:
    run_dirs = find_run_dirs(paths)
    if not run_dirs:
        raise SystemExit(f"no run directories found under {[str(p) for p in paths]}")
    out.mkdir(parents=True, exist_ok=True)
    base = load_baseline(baseline_dir, eval_stem, train_stem)
    missing = []
    if baseline_dir is None:
        missing.append("no --baseline-dir: no base accuracy, predictions or buckets")
    else:
        if base.eval_summary is None:
            missing.append(f"{baseline_dir}/{eval_stem}.json not found (base accuracy)")
        if base.train_summary is None:
            missing.append(f"{baseline_dir}/{train_stem}.json not found (using {eval_stem} "
                           "for group predictions if it has k samples)")

    loaded = [load_run(d, default_G=default_G, eval_stem=eval_stem) for d in run_dirs]
    runs = [r for r in loaded if r.steps and max(r.steps) >= min_steps]
    for r in loaded:
        if r not in runs:
            missing.append(f"excluded {r.path} ({max(r.steps) if r.steps else 0} steps "
                           f"< --min-steps {min_steps})")
    if not runs:
        raise SystemExit(f"no run has >= {min_steps} logged steps (lower --min-steps)")
    for task in {r.task for r in runs}:      # arm labels: show the cap only if it varies
        tr = [r for r in runs if r.task == task]
        show_cap = len({r.cap for r in tr}) > 1
        for r in tr:
            r.arm = f"G={r.G}" + (f" L{r.cap}" if show_cap and r.cap else "")
    rows = [run_scalars(r, base) for r in runs]

    common: dict[str, Any] = {"train": {}}
    for task in {r.task for r in runs}:
        tr = [r for r in runs if r.task == task]
        # Compare arms that share a completion cap with each other (the G comparison);
        # an arm alone at its cap is compared against the whole task.
        by_cap: dict[Any, list[Run]] = defaultdict(list)
        for r in tr:
            by_cap[r.cap].append(r)
        groups = [g for g in by_cap.values() if len({r.arm for r in g}) > 1]
        lone = [r for g in by_cap.values() if len({x.arm for x in g}) <= 1 for r in g]
        for xkey, label in COMMON_AXES:
            for group, targets in [(g, g) for g in groups] + ([(tr, lone)] if lone else []):
                at = train_correct_at_common(group, xkey, max(smooth, 15))
                if not at:
                    continue
                budget = at.pop("budget")
                names = {r.name for r in targets}
                tag = "/".join(sorted({r.arm for r in targets}))
                common["train"].setdefault(str(task), {})[f"{label} [{tag}]"] = budget
                for row in rows:
                    if row["task"] == task and row["run"] in names and row["run"] in at:
                        row[f"train_correct_at_common_{xkey}"] = at[row["run"]]
        acc = accuracy_at_common_compute(tr, base, "cum_gpu_hours")
        if acc:
            common["gpu_hours"] = acc.pop("budget")
            for row in rows:
                if row["task"] == task and row["run"] in acc:
                    row["accuracy_at_common_gpu_hours"] = acc[row["run"]]
    arms = aggregate_arms(rows)

    step_rows = [{"run": r.name, "task": r.task, "arm": r.arm, "G": r.G, "seed": r.seed, "step": s, **m}
                 for r in runs for s, m in sorted(r.steps.items())]
    eval_rows = [{"run": r.name, "task": r.task, "G": r.G, "seed": r.seed, **p}
                 for r in runs for p in eval_curve(r, base.accuracy())]
    bucket_rows = [{"run": r.name, "task": r.task, "G": r.G, "seed": r.seed, "bucket": b, **v}
                   for r in runs for b, v in bucket_accuracy(r, base).items()]
    write_csv(out / "runs.csv", rows)
    write_csv(out / "arms.csv", arms)
    write_csv(out / "steps.csv", step_rows)
    write_csv(out / "evals.csv", eval_rows)
    write_csv(out / "buckets.csv", bucket_rows)
    figs = make_figures(runs, rows, base, out, smooth) if figures else []
    (out / "report.md").write_text(markdown_report(arms, rows, figs, common, base, missing))
    print(f"analyzed {len(runs)} runs -> {out}/report.md ({len(figs)} figures)")
    for r in rows:
        if r["notes"]:
            print(f"  {r['run']}: {r['notes']}")
    return {"runs": rows, "arms": arms, "figures": figs, "common": common}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", type=Path, help="run dirs, or dirs containing them")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--baseline-dir", type=Path, default=None,
                   help="evaluate.py output for the base model on this task")
    p.add_argument("--eval-stem", default="val_n500", help="checkpoint/baseline eval file stem")
    p.add_argument("--train-stem", default="train_n500", help="baseline k-sample file stem")
    p.add_argument("--default-G", type=int, default=None, help="G for runs without a manifest")
    p.add_argument("--smooth", type=int, default=10, help="rolling window for per-step curves")
    p.add_argument("--min-steps", type=int, default=20,
                   help="leave out shorter runs (smoke tests) from the arms")
    p.add_argument("--no-figures", action="store_true")
    a = p.parse_args(argv)
    analyze(a.paths, a.out, baseline_dir=a.baseline_dir, eval_stem=a.eval_stem,
            train_stem=a.train_stem, default_G=a.default_G, smooth=a.smooth,
            figures=not a.no_figures, min_steps=a.min_steps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
