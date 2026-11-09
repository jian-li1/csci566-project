"""
Rollout validation for the GRPO group-size ablation (owner: Mehul).

Proves, for one --G, that the shared trainer (grpo/grpo.py) does what the
ablation assumes:

  geometry   every optimizer step scores exactly 64 completions, built as
             64/G distinct prompts x G completions, in contiguous groups of G
             (TRL groups rewards with view(-1, G), so contiguity is required);
             one fresh generation per optimizer step.
  alignment  each reward belongs to the prompt and completion it was computed
             for: the entry passed with a completion rebuilds the same prompt,
             re-scoring gives the same reward, and the token ids TRL trains on
             are the ids that were scored. A planted reward checks TRL's group
             statistics and advantages end to end.
  CARC       the live mode runs the real grpo.main() (same flags, vLLM on), so
             a CARC job exercises the production path, not a copy of it.

Modes
-----
Offline (no GPU / TRL needed; works on any finished run dir):

    python -m grpo.rollout_validation check RUN_DIR --G 4
    python -m grpo.rollout_validation check RUN_DIR --G 4 \
        --task countdown --dataset data/countdown        # + re-score every row

Live (wraps grpo.main(); everything after "--" goes to grpo.grpo unchanged):

    python -m grpo.rollout_validation live -- \
        --task countdown --G 8 --dataset data/countdown --max-steps 2 \
        --report-to none --output-dir runs/validate-G8
    python -m grpo.rollout_validation live --planted -- <same grpo args>

Summary over many runs (e.g. the CARC array in experiments/rollout_validation/):

    python -m grpo.rollout_validation summary runs/validation

--planted returns reward 1 for the first completion of each group and 0 for
the rest (the real env rewards are still recorded to rollouts.jsonl). TRL must
then log reward == 1/G and frac_reward_zero_std == 0 every step, and its
advantages must equal the ones computed here.

Both modes write <run_dir>/rollout_validation[_<mode>].json and exit 1 on any
failed check. No shared code is edited: live mode swaps grpo.grpo's
make_env_reward_fn / GRPOTrainer module attributes before calling main().
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

ROLLOUTS_PER_STEP = 64          # mirrors grpo.grpo.ROLLOUTS_PER_STEP
MAX_EMPTY_RESPONSE_RATE = 0.01  # empty completions score 0 by design; more than this is a fault
REWARD_ATOL = 1e-5              # TRL stores rewards as float32
ADV_ATOL = 1e-3                 # float32 advantages, divided by (std + 1e-4)
MAX_EXAMPLES = 5                # failure examples kept per check


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
@dataclass
class Check:
    name: str
    description: str
    n_checked: int = 0
    n_failed: int = 0
    examples: list[str] = field(default_factory=list)
    skipped: str | None = None   # reason, if the check could not run

    @property
    def passed(self) -> bool:
        return self.skipped is None and self.n_failed == 0 and self.n_checked > 0

    def ok(self, n: int = 1) -> None:
        self.n_checked += n

    def fail(self, msg: str) -> None:
        self.n_checked += 1
        self.n_failed += 1
        if len(self.examples) < MAX_EXAMPLES:
            self.examples.append(msg)

    def expect(self, cond: bool, msg: str) -> bool:
        if cond:
            self.ok()
        else:
            self.fail(msg)
        return cond

    def to_dict(self) -> dict[str, Any]:
        status = "SKIP" if self.skipped else ("PASS" if self.passed else "FAIL")
        return {
            "status": status, "description": self.description,
            "n_checked": self.n_checked, "n_failed": self.n_failed,
            "examples": self.examples, "skipped": self.skipped,
        }


class Report:
    def __init__(self, mode: str, G: int, **meta: Any) -> None:
        self.mode, self.G, self.meta = mode, G, meta
        self.checks: dict[str, Check] = {}
        self.info: dict[str, Any] = {}

    def check(self, name: str, description: str) -> Check:
        if name not in self.checks:
            self.checks[name] = Check(name, description)
        return self.checks[name]

    @property
    def passed(self) -> bool:
        # Skipped checks are reported but do not fail the run.
        ran = [c for c in self.checks.values() if c.skipped is None]
        return bool(ran) and all(c.passed for c in ran)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode, "G": self.G, "passed": self.passed, **self.meta,
            "checks": {k: c.to_dict() for k, c in self.checks.items()},
            "info": self.info,
        }

    def print(self) -> None:
        print(f"\n=== rollout validation [{self.mode}] G={self.G} ===")
        for c in self.checks.values():
            d = c.to_dict()
            extra = f"  ({c.skipped})" if c.skipped else f"  {c.n_checked - c.n_failed}/{c.n_checked}"
            print(f"  {d['status']:<4}  {c.name:<28}{extra}")
            for ex in c.examples:
                print(f"          - {ex}")
        for k, v in self.info.items():
            print(f"  info  {k}: {v}")
        print(f"RESULT: {'PASS' if self.passed else 'FAIL'}\n")

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        print(f"Wrote {path}")


# ---------------------------------------------------------------------------
# Pure checks (shared by offline + live)
# ---------------------------------------------------------------------------
def _std(xs: Sequence[float]) -> float:
    """Unbiased std, matching trl.trainer.utils.nanstd (Bessel's correction)."""
    if len(xs) < 2:
        return float("nan")
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def _key(x: Any) -> str:
    return json.dumps(x, sort_keys=True, default=str)


def check_geometry(prompts: Sequence[str], G: int, rollouts_per_step: int,
                   chk: Check, label: str) -> None:
    """64 completions = 64/G contiguous blocks of G identical prompts, all distinct."""
    n = len(prompts)
    if not chk.expect(n == rollouts_per_step,
                      f"{label}: {n} completions, expected {rollouts_per_step}"):
        return
    blocks = [prompts[i:i + G] for i in range(0, n, G)]
    bad = [i for i, b in enumerate(blocks) if len(set(b)) != 1]
    chk.expect(not bad, f"{label}: groups {bad[:4]} mix prompts (not contiguous groups of {G})")
    distinct = len({b[0] for b in blocks})
    chk.expect(distinct == rollouts_per_step // G,
               f"{label}: {distinct} distinct prompts, expected {rollouts_per_step // G} "
               "(same prompt in two groups)")


def group_stats(rewards: Sequence[float], G: int) -> dict[str, float]:
    """The per-step numbers TRL logs, recomputed from rewards (sum_then_normalize)."""
    groups = [list(rewards[i:i + G]) for i in range(0, len(rewards), G)]
    zero = [abs(_std(g)) < 1e-8 for g in groups]
    return {
        "reward": sum(rewards) / len(rewards),
        "reward_std": _std(rewards),
        "frac_reward_zero_std": sum(zero) / len(zero),
    }


def expected_advantages(rewards: Sequence[float], G: int, scale_rewards: str) -> list[float]:
    """TRL GRPO advantages for one generation batch (sum_then_normalize path)."""
    adv: list[float] = []
    batch_std = _std(rewards)
    for i in range(0, len(rewards), G):
        g = list(rewards[i:i + G])
        mu = sum(g) / len(g)
        sd = _std(g) if scale_rewards in ("group", "none") else batch_std
        for r in g:
            a = r - mu
            if scale_rewards != "none":
                a = a / (sd + 1e-4)
            adv.append(a)
    return adv


def planted_rewards(n: int, G: int) -> list[float]:
    """1.0 for the first completion of each group, 0.0 for the rest."""
    return [1.0 if i % G == 0 else 0.0 for i in range(n)]


# ---------------------------------------------------------------------------
# Offline: validate a finished run directory
# ---------------------------------------------------------------------------
def load_rollouts(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "rollouts.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"No rollouts.jsonl in {run_dir}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def find_log_history(run_dir: Path) -> list[dict[str, Any]] | None:
    """log_history from run_dir/trainer_state.json, else the latest checkpoint's."""
    cands = [run_dir / "trainer_state.json"]
    ckpts = sorted(run_dir.glob("checkpoint-*/trainer_state.json"),
                   key=lambda p: int(p.parent.name.split("-")[-1]))
    cands += ckpts[::-1]
    for p in cands:
        if p.is_file():
            return json.loads(p.read_text()).get("log_history", [])
    return None


def _train_logs(log_history: Iterable[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(e["step"]): e for e in log_history if "step" in e and "reward" in e}


def validate_rows(rows: list[dict[str, Any]], G: int, report: Report, *,
                  rollouts_per_step: int = ROLLOUTS_PER_STEP,
                  max_completion_length: int | None = None,
                  log_history: list[dict[str, Any]] | None = None,
                  compare_rewards_to_trl: bool = True) -> None:
    """Checks on rollouts.jsonl rows. Each reward call appends one block of 64 rows."""
    geo = report.check("geometry", f"each step = {rollouts_per_step // G} prompts x {G} "
                       "completions in contiguous groups")
    lab = report.check("step_labels", "each 64-row block has one step label; labels run 0,1,2,...")
    scored = report.check("rewards_scored", "every rollout is scored (empty completions count "
                          f"as reward 0; allowed up to {MAX_EMPTY_RESPONSE_RATE:.0%})")
    n_empty = 0
    n_rem = len(rows) % rollouts_per_step
    if n_rem:
        geo.fail(f"{len(rows)} rows is not a multiple of {rollouts_per_step} "
                 f"(trailing partial block of {n_rem})")
    chunks = [rows[i:i + rollouts_per_step]
              for i in range(0, len(rows) - n_rem, rollouts_per_step)]

    labels: list[int] = []
    for k, chunk in enumerate(chunks):
        steps = {r.get("step") for r in chunk}
        lab.expect(len(steps) == 1, f"block {k}: step labels {sorted(map(str, steps))}")
        labels.append(chunk[0].get("step"))
        check_geometry([r["prompt"] for r in chunk], G, rollouts_per_step, geo, f"block {k}")
        for j, r in enumerate(chunk):
            rew = r.get("reward")
            if rew is None and r.get("fail") == "empty_response":
                # The model produced nothing: the env leaves it unscored and the reward
                # fn (and TRL) treat it as 0. A model outcome, not a pipeline fault.
                n_empty += 1
                scored.ok()
                continue
            scored.expect(isinstance(rew, (int, float)) and rew == rew,
                          f"block {k} row {j}: reward={rew!r} fail={r.get('fail')}")
    n_rows = sum(map(len, chunks))
    report.info["empty_responses"] = n_empty
    if n_rows and n_empty / n_rows > MAX_EMPTY_RESPONSE_RATE:
        scored.fail(f"{n_empty}/{n_rows} empty completions "
                    f"(> {MAX_EMPTY_RESPONSE_RATE:.0%}): check generation")
    lab.expect(labels == list(range(len(labels))),
               f"labels {labels[:12]}... are not 0..{len(labels) - 1} "
               "(repeats usually mean a resumed run appended to rollouts.jsonl)")

    if max_completion_length is not None:
        ln = report.check("completion_length", f"n_tokens <= {max_completion_length}")
        for k, chunk in enumerate(chunks):
            for j, r in enumerate(chunk):
                n = r.get("n_tokens")
                if n is None:
                    ln.fail(f"block {k} row {j}: n_tokens missing")
                else:
                    ln.expect(n <= max_completion_length, f"block {k} row {j}: n_tokens={n}")

    # Fresh prompts each step (RepeatSampler without replacement within an epoch).
    seen: Counter[str] = Counter()
    for chunk in chunks:
        for p in {r["prompt"] for r in chunk}:
            seen[p] += 1
    report.info["steps"] = len(chunks)
    report.info["distinct_prompts"] = len(seen)
    report.info["prompts_seen_in_more_than_one_step"] = sum(c > 1 for c in seen.values())

    # Cross-check against what TRL itself logged for the same steps: proves the
    # rows we recorded are the batch TRL trained on, grouped the same way.
    trl = report.check("matches_trl_logs", "per-step reward, reward_std, frac_reward_zero_std "
                       "and mean length equal TRL's log_history")
    if log_history is None:
        trl.skipped = "no trainer_state.json found"
    elif not compare_rewards_to_trl:
        trl.skipped = "planted run: TRL saw planted rewards, rollouts.jsonl has real ones"
    elif labels != list(range(len(labels))):
        trl.skipped = "step labels not contiguous; cannot map blocks to TRL steps"
    else:
        logs = _train_logs(log_history)
        for k, chunk in enumerate(chunks):
            e = logs.get(k + 1)   # metrics.step lags global_step by one
            if e is None:
                continue
            rewards = [float(r["reward"] or 0.0) for r in chunk]
            ours = group_stats(rewards, G)
            for key in ("reward", "reward_std", "frac_reward_zero_std"):
                if key in e:
                    trl.expect(abs(ours[key] - e[key]) < 1e-4,
                               f"step {k + 1} {key}: rollouts={ours[key]:.6f} trl={e[key]:.6f}")
            lens = [r["n_tokens"] for r in chunk if r.get("n_tokens") is not None]
            if "completions/mean_length" in e and len(lens) == len(chunk):
                m = sum(lens) / len(lens)
                trl.expect(abs(m - e["completions/mean_length"]) < 1e-2,
                           f"step {k + 1} mean_length: rollouts={m:.3f} "
                           f"trl={e['completions/mean_length']:.3f}")
        if trl.n_checked == 0:
            trl.skipped = "no logged steps overlap the rollout blocks"


def rescore_rows(rows: list[dict[str, Any]], prompt_to_entry: dict[str, Any],
                 make_env: Callable[[], Any], report: Report,
                 unscored_reward: float = 0.0) -> None:
    """Re-score each rollout against the dataset entry whose prompt it carries."""
    known = report.check("prompt_in_dataset", "every rollout prompt is a dataset prompt")
    same = report.check("rescore_matches", "re-scoring completion vs its prompt's entry "
                        "reproduces the recorded reward")
    env = make_env()
    for i, r in enumerate(rows):
        entry = prompt_to_entry.get(r["prompt"])
        if not known.expect(entry is not None, f"row {i}: prompt not found in dataset"):
            continue
        env.load_problem(entry)
        out = env.step(r["completion"])
        new = unscored_reward if out.get("reward") is None else float(out["reward"])
        old = unscored_reward if r.get("reward") is None else float(r["reward"])
        same.expect(abs(new - old) < REWARD_ATOL,
                    f"row {i} (step {r.get('step')}): recorded {old} vs re-scored {new}")


def validate_run_dir(run_dir: Path, G: int, *, task: str | None = None,
                     dataset: str | None = None, dataset_split: str = "train",
                     max_completion_length: int | None = None,
                     planted: bool = False) -> Report:
    rows = load_rollouts(run_dir)
    report = Report("offline", G, run_dir=str(run_dir), task=task, n_rows=len(rows))
    validate_rows(rows, G, report, max_completion_length=max_completion_length,
                  log_history=find_log_history(run_dir),
                  compare_rewards_to_trl=not planted)
    if dataset:
        if not task:
            raise ValueError("--dataset needs --task")
        import grpo.grpo as g   # needs the training deps (datasets, trl, ...)
        ds = g.load_train_dataset(dataset, dataset_split, task=task)
        rescore_rows(rows, {p: e for p, e in zip(ds["prompt"], ds["entry"])},
                     lambda: g.make_env_for_task(task), report)
    else:
        report.check("rescore_matches", "re-score against dataset").skipped = "no --dataset given"
    return report


# ---------------------------------------------------------------------------
# Live: wrap the real reward function and trainer inside grpo.main()
# ---------------------------------------------------------------------------
class LiveValidator:
    """Holds the per-step checks done inside the reward fn and the trainer."""

    def __init__(self, G: int, *, rollouts_per_step: int, max_completion_length: int | None,
                 make_env: Callable[[str], Any], planted: bool = False) -> None:
        self.G, self.rollouts_per_step = G, rollouts_per_step
        self.max_completion_length = max_completion_length
        self.make_env, self.planted = make_env, planted
        self.report = Report("live", G, planted=planted)
        self.calls = 0
        self.trainer = None
        self._last: dict[str, Any] | None = None
        self._envs: dict[str, Any] = {}
        r = self.report
        self.c_geo = r.check("geometry", f"each reward call = {rollouts_per_step // G} prompts "
                             f"x {G} completions in contiguous groups")
        self.c_entry = r.check("group_entry_identical", "all G completions of a group carry "
                               "the same dataset entry")
        self.c_prompt = r.check("prompt_matches_entry", "prompt == env.load_problem(entry)"
                                "['prompt'] for every completion")
        self.c_rescore = r.check("rescore_matches", "re-scoring with a fresh env reproduces "
                                 "the reward")
        self.c_len = r.check("completion_length", f"len(completion_ids) <= {max_completion_length}")
        self.c_ids = r.check("trained_ids_match_scored", "completion ids TRL trains on == ids "
                             "passed to the reward fn; prompt ids decode to the prompt")
        self.c_adv = r.check("advantages_match", "TRL advantages == group-normalized rewards "
                             "computed here")
        self.c_fresh = r.check("one_generation_per_step", "reward fn called once per optimizer "
                               "step (fresh rollouts)")
        if planted:
            self.c_plant = r.check("planted_reward_logged", "TRL logs reward == 1/G and "
                                   "frac_reward_zero_std == 0 every step")

    def _env(self, task: str):
        if task not in self._envs:
            self._envs[task] = self.make_env(task)
        return self._envs[task]

    # ---- reward-fn side -------------------------------------------------------
    def wrap_reward(self, real_fn: Callable[..., list[float]]) -> Callable[..., list[float]]:
        def env_reward(prompts, completions, task, entry, completion_ids=None, **kw):
            self.calls += 1
            label = f"call {self.calls}"
            ptxt = [p if isinstance(p, str) else p[-1]["content"] for p in prompts]
            ctxt = [c if isinstance(c, str) else c[-1]["content"] for c in completions]
            check_geometry(ptxt, self.G, self.rollouts_per_step, self.c_geo, label)

            for i in range(0, len(entry), self.G):
                keys = {_key(e) for e in entry[i:i + self.G]}
                self.c_entry.expect(len(keys) == 1, f"{label} group {i // self.G}: "
                                    f"{len(keys)} different entries")
            for i, (p, t, e) in enumerate(zip(ptxt, task, entry)):
                rebuilt = self._env(t).load_problem(e)["prompt"]
                self.c_prompt.expect(rebuilt == p, f"{label} row {i}: prompt != rebuilt prompt")
            if completion_ids is not None and self.max_completion_length is not None:
                for i, ids in enumerate(completion_ids):
                    self.c_len.expect(len(ids) <= self.max_completion_length,
                                      f"{label} row {i}: {len(ids)} tokens")

            real = real_fn(prompts, completions, task=task, entry=entry,
                           completion_ids=completion_ids, **kw)

            for i, (c, t, e) in enumerate(zip(ctxt, task, entry)):
                env = self._env(t)
                env.load_problem(e)
                out = env.step(c)
                new = 0.0 if out.get("reward") is None else float(out["reward"])
                self.c_rescore.expect(abs(new - real[i]) < REWARD_ATOL,
                                      f"{label} row {i}: reward {real[i]} vs re-scored {new}")

            returned = planted_rewards(len(real), self.G) if self.planted else list(real)
            self._last = {"prompts": ptxt, "completion_ids": completion_ids,
                          "rewards": returned}
            return returned

        env_reward.__name__ = "env_reward"   # keep TRL's rewards/env_reward/* log keys
        return env_reward

    # ---- trainer side -----------------------------------------------------------
    def check_trainer_output(self, trainer, out: dict[str, Any]) -> None:
        last, self._last = self._last, None
        if last is None:
            self.c_adv.fail("trainer produced a batch without a reward call")
            return
        label = f"call {self.calls}"
        adv = out["advantages"].detach().float().cpu().tolist()
        scale = getattr(trainer, "scale_rewards", "group")
        scale = {True: "group", False: "none"}.get(scale, scale)   # older TRL used a bool
        exp = expected_advantages(last["rewards"], self.G, str(scale))
        if self.c_adv.expect(len(adv) == len(exp), f"{label}: {len(adv)} advantages, "
                             f"expected {len(exp)}"):
            worst = max(abs(a - b) for a, b in zip(adv, exp))
            self.c_adv.expect(worst < ADV_ATOL, f"{label}: max |adv diff| = {worst:.2e}")

        cids, cmask = out["completion_ids"].cpu(), out["completion_mask"].cpu()
        pids, pmask = out["prompt_ids"].cpu(), out["prompt_mask"].cpu()
        tok = getattr(trainer.processing_class, "tokenizer", trainer.processing_class)
        scored_ids = last["completion_ids"]
        for i in range(cids.shape[0]):
            if scored_ids is not None:
                trained = cids[i][: int(cmask[i].sum())].tolist()
                self.c_ids.expect(trained == list(scored_ids[i]),
                                  f"{label} row {i}: trained completion ids != scored ids")
            ptoks = pids[i][pmask[i].bool()].tolist()
            dec = tok.decode(ptoks, skip_special_tokens=True)
            self.c_ids.expect(dec.strip() == last["prompts"][i].strip(),
                              f"{label} row {i}: prompt ids decode to a different prompt")

    def finish(self, trainer) -> Report:
        steps = int(trainer.state.global_step)
        self.c_fresh.expect(self.calls == steps,
                            f"{self.calls} reward calls for {steps} optimizer steps")
        if self.planted:
            for step, e in sorted(_train_logs(trainer.state.log_history).items()):
                self.c_plant.expect(abs(e["reward"] - 1.0 / self.G) < 1e-6,
                                    f"step {step}: reward {e['reward']} != 1/G")
                if "frac_reward_zero_std" in e:
                    self.c_plant.expect(e["frac_reward_zero_std"] == 0.0,
                                        f"step {step}: frac_reward_zero_std "
                                        f"{e['frac_reward_zero_std']}")
        for name in ("completion_length", "trained_ids_match_scored"):
            c = self.report.checks[name]
            if c.n_checked == 0:
                c.skipped = "no completion_ids from TRL" if name == "completion_length" \
                    else "trainer hook did not run"
        return self.report


def run_live(grpo_argv: list[str], *, planted: bool = False,
             model_id: str | None = None) -> Report:
    import grpo.grpo as g

    sys.argv = ["grpo.grpo", *grpo_argv]
    args = g.parse_args()
    if not args.output_dir:
        raise SystemExit("pass --output-dir so the validation report has a fixed home")
    out_dir = Path(args.output_dir)
    if (out_dir / "rollouts.jsonl").exists():
        raise SystemExit(f"{out_dir}/rollouts.jsonl exists; use a fresh --output-dir")
    if args.epochs is not None:
        raise SystemExit("use --max-steps for validation runs (not --epochs)")
    if model_id:
        g.MODEL_ID = model_id

    v = LiveValidator(args.G, rollouts_per_step=g.ROLLOUTS_PER_STEP,
                      max_completion_length=args.completion_tokens,
                      make_env=g.make_env_for_task, planted=planted)
    v.report.meta.update(task=args.task, model_id=g.MODEL_ID, argv=grpo_argv,
                         use_vllm=not args.no_vllm, run_dir=str(out_dir))

    orig_make_reward = g.make_env_reward_fn

    def make_env_reward_fn(envs, metrics, *a, **kw):
        return v.wrap_reward(orig_make_reward(envs, metrics, *a, **kw))

    base = g.GRPOTrainer

    class ValidatingGRPOTrainer(base):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            v.trainer = self

        def _generate_and_score_completions(self, inputs):
            out = super()._generate_and_score_completions(inputs)
            if self.model.training:
                v.check_trainer_output(self, out)
            return out

    g.make_env_reward_fn = make_env_reward_fn
    g.GRPOTrainer = ValidatingGRPOTrainer
    try:
        g.main()
    finally:
        g.make_env_reward_fn, g.GRPOTrainer = orig_make_reward, base

    report = v.finish(v.trainer)
    try:
        import trl
        report.meta["trl_version"] = trl.__version__
    except Exception:
        pass

    # Same offline checks on what landed on disk (rollouts.jsonl + log_history).
    rows = load_rollouts(out_dir)
    validate_rows(rows, args.G, report, rollouts_per_step=g.ROLLOUTS_PER_STEP,
                  max_completion_length=args.completion_tokens,
                  log_history=v.trainer.state.log_history,
                  compare_rewards_to_trl=not planted)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def summarize_reports(root: Path) -> int:
    """One line per rollout_validation*.json under root; exit 1 if any failed."""
    paths = sorted(root.rglob("rollout_validation*.json"))
    if not paths:
        print(f"No rollout_validation*.json under {root}")
        return 1
    ok = True
    print(f"{'run':<52}{'report':<34}{'G':>3}  result  failed checks")
    for p in paths:
        d = json.loads(p.read_text())
        failed = [k for k, c in d["checks"].items() if c["status"] == "FAIL"]
        ok &= bool(d["passed"])
        print(f"{p.parent.name:<52}{p.name:<34}{d['G']:>3}  "
              f"{'PASS' if d['passed'] else 'FAIL':<6}  {', '.join(failed)}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "summary":
        return summarize_reports(Path(argv[1] if len(argv) > 1 else "runs/validation"))
    if argv and argv[0] == "live":
        rest = argv[1:]
        grpo_argv: list[str] = []
        if "--" in rest:
            i = rest.index("--")
            rest, grpo_argv = rest[:i], rest[i + 1:]
        p = argparse.ArgumentParser(prog="rollout_validation live")
        p.add_argument("--planted", action="store_true",
                       help="return reward 1 for the first completion of each group, 0 otherwise")
        p.add_argument("--model-id", default=None,
                       help="override grpo.grpo.MODEL_ID (e.g. a tiny model for a CPU test)")
        a = p.parse_args(rest)
        report = run_live(grpo_argv, planted=a.planted, model_id=a.model_id)
        out = Path(report.meta["run_dir"]) / (
            "rollout_validation_planted.json" if a.planted else "rollout_validation.json")
    else:
        if argv and argv[0] == "check":
            argv = argv[1:]
        p = argparse.ArgumentParser(prog="rollout_validation check")
        p.add_argument("run_dir", type=Path)
        p.add_argument("--G", type=int, required=True, choices=(2, 4, 8, 16))
        p.add_argument("--task", default=None)
        p.add_argument("--dataset", default=None, help="re-score every rollout against this data")
        p.add_argument("--dataset-split", default="train")
        p.add_argument("--max-completion-length", type=int, default=None)
        p.add_argument("--planted", action="store_true", help="run was a --planted live run")
        p.add_argument("--report-dir", type=Path, default=None,
                       help="also write the report here as <run name>.json (e.g. runs/validate)")
        a = p.parse_args(argv)
        report = validate_run_dir(a.run_dir, a.G, task=a.task, dataset=a.dataset,
                                  dataset_split=a.dataset_split,
                                  max_completion_length=a.max_completion_length,
                                  planted=a.planted)
        out = a.run_dir / "rollout_validation_offline.json"
        if a.report_dir:
            report.write(a.report_dir / f"{a.run_dir.resolve().name}.json")
    report.print()
    report.write(out)
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
