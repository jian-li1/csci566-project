"""
Tests for grpo/rollout_validation.py.

Fast tests need no GPU or TRL:

    python -m unittest grpo.test_rollout_validation

The tiny-model live test runs the real grpo.main() on CPU with a ~5 MB model
(needs torch/trl/peft/datasets/reasoning-gym and network for the model):

    RUN_LIVE_GRPO_TEST=1 python -m unittest grpo.test_rollout_validation
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from grpo.rollout_validation import (
    LiveValidator,
    Report,
    check_geometry,
    expected_advantages,
    group_stats,
    planted_rewards,
    rescore_rows,
    validate_rows,
)

N = 64


def _prompts(G: int, step: int = 0) -> list[str]:
    return [f"s{step}-p{i // G}" for i in range(N)]


def _rows(G: int, steps: int = 2, reward_fn=lambda i: float(i % 3 == 0)) -> list[dict]:
    rows = []
    for s in range(steps):
        for i, p in enumerate(_prompts(G, s)):
            rows.append({"step": s, "task": "toy", "prompt": p, "completion": f"c{i}",
                         "reward": reward_fn(i), "acceptor": False, "n_tokens": 10})
    return rows


def _trl_logs(rows: list[dict], G: int) -> list[dict]:
    logs = []
    for k in range(len(rows) // N):
        chunk = rows[k * N:(k + 1) * N]
        st = group_stats([r["reward"] for r in chunk], G)
        logs.append({"step": k + 1, **st, "completions/mean_length": 10.0})
    return logs


class GeometryTests(unittest.TestCase):
    def run_geo(self, prompts, G):
        r = Report("t", G)
        c = r.check("geometry", "")
        check_geometry(prompts, G, N, c, "x")
        return c

    def test_contiguous_groups_pass_for_each_G(self):
        for G in (2, 4, 8, 16):
            self.assertTrue(self.run_geo(_prompts(G), G).passed, G)

    def test_interleaved_groups_fail(self):
        # p0 p1 p2 ... p15 p0 p1 ... : right counts, wrong layout for view(-1, G)
        G = 4
        interleaved = [f"p{i % (N // G)}" for i in range(N)]
        self.assertFalse(self.run_geo(interleaved, G).passed)

    def test_wrong_batch_size_fails(self):
        self.assertFalse(self.run_geo(_prompts(4)[:32], 4).passed)

    def test_duplicate_prompt_across_groups_fails(self):
        p = _prompts(8)
        p[8:16] = [p[0]] * 8      # group 1 repeats group 0's prompt
        self.assertFalse(self.run_geo(p, 8).passed)

    def test_G_mismatch_fails(self):
        # data built with G=8 checked as G=4: groups fine, but only 8 distinct prompts
        self.assertFalse(self.run_geo(_prompts(8), 4).passed)


class AdvantageTests(unittest.TestCase):
    def test_planted_group_advantages(self):
        for G in (4, 8, 16):
            r = planted_rewards(N, G)
            self.assertAlmostEqual(sum(r) / N, 1 / G)
            adv = expected_advantages(r, G, "group")
            # first of group: (1-1/G)/std ; rest: (-1/G)/std ; std = sqrt(1/G) (unbiased)
            sd = (1 / G) ** 0.5
            self.assertAlmostEqual(adv[0], (1 - 1 / G) / (sd + 1e-4), places=6)
            self.assertAlmostEqual(adv[1], (-1 / G) / (sd + 1e-4), places=6)
            self.assertEqual(group_stats(r, G)["frac_reward_zero_std"], 0.0)

    def test_zero_std_group_gives_zero_advantage(self):
        adv = expected_advantages([1.0] * 4 + [0.0, 1.0, 0.0, 1.0], 4, "group")
        self.assertEqual(adv[:4], [0.0] * 4)
        self.assertEqual(group_stats([1.0] * 4 + [0.0, 1.0, 0.0, 1.0], 4)
                         ["frac_reward_zero_std"], 0.5)

    def test_scale_none_is_mean_centered_only(self):
        self.assertEqual(expected_advantages([1.0, 0.0], 2, "none"), [0.5, -0.5])

    def test_matches_torch_formula_if_available(self):
        try:
            import torch
            from trl.trainer.utils import nanstd
        except Exception:
            self.skipTest("torch/trl not installed")
        G = 8
        rewards = [((i * 37) % 11) / 10 for i in range(N)]
        t = torch.tensor(rewards)
        mean = t.view(-1, G).mean(1).repeat_interleave(G)
        std = nanstd(t.view(-1, G), dim=1).repeat_interleave(G)
        want = ((t - mean) / (std + 1e-4)).tolist()
        got = expected_advantages(rewards, G, "group")
        self.assertLess(max(abs(a - b) for a, b in zip(got, want)), 1e-4)


class OfflineRowTests(unittest.TestCase):
    def test_good_run_passes(self):
        rows = _rows(4)
        r = Report("t", 4)
        validate_rows(rows, 4, r, max_completion_length=16, log_history=_trl_logs(rows, 4))
        self.assertTrue(r.passed, r.to_dict())
        self.assertEqual(r.checks["matches_trl_logs"].n_failed, 0)
        self.assertGreater(r.checks["matches_trl_logs"].n_checked, 0)

    def test_reward_permuted_against_trl_fails(self):
        # TRL saw one batch of rewards, rollouts.jsonl holds a different grouping
        rows = _rows(4)
        logs = _trl_logs(rows, 4)
        rs = [r["reward"] for r in rows[:N]]
        for r, new in zip(rows[:N], sorted(rs)):     # cluster the 1s into whole groups
            r["reward"] = new
        r = Report("t", 4)
        validate_rows(rows, 4, r, log_history=logs)
        self.assertFalse(r.checks["matches_trl_logs"].passed)

    def test_resumed_run_labels_flagged(self):
        rows = _rows(4, steps=3)
        for r in rows[2 * N:]:
            r["step"] = 1                                # resumed run relabels from 1
        r = Report("t", 4)
        validate_rows(rows, 4, r, log_history=_trl_logs(rows, 4))
        self.assertFalse(r.checks["step_labels"].passed)
        self.assertIsNotNone(r.checks["matches_trl_logs"].skipped)

    def test_partial_block_and_unscored_fail(self):
        rows = _rows(4)[:-3]
        rows[0]["reward"] = None
        r = Report("t", 4)
        validate_rows(rows, 4, r)
        self.assertFalse(r.checks["geometry"].passed)
        self.assertFalse(r.checks["rewards_scored"].passed)

    def test_empty_responses_tolerated_up_to_threshold(self):
        rows = _rows(4)                                  # 128 rows
        rows[3].update(reward=None, fail="empty_response")
        r = Report("t", 4)
        validate_rows(rows, 4, r)
        self.assertTrue(r.checks["rewards_scored"].passed)
        self.assertEqual(r.info["empty_responses"], 1)
        for row in rows[:5]:                             # 5/128 > 1%
            row.update(reward=None, fail="empty_response")
        r = Report("t", 4)
        validate_rows(rows, 4, r)
        self.assertFalse(r.checks["rewards_scored"].passed)

    def test_verifier_crash_still_fails(self):
        rows = _rows(4)
        rows[3].update(reward=None, fail="verifier_crash")
        r = Report("t", 4)
        validate_rows(rows, 4, r)
        self.assertFalse(r.checks["rewards_scored"].passed)

    def test_overlong_completion_fails(self):
        rows = _rows(4)
        rows[5]["n_tokens"] = 99
        r = Report("t", 4)
        validate_rows(rows, 4, r, max_completion_length=16)
        self.assertFalse(r.checks["completion_length"].passed)


class ToyEnv:
    """Answer is correct iff completion == entry['answer']."""

    def load_problem(self, entry):
        self.entry = entry
        return {"prompt": f"Q: {entry['q']}"}

    def step(self, text):
        ok = text == self.entry["answer"]
        return {"reward": float(ok), "acceptor": ok, "check": {}}


def _toy_batch(G):
    entries = [{"q": f"q{i // G}", "answer": f"a{i // G}"} for i in range(N)]
    prompts = [f"Q: {e['q']}" for e in entries]
    # one correct completion per group, at position (group index % G)
    completions = [f"a{i // G}" if i % G == (i // G) % G else "wrong" for i in range(N)]
    return prompts, completions, ["toy"] * N, entries


def _real_reward(prompts, completions, task, entry, completion_ids=None, **_):
    env = ToyEnv()
    out = []
    for c, e in zip(completions, entry):
        env.load_problem(e)
        out.append(env.step(c)["reward"])
    return out


class LiveWrapperTests(unittest.TestCase):
    def make(self, G, planted=False):
        return LiveValidator(G, rollouts_per_step=N, max_completion_length=8,
                             make_env=lambda t: ToyEnv(), planted=planted)

    def test_aligned_reward_fn_passes(self):
        G = 4
        v = self.make(G)
        p, c, t, e = _toy_batch(G)
        got = v.wrap_reward(_real_reward)(p, c, t, e, completion_ids=[[1, 2]] * N)
        self.assertEqual(got, _real_reward(p, c, t, e))
        for name in ("geometry", "group_entry_identical", "prompt_matches_entry",
                     "rescore_matches", "completion_length"):
            self.assertTrue(v.report.checks[name].passed, name)

    def test_misaligned_reward_fn_is_caught(self):
        G = 4
        v = self.make(G)
        p, c, t, e = _toy_batch(G)

        def shifted(prompts, completions, task, entry, **kw):
            r = _real_reward(prompts, completions, task, entry)
            return r[1:] + r[:1]           # off-by-one: reward i belongs to completion i+1

        v.wrap_reward(shifted)(p, c, t, e)
        self.assertFalse(v.report.checks["rescore_matches"].passed)

    def test_entry_prompt_mismatch_is_caught(self):
        G = 4
        v = self.make(G)
        p, c, t, e = _toy_batch(G)
        e = e[:]
        e[5] = {"q": "other", "answer": "x"}     # entry no longer rebuilds its prompt
        v.wrap_reward(_real_reward)(p, c, t, e)
        self.assertFalse(v.report.checks["prompt_matches_entry"].passed)
        self.assertFalse(v.report.checks["group_entry_identical"].passed)

    def test_planted_returns_planted(self):
        v = self.make(8, planted=True)
        p, c, t, e = _toy_batch(8)
        self.assertEqual(v.wrap_reward(_real_reward)(p, c, t, e), planted_rewards(N, 8))
        self.assertTrue(v.report.checks["rescore_matches"].passed)


@unittest.skipUnless(os.environ.get("RUN_LIVE_GRPO_TEST") == "1", "set RUN_LIVE_GRPO_TEST=1")
class TinyModelLiveTest(unittest.TestCase):
    """Real grpo.main() on CPU with a tiny Qwen2 model; one run per G, planted reward."""

    MODEL = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"

    def test_planted_all_G(self):
        import reasoning_gym

        from grpo.rollout_validation import run_live

        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data"
            data.mkdir()
            ds = reasoning_gym.create_dataset("countdown", size=200, seed=1)
            with open(data / "train.jsonl", "w") as f:
                for x in ds:
                    f.write(json.dumps({"question": x["question"], "answer": x["answer"],
                                        "metadata": json.dumps(x["metadata"])}) + "\n")
            for G in (4, 8, 16):
                with self.subTest(G=G):
                    out = Path(tmp) / f"G{G}"
                    report = run_live(
                        ["--task", "countdown", "--G", str(G), "--dataset", str(data),
                         "--max-steps", "2", "--completion-tokens", "16", "--no-vllm",
                         "--report-to", "none", "--output-dir", str(out)],
                        planted=True, model_id=self.MODEL)
                    self.assertTrue(report.passed, json.dumps(report.to_dict(), indent=1))


if __name__ == "__main__":
    unittest.main()
