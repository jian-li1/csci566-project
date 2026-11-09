"""
Tests for analysis/rollout_ablation.py on a synthetic ablation with known answers.

    python -m unittest analysis.test_rollout_ablation

make_synthetic_ablation() writes run dirs that look exactly like the real ones
(rollouts.jsonl, checkpoint-*/trainer_state.json + eval/, compute.json, manifest.json)
plus evaluate.py-style baseline files, from per-prompt solve rates we control.
"""

from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

from analysis.rollout_ablation import (
    analyze,
    group_metrics,
    interp,
    rolling,
    trapz_mean,
)
from grpo.evaluate import per_prompt_record, prompt_hash, summarize

N = 64


def _p_at(p0: float, step: int, steps: int, G: int) -> float:
    """Toy learning curve: flat for the first 10 steps (policy == base model, so the
    base-model group predictions apply), then rises toward 1; smaller G learns faster."""
    rate = {4: 0.9, 8: 0.7, 16: 0.5}.get(G, 0.7)
    return p0 + (1 - p0) * rate * max(0, step - 10) / (steps - 10)


def make_synthetic_ablation(root: Path, *, steps: int = 60, seeds=(42, 43), Gs=(4, 8, 16),
                            n_train: int = 2000, n_val: int = 200, k: int = 16) -> dict:
    rng = random.Random(0)
    train_p = [rng.betavariate(0.6, 0.9) for _ in range(n_train)]
    val_p = [rng.betavariate(0.6, 0.9) for _ in range(n_val)]
    train_prompts = [f"train prompt {i}" for i in range(n_train)]
    val_prompts = [f"val prompt {i}" for i in range(n_val)]

    def sample_record(idx, prompt, p, greedy_p):
        samples = [{"reward": float(a), "accepted": a, "fail": None if a else "wrong",
                    "n_tokens": 100, "hit_length": False}
                   for a in (rng.random() < p for _ in range(k))]
        g = rng.random() < greedy_p
        greedy = {"reward": float(g), "accepted": g, "fail": None if g else "wrong",
                  "n_tokens": 100, "hit_length": False}
        return per_prompt_record(idx, prompt, greedy, samples)

    base_dir = root / "baselines" / "toy"
    base_dir.mkdir(parents=True)
    for stem, prompts, ps in (("train_n500", train_prompts[:500], train_p[:500]),
                              ("val_n500", val_prompts, val_p)):
        recs = [sample_record(i, pr, p, p) for i, (pr, p) in enumerate(zip(prompts, ps))]
        (base_dir / f"{stem}.json").write_text(json.dumps(
            {"split": stem.split("_")[0], "limit": 500, "metrics": summarize(recs)}))
        with open(base_dir / f"{stem}_per_prompt.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")

    for seed in seeds:
        order = list(range(n_train))
        random.Random(seed).shuffle(order)          # same order for every G (nested prefixes)
        for G in Gs:
            rd = root / "runs" / "toy" / f"G{G}-s{seed}"
            rd.mkdir(parents=True)
            (rd / "manifest.json").write_text(json.dumps({"task": "toy", "G": G, "seed": seed}))
            hist, num_tokens, cursor = [], 0, 0
            srng = random.Random(seed * 100 + G)
            with open(rd / "rollouts.jsonl", "w") as f:
                for s in range(1, steps + 1):
                    rewards = []
                    for _ in range(N // G):
                        i = order[cursor]
                        cursor += 1
                        p = _p_at(train_p[i], s - 1, steps, G)
                        for _ in range(G):
                            a = srng.random() < p
                            rewards.append(float(a))
                            f.write(json.dumps({"step": s - 1, "task": "toy",
                                                "prompt": train_prompts[i], "reward": float(a),
                                                "acceptor": a, "n_tokens": 300,
                                                "hit_length_limit": False}) + "\n")
                    gr = [rewards[j:j + G] for j in range(0, N, G)]
                    num_tokens += N * 400
                    hist.append({"step": s, "reward": sum(rewards) / N,
                                 "frac_reward_zero_std": sum(len(set(g)) == 1 for g in gr) / len(gr),
                                 "completions/mean_length": 300.0, "completions/clipped_ratio": 0.05,
                                 "step_time": 300.0 + 10 * G, "num_tokens": float(num_tokens)})
            (rd / "compute.json").write_text(json.dumps(
                {"config": {"weight_per_token": 8.0}, "gpu_hours": steps * (300 + 10 * G) / 3600}))
            for ck in range(20, steps + 1, 20):
                cdir = rd / f"checkpoint-{ck}"
                (cdir / "eval" / "toy").mkdir(parents=True)
                (cdir / "trainer_state.json").write_text(json.dumps(
                    {"global_step": ck, "log_history": hist[:ck]}))
                recs = []
                for i, (pr, p) in enumerate(zip(val_prompts, val_p)):
                    g = srng.random() < _p_at(p, ck, steps, G)
                    recs.append(per_prompt_record(i, pr, {"reward": float(g), "accepted": g,
                                                          "fail": None, "n_tokens": 1,
                                                          "hit_length": False}, []))
                (cdir / "eval" / "toy" / "val_n500.json").write_text(json.dumps(
                    {"split": "val", "limit": 500, "metrics": summarize(recs)}))
                with open(cdir / "eval" / "toy" / "val_n500_per_prompt.jsonl", "w") as f:
                    for r in recs:
                        f.write(json.dumps(r) + "\n")
    return {"base_dir": base_dir, "runs": root / "runs" / "toy"}


class HelperTests(unittest.TestCase):
    def test_group_metrics(self):
        # 4 groups of G=4 (16 rollouts): all-correct, all-wrong, mixed, partial-only
        block = []
        for p, rs in (("a", [1, 1, 1, 1]), ("b", [0, 0, 0, 0]), ("c", [1, 0, 0, 1]),
                      ("d", [0.5, 0.5, 0.5, 0.5])):
            block += [(p, float(r), r >= 0.99, 10, False) for r in rs]
        m = group_metrics(block, 4)
        self.assertEqual(m["n_groups"], 4)
        self.assertEqual(m["frac_zero_std"], 0.75)
        self.assertEqual(m["effective_rollout_frac"], 0.25)
        self.assertEqual(m["frac_all_accepted"], 0.25)
        self.assertEqual(m["frac_none_accepted"], 0.5)        # b and d (partial, not accepted)
        self.assertEqual(m["frac_all_reward_zero"], 0.25)     # only b
        self.assertEqual(m["frac_mixed_acceptance"], 0.25)

    def test_curve_helpers(self):
        self.assertEqual(interp([0, 10], [0.0, 1.0], 5), 0.5)
        self.assertIsNone(interp([0, 10], [0.0, 1.0], 11))
        self.assertAlmostEqual(trapz_mean([0, 10], [0.0, 1.0]), 0.5)
        self.assertEqual(rolling([1, None, 3], 3), [1, 2, 3])


class SyntheticAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.paths = make_synthetic_ablation(root)
        cls.out = root / "out"
        cls.res = analyze([cls.paths["runs"]], cls.out, baseline_dir=cls.paths["base_dir"],
                          figures=False, min_steps=1)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def arm(self, G):
        return next(a for a in self.res["arms"] if a["G"] == G)

    def test_arms_and_runs(self):
        self.assertEqual(len(self.res["runs"]), 6)
        self.assertEqual([a["G"] for a in self.res["arms"]], [4, 8, 16])
        self.assertTrue(all(a["n_runs"] == 2 for a in self.res["arms"]))

    def test_rollout_metrics_match_trl_logs(self):
        for r in self.res["runs"]:
            self.assertAlmostEqual(r["frac_zero_std_mean"], r["trl_frac_zero_std_mean"], places=9)

    def test_early_groups_match_base_prediction(self):
        # At the start the policy ~ base model, so observed all-/none-accepted fractions
        # over the first steps should be close to evaluate.py's prediction for that G.
        for G in (4, 8, 16):
            rows = [r for r in self.res["runs"] if r["G"] == G]
            obs = sum(r["frac_all_accepted_early"] for r in rows) / len(rows)
            pred = rows[0]["pred_frac_all_accepted"]
            self.assertAlmostEqual(obs, pred, delta=0.08, msg=f"G={G}")
            obs = sum(r["frac_none_accepted_early"] for r in rows) / len(rows)
            self.assertAlmostEqual(obs, rows[0]["pred_frac_none_accepted"], delta=0.08)

    def test_bigger_G_fewer_degenerate_groups(self):
        z = [self.arm(G)["trl_frac_zero_std_mean_mean"] for G in (4, 8, 16)]
        self.assertGreater(z[0], z[1])
        self.assertGreater(z[1], z[2])

    def test_learning_and_cost(self):
        for G in (4, 8, 16):
            a = self.arm(G)
            self.assertGreater(a["gain_mean"], 0.1)
            self.assertIsNotNone(a["accuracy_at_common_gpu_hours_mean"])
            self.assertIsNotNone(a["final_accuracy_hard_mean"])
        self.assertGreater(self.arm(4)["final_accuracy_mean"], self.arm(16)["final_accuracy_mean"])
        self.assertLess(self.arm(4)["train_gpu_hours_mean"], self.arm(16)["train_gpu_hours_mean"])
        r = self.res["runs"][0]
        self.assertAlmostEqual(r["weighted_tokens_total"], 60 * N * 400 * 8)

    def test_outputs_written(self):
        for name in ("report.md", "runs.csv", "arms.csv", "steps.csv", "evals.csv", "buckets.csv"):
            self.assertTrue((self.out / name).is_file(), name)
        report = (self.out / "report.md").read_text()
        self.assertIn("| final val accuracy |", report)
        self.assertIn("G=16 (n=2)", report)

    def test_figures_if_matplotlib(self):
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            self.skipTest("matplotlib not installed")
        out = self.out.parent / "figs"
        res = analyze([self.paths["runs"]], out, baseline_dir=self.paths["base_dir"], min_steps=1)
        self.assertEqual(len(res["figures"]), 6)
        for f in res["figures"]:
            self.assertTrue((out / f).is_file() and (out / f.replace(".png", ".csv")).is_file())


if __name__ == "__main__":
    unittest.main()
