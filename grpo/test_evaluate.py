"""
Tests for grpo/evaluate.py.

    python -m unittest grpo.test_evaluate                  # metric math, no model
    RUN_LIVE_EVAL_TEST=1 python -m unittest grpo.test_evaluate
        # + end-to-end on CPU with a tiny model (needs torch/trl/peft/reasoning-gym)
"""

from __future__ import annotations

import itertools
import json
import os
import random
import tempfile
import unittest
from math import comb
from pathlib import Path

from grpo.evaluate import (
    bucket,
    parse_split_specs,
    pass_at_k,
    per_prompt_record,
    prob_group_all_equal,
    sample_indices,
    summarize,
)


def _s(reward, accepted, fail=None, n_tokens=10, hit=False):
    return {"reward": reward, "accepted": accepted, "fail": fail,
            "n_tokens": n_tokens, "hit_length": hit}


class MetricMathTests(unittest.TestCase):
    def test_pass_at_k_edges(self):
        self.assertEqual(pass_at_k(16, 0, 1), 0.0)
        self.assertEqual(pass_at_k(16, 16, 16), 1.0)
        self.assertAlmostEqual(pass_at_k(16, 4, 1), 4 / 16)
        self.assertEqual(pass_at_k(16, 1, 16), 1.0)

    def test_pass_at_k_matches_enumeration(self):
        n, c = 8, 3
        for k in (1, 2, 4):
            subsets = list(itertools.combinations(range(n), k))
            brute = sum(any(i < c for i in s) for s in subsets) / len(subsets)
            self.assertAlmostEqual(pass_at_k(n, c, k), brute)

    def test_group_all_equal_matches_enumeration(self):
        vals = [1.0, 1.0, 0.0, 0.5, 0.0, 0.0]
        for G in (2, 3, 4):
            subsets = list(itertools.combinations(vals, G))
            brute = sum(len(set(s)) == 1 for s in subsets) / len(subsets)
            self.assertAlmostEqual(prob_group_all_equal(vals, G), brute)

    def test_buckets(self):
        self.assertEqual(bucket(0.0), "hard")
        self.assertEqual(bucket(0.1875), "hard")
        self.assertEqual(bucket(0.2), "intermediate")
        self.assertEqual(bucket(0.8), "intermediate")
        self.assertEqual(bucket(0.8125), "easy")
        self.assertIsNone(bucket(None))

    def test_split_specs_and_sampling(self):
        self.assertEqual(parse_split_specs("val, test,train:500"),
                         [("val", None), ("test", None), ("train", 500)])
        a = sample_indices(1000, 50, seed=3)
        self.assertEqual(a, sample_indices(1000, 50, seed=3))       # deterministic
        self.assertEqual(a, sorted(a))
        self.assertEqual(len(set(a)), 50)
        self.assertEqual(sample_indices(10, None, 0), list(range(10)))
        self.assertEqual(sample_indices(10, 99, 0), list(range(10)))


class SummaryTests(unittest.TestCase):
    def test_per_prompt_and_summary(self):
        k = 16
        recs = []
        for i, c in enumerate([0, 2, 8, 16]):              # solve counts out of 16
            samples = [_s(1.0, True) if j < c else _s(0.0, False, "wrong") for j in range(k)]
            greedy = _s(1.0, True) if c >= 8 else _s(0.0, False, "missing_answer")
            recs.append(per_prompt_record(i, f"prompt {i}", greedy, samples))
        self.assertEqual([r["bucket"] for r in recs], ["hard", "hard", "intermediate", "easy"])

        m = summarize(recs)
        self.assertEqual(m["greedy"]["accuracy"], 0.5)
        self.assertEqual(m["greedy"]["fail_rates"], {"missing_answer": 0.5})
        s = m["sampled"]
        self.assertAlmostEqual(s["pass_at"]["1"], (0 + 2 + 8 + 16) / 64)
        self.assertAlmostEqual(s["pass_at"]["16"], 0.75)
        self.assertEqual(s["bucket_counts"], {"hard": 2, "intermediate": 1, "easy": 1})
        g4 = s["group_predictions"]["4"]
        want_all = (0 + 0 + comb(8, 4) / comb(16, 4) + 1) / 4
        want_none = (1 + comb(14, 4) / comb(16, 4) + comb(8, 4) / comb(16, 4) + 0) / 4
        self.assertAlmostEqual(g4["frac_all_accepted"], want_all)
        self.assertAlmostEqual(g4["frac_none_accepted"], want_none)
        # binary rewards: zero-std groups = all-accepted + none-accepted
        self.assertAlmostEqual(g4["frac_zero_std"], want_all + want_none)
        self.assertNotIn("32", s["group_predictions"])

    def test_group_prediction_tracks_simulation(self):
        """Predicted all-correct fraction ~ simulated GRPO groups drawn from the samples."""
        rng = random.Random(0)
        recs = []
        for i in range(40):
            c = rng.randint(0, 16)
            recs.append(per_prompt_record(i, str(i), None,
                                          [_s(float(j < c), j < c) for j in range(16)]))
        pred = summarize(recs)["sampled"]["group_predictions"]["8"]["frac_all_accepted"]
        sims = []
        for _ in range(4000):
            r = rng.choice(recs)
            sims.append(all(rng.sample(r["accepted"], 8)))
        self.assertAlmostEqual(pred, sum(sims) / len(sims), delta=0.03)

    def test_partial_reward_zero_std_differs_from_acceptance(self):
        # OCI-style partial credit: none accepted, but rewards vary -> not zero std
        samples = [_s(0.5 if j % 2 else 0.25, False, "partial") for j in range(16)]
        m = summarize([per_prompt_record(0, "p", None, samples)])
        g = m["sampled"]["group_predictions"]["4"]
        self.assertEqual(g["frac_none_accepted"], 1.0)
        self.assertLess(g["frac_zero_std"], 0.3)

    def test_greedy_only(self):
        m = summarize([per_prompt_record(0, "p", _s(1.0, True), [])])
        self.assertIn("greedy", m)
        self.assertNotIn("sampled", m)


@unittest.skipUnless(os.environ.get("RUN_LIVE_EVAL_TEST") == "1", "set RUN_LIVE_EVAL_TEST=1")
class TinyModelEvalTest(unittest.TestCase):
    MODEL = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"

    def test_end_to_end_hf_backend(self):
        import reasoning_gym

        from grpo.evaluate import main

        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data"
            data.mkdir()
            for split, seed in (("val", 2), ("train", 3)):
                ds = reasoning_gym.create_dataset("countdown", size=30, seed=seed)
                with open(data / f"{split}.jsonl", "w") as f:
                    for x in ds:
                        f.write(json.dumps({"question": x["question"], "answer": x["answer"],
                                            "metadata": json.dumps(x["metadata"])}) + "\n")
            out = Path(tmp) / "out"
            rc = main(["--task", "countdown", "--dataset", str(data), "--splits", "val:6,train:4",
                       "--k", "4", "--completion-tokens", "12", "--backend", "hf",
                       "--model-id", self.MODEL, "--output-dir", str(out),
                       "--save-completions", "all"])
            self.assertEqual(rc, 0)
            summ = json.loads((out / "val_n6.json").read_text())
            self.assertEqual(summ["n_eval"], 6)
            self.assertEqual(summ["metrics"]["sampled"]["k"], 4)
            self.assertIsNotNone(summ["file_sha256"])
            rows = [json.loads(l) for l in open(out / "val_n6_per_prompt.jsonl")]
            self.assertEqual(len(rows), 6)
            self.assertTrue(all(len(r["rewards"]) == 4 for r in rows))
            self.assertTrue(all(r["greedy_n_tokens"] <= 12 for r in rows))
            comps = [json.loads(l) for l in open(out / "val_n6_completions.jsonl")]
            self.assertEqual(len(comps), 6 * 5)

            # Prompts are byte-identical to training prompts.
            import grpo.grpo as g
            ds = g.load_train_dataset(str(data), "val", task="countdown")
            from grpo.evaluate import prompt_hash, sample_indices
            want = [prompt_hash(ds[i]["prompt"]) for i in sample_indices(len(ds), 6, 0)]
            self.assertEqual([r["prompt_hash"] for r in rows], want)

            # Refuses to overwrite.
            with self.assertRaises(SystemExit):
                main(["--task", "countdown", "--dataset", str(data), "--splits", "val:6",
                      "--backend", "hf", "--model-id", self.MODEL, "--output-dir", str(out)])


if __name__ == "__main__":
    unittest.main()
