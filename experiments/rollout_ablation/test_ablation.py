"""
Tests for the ablation launcher and resume wrapper.

    python -m unittest experiments.rollout_ablation.test_ablation
    RUN_LIVE_ABLATION_TEST=1 python -m unittest experiments.rollout_ablation.test_ablation
        # + tiny-model stop/resume through the real grpo.main() on CPU
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from experiments.rollout_ablation import config as C
from experiments.rollout_ablation.launch import (
    build_manifest,
    dataset_checks,
    manifest_mismatch,
    prompts_needed,
    run_status,
)
from experiments.rollout_ablation.train import (
    latest_checkpoint,
    prepare_resume,
    rewind_compute,
    rewind_rollouts,
    should_stop,
)

N = 64


def _write_rollouts(rd: Path, steps: int, extra: int = 0) -> None:
    with open(rd / "rollouts.jsonl", "w") as f:
        for s in range(steps):
            for i in range(N):
                f.write(json.dumps({"step": s, "prompt": f"{s}-{i // 4}", "reward": 0.0}) + "\n")
        for i in range(extra):           # partial step written before a crash
            f.write(json.dumps({"step": steps, "prompt": "x", "reward": 0.0}) + "\n")


def _ckpt(rd: Path, step: int, tokens: int = 0, complete: bool = True) -> Path:
    p = rd / f"checkpoint-{step}"
    p.mkdir()
    if complete:
        (p / "trainer_state.json").write_text(json.dumps(
            {"global_step": step, "num_input_tokens_seen": tokens, "log_history": []}))
        (p / "adapter_model.safetensors").write_bytes(b"x")
    return p


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.rd = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_latest_checkpoint_skips_incomplete(self):
        self.assertIsNone(latest_checkpoint(self.rd))
        _ckpt(self.rd, 50)
        _ckpt(self.rd, 100)
        _ckpt(self.rd, 150, complete=False)       # killed mid-save
        self.assertEqual(latest_checkpoint(self.rd).name, "checkpoint-100")

    def test_rewind_rollouts_cuts_to_checkpoint(self):
        _write_rollouts(self.rd, steps=5, extra=10)
        info = rewind_rollouts(self.rd, 3, N, "t")
        self.assertEqual(info, {"rollouts_kept": 3 * N, "rollouts_dropped": 2 * N + 10})
        rows = [json.loads(l) for l in (self.rd / "rollouts.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 3 * N)
        self.assertEqual(rows[-1]["step"], 2)
        self.assertTrue((self.rd / "rollouts.jsonl.t.bak").is_file())

    def test_rewind_rollouts_refuses_misaligned(self):
        _write_rollouts(self.rd, steps=3)
        lines = (self.rd / "rollouts.jsonl").read_text().splitlines(keepends=True)
        (self.rd / "rollouts.jsonl").write_text("".join(lines[5:]))     # shifted by 5 rows
        with self.assertRaises(RuntimeError):
            rewind_rollouts(self.rd, 2, N, "t")

    def test_rewind_rollouts_too_short(self):
        _write_rollouts(self.rd, steps=1)
        with self.assertRaises(RuntimeError):
            rewind_rollouts(self.rd, 2, N, "t")

    def test_rewind_compute_resets_tokens_keeps_gpu_hours(self):
        (self.rd / "compute.json").write_text(json.dumps({
            "config": {"weight_per_token": 8.0}, "tokens": 999.0, "weighted_tokens": 7992.0,
            "steps": 7, "gpu_hours": 3.5, "budget_this_run": None}))
        ck = _ckpt(self.rd, 5, tokens=500)
        rewind_compute(self.rd, ck)
        d = json.loads((self.rd / "compute.json").read_text())
        self.assertEqual((d["tokens"], d["weighted_tokens"], d["steps"]), (500.0, 4000.0, 5))
        self.assertEqual(d["gpu_hours"], 3.5)
        self.assertEqual(d["rewinds"][0]["before"]["steps"], 7)

    def test_prepare_resume_moves_completion_tokens(self):
        _write_rollouts(self.rd, steps=2)
        (self.rd / "completion_tokens.json").write_text("{}")
        info = prepare_resume(self.rd, _ckpt(self.rd, 2), N)
        self.assertEqual(info["resume_step"], 2)
        self.assertFalse((self.rd / "completion_tokens.json").exists())
        self.assertEqual(len(list(self.rd.glob("completion_tokens.pre-resume-2-*.json"))), 1)

    def test_should_stop(self):
        self.assertFalse(should_stop(0, None, 100, 900))
        self.assertFalse(should_stop(0, 10_000, 400, 900))           # 1400 s needed, 10000 left
        self.assertTrue(should_stop(0, 1_400, 400, 900))             # 400*1.25 + 900 = 1400
        self.assertTrue(should_stop(0, 800, None, 900))              # inside the margin


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _data(self, n_train: int, val: bool = True):
        (self.d / "train.jsonl").write_text("".join(f'{{"i": {i}}}\n' for i in range(n_train)))
        if val:
            (self.d / "val.jsonl").write_text('{"i": 0}\n')

    def test_prompt_budget(self):
        self.assertEqual(prompts_needed(4, 400), 6400)
        self.assertEqual(prompts_needed(16, 400), 1600)

    def test_dataset_big_enough(self):
        self._data(6400)
        info, err, warn = dataset_checks(str(self.d), [4, 8, 16], 400,
                                         allow_repeats=False, need_eval=True)
        self.assertEqual(err, [])
        self.assertEqual(info["n_rows"], 6400)
        self.assertEqual(len(info["dataset_sha256"]), 64)

    def test_dataset_too_small_errors_or_warns(self):
        self._data(1750)                       # Reasoning Gym pilot train size
        _, err, _ = dataset_checks(str(self.d), [4, 8, 16], 400,
                                   allow_repeats=False, need_eval=True)
        self.assertEqual(len(err), 2)          # G=4 (6400) and G=8 (3200); G=16 needs 1600
        _, err, warn = dataset_checks(str(self.d), [4, 8, 16], 400,
                                      allow_repeats=True, need_eval=True)
        self.assertEqual((len(err), len(warn)), (0, 2))

    def test_missing_val_split(self):
        self._data(7000, val=False)
        _, err, _ = dataset_checks(str(self.d), [4], 400, allow_repeats=False, need_eval=True)
        self.assertEqual(len(err), 1)
        _, err, warn = dataset_checks(str(self.d), [4], 400, allow_repeats=False, need_eval=False)
        self.assertEqual((len(err), len(warn)), (0, 1))

    def test_manifest_mismatch(self):
        self._data(7000)
        info, _, _ = dataset_checks(str(self.d), [4], 400, allow_repeats=False, need_eval=True)
        a = build_manifest("math", 4, 42, info)
        b = dict(a, created="later")
        self.assertEqual(manifest_mismatch(a, b), [])
        c = dict(a, dataset_sha256="0" * 64, hparams={**a["hparams"], "lr": 1e-5})
        self.assertEqual(len(manifest_mismatch(a, c)), 2)
        self.assertEqual(a["prompts_per_step"], 16)
        self.assertEqual(a["hparams"], C.HPARAMS)

    def test_epochs_rule_steps_and_no_repeat_error(self):
        from experiments.rollout_ablation.launch import effective_hparams, job_env
        self._data(1750)                       # team dataset size
        hp = effective_hparams()               # default: 2 epochs
        info, err, _ = dataset_checks(str(self.d), [4, 8, 16], hp,
                                      allow_repeats=False, need_eval=True)
        self.assertEqual(err, [])
        # matches the real runs: G=4 -> 218 steps, G=8 -> 436
        self.assertEqual((info["steps_G4"], info["steps_G8"], info["steps_G16"]), (218, 436, 874))
        env = job_env("math", 4, 42, self.d, str(self.d), type("A", (), {"time": "1:00", "no_eval": False}), hp)
        self.assertEqual(env["EPOCHS"], "2.0")
        self.assertNotIn("MAX_STEPS", env)
        hp2 = effective_hparams(max_steps=400, completion_tokens=2048)
        env = job_env("math", 4, 42, self.d, str(self.d), type("A", (), {"time": "1:00", "no_eval": False}), hp2)
        self.assertEqual((env["MAX_STEPS"], env["COMPLETION_TOKENS"]), ("400", "2048"))
        self.assertNotIn("EPOCHS", env)
        self.assertEqual(C.run_dir("math", 4, 42, 2048).name, "G4-L2048-s42")
        self.assertEqual(C.run_dir("math", 4, 42, 4096).name, "G4-s42")

    def test_run_status(self):
        rd = self.d / "G8-s42"
        rd.mkdir()
        _ckpt(rd, 50)
        _ckpt(rd, 100)
        (rd / "attempts.jsonl").write_text(json.dumps(
            {"init_adapter_sha256": "abc1234", "git_sha": "deadbee", "git_dirty": False}) + "\n")
        (rd / "jobs.jsonl").write_text('{"attempt": 1, "slurm_job_id": "123"}\n')
        (rd / "rollout_validation_offline.json").write_text('{"passed": true}')
        e = rd / "checkpoint-50" / "eval" / "math"
        e.mkdir(parents=True)
        (e / "val_n500.json").write_text("{}")
        st = run_status(rd)
        self.assertEqual((st["global_step"], st["attempts"], st["last_job"]), (100, 1, "123"))
        self.assertEqual((st["validation"], st["n_evals"], st["done"]), ("PASS", 1, False))
        self.assertEqual(st["init_sha"], {"abc1234"})


@unittest.skipUnless(os.environ.get("RUN_LIVE_ABLATION_TEST") == "1",
                     "set RUN_LIVE_ABLATION_TEST=1")
class TinyModelResumeTest(unittest.TestCase):
    """Stop after step 1 via the deadline, resume to step 3, compare with an uninterrupted run."""

    MODEL = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"

    def test_stop_resume_matches_uninterrupted(self):
        import time

        import reasoning_gym

        from experiments.rollout_ablation.make_init_adapter import make_init_adapter
        from experiments.rollout_ablation.train import run
        from grpo.rollout_validation import validate_run_dir

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "data").mkdir()
            with open(tmp / "data" / "train.jsonl", "w") as f:
                for x in reasoning_gym.create_dataset("countdown", size=200, seed=1):
                    f.write(json.dumps({"question": x["question"], "answer": x["answer"],
                                        "metadata": json.dumps(x["metadata"])}) + "\n")
            make_init_adapter(tmp / "init", 0, self.MODEL)

            def argv(out):
                return ["--task", "countdown", "--G", "8", "--dataset", str(tmp / "data"),
                        "--seed", "42", "--max-steps", "3", "--completion-tokens", "16",
                        "--no-vllm", "--report-to", "none", "--init-adapter", str(tmp / "init"),
                        "--output-dir", str(out)]

            rc = run(argv(tmp / "a"), deadline=time.time(), margin=0, model_id=self.MODEL)
            self.assertEqual(rc, C.EXIT_NEEDS_RESUME)
            self.assertEqual(json.loads((tmp / "a" / "status.json").read_text())["global_step"], 1)
            self.assertEqual(run(argv(tmp / "a"), deadline=None, margin=0, model_id=self.MODEL), 0)
            self.assertEqual(run(argv(tmp / "b"), deadline=None, margin=0, model_id=self.MODEL), 0)

            report = validate_run_dir(tmp / "a", 8, max_completion_length=16)
            self.assertTrue(report.passed, json.dumps(report.to_dict(), indent=1))

            def prompt_order(rd):
                rows = [json.loads(l) for l in open(rd / "rollouts.jsonl")]
                return [r["prompt"] for r in rows[::8]]
            self.assertEqual(prompt_order(tmp / "a"), prompt_order(tmp / "b"))
            comp = json.loads((tmp / "a" / "compute.json").read_text())
            st = json.loads((tmp / "a" / "checkpoint-3" / "trainer_state.json").read_text())
            self.assertEqual(comp["tokens"], st["num_input_tokens_seen"])


if __name__ == "__main__":
    unittest.main()
