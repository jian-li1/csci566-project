from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_test
from src.evaluate import checkpoint_identity, resolve_config, run_one, summary
from src.io_utils import file_hash, read_journal, read_json
from src.math_grader import final_box
from src.prompts import extract_code, final_text, missing_signatures, render_prompt
from src.shortest_path_grader import grade, grid_problem, shortest_distance

ROOT = Path(__file__).resolve().parents[1]


class TokenizerStub:
    chat_template = "stub"
    bos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return list(text.encode())

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
        return "".join(f"{m['role']}: {m['content']}\n" for m in messages) + "assistant:\n"


class EvalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = resolve_config(ROOT, read_json(ROOT / "configs/eval_all.example.json"))
        cls.rows = {r["task"]: load_test(ROOT, r) for r in cls.config["runs"]}

    def test_all_frozen_test_counts_and_hashes(self):
        self.assertEqual({k: len(v) for k, v in self.rows.items()}, {
            "kodcode": 250, "opencode": 250, "competition_math": 250, "shortest_path": 250})

    def test_all_example_configs_are_valid(self):
        for path in (ROOT / "configs").glob("*.json"):
            with self.subTest(path=path.name):
                resolve_config(ROOT, read_json(path))

    def test_hash_mismatch_is_rejected(self):
        run = copy.deepcopy(self.config["runs"][0])
        run["data"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            load_test(ROOT, run)

    def test_non_test_rows_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.jsonl"
            path.write_text(json.dumps({"split": "train"}) + "\n")
            run = copy.deepcopy(self.config["runs"][0])
            run["data"] = {"path": str(path), "rows": 1, "sha256": file_hash(path)}
            with self.assertRaisesRegex(ValueError, "test rows only"):
                load_test(ROOT, run)

    def test_all_1000_prompts_exclude_reference_and_test_fields(self):
        tokenizer = TokenizerStub()
        for run in self.config["runs"]:
            for original in self.rows[run["task"]]:
                row = copy.deepcopy(original)
                row.update(solution="FORBIDDEN_GOLD_CONTENT", output="FORBIDDEN_GOLD_CONTENT", answer="FORBIDDEN_GOLD_CONTENT",
                           test="FORBIDDEN_TEST_CONTENT", unit_tests=["FORBIDDEN_TEST_CONTENT"], assistant="FORBIDDEN_GOLD_CONTENT")
                text, ids = render_prompt(tokenizer, row, run)
                self.assertNotIn("FORBIDDEN_GOLD_CONTENT", text)
                self.assertNotIn("FORBIDDEN_TEST_CONTENT", text)
                self.assertTrue(ids)

    def test_signature_metadata_normalizes_and_renders(self):
        count = sum(bool(missing_signatures(r)) for r in self.rows["kodcode"])
        self.assertEqual(count, 124)

    def test_base_prompt_matches_original_kodcode_serialization(self):
        config = resolve_config(ROOT, read_json(ROOT / "configs/qwen_base.example.json"))
        run, row = config["runs"][0], self.rows["kodcode"][0]
        text, ids = render_prompt(TokenizerStub(), row, run)
        self.assertEqual(text, "### Instruction:\n" + row["question"].rstrip() +
                         "\n\nReturn a complete Python solution in a single fenced code block.\n\n### Response:\n")
        self.assertEqual(ids[0], 1)

    def test_thinking_must_finish_before_final_answer(self):
        self.assertEqual(final_text("still reasoning", "assistant:<think>\n"), ("", True))
        self.assertEqual(final_text("reasoning</think>\n42", "assistant:<think>\n"), ("42", False))
        self.assertEqual(final_text("42", "<think>\n\n</think>\n\n"), ("42", False))

    def test_code_fence_extraction(self):
        self.assertEqual(extract_code("Explanation\n```python\ndef add(a,b): return a+b\n```"), "def add(a,b): return a+b")
        self.assertEqual(extract_code("```python\ndef add(a,b): return a+b"), "def add(a,b): return a+b")

    def test_shortest_path_accepts_equal_length_alternative(self):
        row = {"matrix": [["*", "O"], ["O", "#"]]}
        start, end = grid_problem(row["matrix"])
        row.update(start=start, end=end, shortest_distance=shortest_distance(row["matrix"], start, end))
        self.assertEqual(grade("<answer>right down</answer>", row)[1], 1.0)
        self.assertEqual(grade("<answer>down right</answer>", row)[1], 1.0)
        self.assertEqual(grade("<answer>right left right down</answer>", row)[1], 0.5)
        self.assertEqual(grade("<answer>infeasible</answer>", row)[1], 0.0)
        self.assertEqual(grade("2", row)[1], 0.0)

    def test_unreachable_grids(self):
        row = {"matrix": [["*", "X", "#"]], "start": (0, 0), "end": (0, 2), "shortest_distance": None}
        self.assertEqual(grade("<answer>infeasible</answer>", row)[1], 1.0)
        self.assertEqual(grade("<answer>right right</answer>", row)[1], 0.0)

    def test_math_final_box_preserves_nested_braces_and_radix(self):
        self.assertEqual(final_box(r"Work \boxed{\frac{1}{2}}"), r"\frac{1}{2}")
        self.assertEqual(final_box(r"\boxed{101}_2"), "101_2")
        self.assertEqual(final_box(r"<think>\boxed{9}</think> Final \boxed{3}"), "3")
        self.assertIsNone(final_box("No boxed final answer"))

    def test_torn_journal_tail_is_repaired(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            path.write_bytes(b'{"question_id":"a"}\n{"question_')
            self.assertEqual(read_journal(path), [{"question_id": "a"}])
            self.assertEqual(path.read_bytes(), b'{"question_id":"a"}\n')

    def test_corrupt_journal_interior_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            path.write_bytes(b'{"question_id":"a"}\nBROKEN\n{"question_id":"b"}\n')
            with self.assertRaises(ValueError):
                read_journal(path)

    def test_incomplete_run_has_no_final_accuracy(self):
        result = summary(self.config["runs"][0], [1, 2], [], [{"correct": True, "score": 1.0}], False)
        self.assertIsNone(result["pass_at_1"])
        self.assertFalse(result["complete"])
        self.assertEqual(result["accuracy_on_completed"], 1.0)

    def test_missing_checkpoint_stops_before_model_loading(self):
        with self.assertRaises(FileNotFoundError):
            checkpoint_identity(ROOT, self.config["runs"][0])

    @unittest.skipIf(sys.platform == "win32", "The grading worker runs inside a Linux container")
    def test_known_authored_code_through_worker_interface(self):
        # Only this fixed, authored addition function is run outside a container.
        # Actual model output always goes through Docker/Apptainer in production.
        worker = ROOT / "src/grade_worker.py"
        for task, tests in [
            ("kodcode", "from solution import add\ndef test_add():\n    assert add(1,2)==3\n"),
            ("opencode", ["assert add(1,2)==3"]),
        ]:
            with tempfile.TemporaryDirectory() as directory:
                proc = subprocess.run([sys.executable, str(worker)], cwd=directory, text=True, capture_output=True,
                                      input=json.dumps({"task": task, "code": "def add(a,b): return a+b", "tests": tests}), timeout=20)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertTrue(json.loads(proc.stdout)["passed"])

    def test_interrupted_evaluation_resumes_and_changed_settings_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            weights = Path(directory) / "adapter"
            weights.mkdir()
            (weights / "adapter_config.json").write_text("{}")
            (weights / "adapter_model.safetensors").write_bytes(b"synthetic fixture, never loaded")
            run = copy.deepcopy(self.config["runs"][3])
            run["checkpoint"]["path"] = str(weights)
            rows = self.rows["shortest_path"][:3]

            class FakeRunner:
                calls = 0
                failed = False
                metadata = {"test": "synthetic runner"}

                def __init__(self, *args):
                    pass

                def generate(self, row):
                    FakeRunner.calls += 1
                    if FakeRunner.calls == 3 and not FakeRunner.failed:
                        FakeRunner.failed = True
                        raise RuntimeError("synthetic interruption")
                    return {"question_id": row["question_id"], "final_text": "<answer>" + row["answer"] + "</answer>",
                            "raw_response": "synthetic fixture", "generated_tokens": 1, "generation_seconds": 0.0}

                def close(self):
                    pass

            transformers_stub = types.ModuleType("transformers")
            transformers_stub.set_seed = lambda seed: None
            output = Path(directory) / "outputs"
            with patch.dict(sys.modules, {"transformers": transformers_stub}), patch("src.model_loader.ModelRunner", FakeRunner):
                with self.assertRaisesRegex(RuntimeError, "synthetic interruption"):
                    run_one(ROOT, output, run, self.config["generation"], rows, 3, "all")
                result = run_one(ROOT, output, run, self.config["generation"], rows, 3, "all")
                self.assertTrue(result["complete"])
                self.assertEqual(result["correct"], 3)
                self.assertEqual(FakeRunner.calls, 4)
                run_one(ROOT, output, run, self.config["generation"], rows, 3, "all")
                self.assertEqual(FakeRunner.calls, 4)
                changed = {**self.config["generation"], "max_new_tokens": 2048}
                with self.assertRaisesRegex(ValueError, "changed"):
                    run_one(ROOT, output, run, changed, rows, 3, "all")


if __name__ == "__main__":
    unittest.main()
