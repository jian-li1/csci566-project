"""Focused tests for the MBPP environment and fractional test reward."""

import unittest

from mbpp import MBPPEnv, check_code, extract_python_code


class MBPPEnvTests(unittest.TestCase):
    def setUp(self):
        self.row = {
            "task_id": 1,
            "text": "Write a function that returns the square of an integer.",
            "code": "def square(value):\n    return value * value",
            "test_list": [
                "assert square(2) == 4",
                "assert square(-3) == 9",
                "assert square(0) == 0",
            ],
            "test_setup_code": "",
            "challenge_test_list": [],
        }

    def test_extracts_last_python_fence(self):
        answer = "Explanation\n```python\ndef square(x):\n    return x * x\n```"
        self.assertEqual(
            extract_python_code(answer),
            "def square(x):\n    return x * x",
        )

    def test_correct_code_gets_one(self):
        env = MBPPEnv()
        observation = env.reset(self.row)
        self.assertNotIn("code", observation)
        self.assertNotIn("test_list", observation)

        result = env.step(
            "<think>Multiply the input by itself.</think>"
            "<answer>def square(x):\n    return x * x</answer>"
        )
        self.assertEqual(result["reward"], 1.0)
        self.assertTrue(result["acceptor"])
        self.assertEqual(result["check"]["passed"], 3)
        self.assertTrue(result["terminated"])

    def test_partial_credit_is_tests_passed_over_total(self):
        result = check_code(
            "def square(x):\n    return 0 if x == -3 else x * x",
            self.row["test_list"],
        )
        self.assertEqual(result["passed"], 2)
        self.assertEqual(result["total"], 3)
        self.assertAlmostEqual(result["reward"], 2 / 3)

    def test_wrong_code_gets_zero(self):
        env = MBPPEnv()
        env.reset(self.row)
        result = env.step("<answer>def square(x):\n    return 7</answer>")
        self.assertEqual(result["reward"], 0.0)
        self.assertEqual(result["severity"], "end_episode")

    def test_markdown_fenced_answer_is_accepted(self):
        env = MBPPEnv()
        env.reset(self.row)
        result = env.step(
            "<answer>```python\ndef square(x):\n    return x * x\n```</answer>"
        )
        self.assertEqual(result["reward"], 1.0)

    def test_setup_code_runs_before_candidate(self):
        result = check_code(
            "def circle_area(radius):\n    return pi * radius ** 2",
            ["assert circle_area(2) == 4 * pi"],
            test_setup_code="from math import pi",
        )
        self.assertEqual(result["reward"], 1.0)

    def test_syntax_error_fails_every_test(self):
        result = check_code("def square(:", self.row["test_list"])
        self.assertEqual(result["passed"], 0)
        self.assertEqual(result["reward"], 0.0)
        self.assertTrue(all("SyntaxError" in test["error"] for test in result["tests"]))

    def test_timeout_fails_the_test(self):
        result = check_code(
            "def square(x):\n    while True:\n        pass",
            ["assert square(2) == 4"],
            timeout_seconds=0.2,
        )
        self.assertEqual(result["reward"], 0.0)
        self.assertIn("timeout", result["tests"][0]["error"])

    def test_missing_answer_tags_gets_zero(self):
        env = MBPPEnv()
        env.reset(self.row)
        result = env.step("def square(x): return x * x")
        self.assertEqual(result["reward"], 0.0)
        self.assertEqual(result["check"]["error"], "missing <answer>...</answer>")

    def test_empty_response_is_unscored(self):
        env = MBPPEnv()
        env.reset(self.row)
        result = env.step("")
        self.assertIsNone(result["reward"])
        self.assertEqual(result["severity"], "end_rollout")

    def test_step_before_reset_is_unscored(self):
        result = MBPPEnv().step("<answer>def square(x): return x * x</answer>")
        self.assertIsNone(result["reward"])
        self.assertEqual(result["check"]["error"], "step before reset")


if __name__ == "__main__":
    unittest.main()
