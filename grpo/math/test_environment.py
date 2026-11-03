"""Focused tests for the Competition Math environment and binary reward."""

import unittest

from grpo.math.environment import (
    CompetitionMathEnv,
    check_answer,
    extract_boxed_answer,
)
from grpo.metrics import make_env_reward_fn


class CompetitionMathEnvTests(unittest.TestCase):
    def setUp(self):
        self.row = {
            "problem": "Compute one half plus one half.",
            "level": "Level 1",
            "type": "Prealgebra",
            "solution": r"The sum is $\frac12+\frac12=\boxed{1}$.",
        }

    def test_extracts_nested_last_box(self):
        solution = r"First \boxed{2}, finally \boxed{\frac{10}{3}}."
        self.assertEqual(extract_boxed_answer(solution), r"\frac{10}{3}")

    def test_numeric_equivalence(self):
        result = check_answer("0.5", r"\frac{1}{2}")
        self.assertTrue(result["correct"])

    def test_correct_answer_gets_one(self):
        env = CompetitionMathEnv()
        observation = env.reset(self.row)
        self.assertNotIn("solution", observation)
        self.assertNotIn("reference_answer", observation)

        result = env.step("<think>Add the fractions.</think><answer>1</answer>")
        self.assertEqual(result["reward"], 1.0)
        self.assertTrue(result["acceptor"])
        self.assertTrue(result["terminated"])

    def test_think_mention_of_answer_tag_does_not_break_extraction(self):
        """Regression: mention of <answer> inside <think> must not steal the final body."""
        env = CompetitionMathEnv()
        env.reset(self.row)
        completion = (
            "<think>\n"
            "Put the result in `<answer>` tags.\n"
            "Half plus half is 1.\n"
            "</think>\n\n"
            "<answer>1</answer>"
        )
        result = env.step(completion)
        self.assertEqual(result["answer"], "1")
        self.assertEqual(result["reward"], 1.0)
        self.assertTrue(result["acceptor"])
        self.assertIsNone(result["check"]["error"])

    def test_load_problem_accepts_metrics_entry_dict(self):
        env = CompetitionMathEnv()
        observation = env.load_problem(self.row)

        self.assertEqual(observation["problem"], self.row["problem"])
        self.assertEqual(observation["level"], self.row["level"])
        self.assertEqual(observation["type"], self.row["type"])
        self.assertNotIn("solution", observation)
        self.assertNotIn("reference_answer", observation)

    def test_load_problem_keeps_legacy_positional_form(self):
        env = CompetitionMathEnv()
        observation = env.load_problem(
            self.row["problem"],
            self.row["solution"],
            self.row["level"],
            self.row["type"],
        )

        self.assertEqual(observation["problem"], self.row["problem"])
        self.assertEqual(observation["level"], self.row["level"])
        self.assertEqual(observation["type"], self.row["type"])

    def test_metrics_reward_hook_loads_entry_dict(self):
        class RecordingMetrics:
            def __init__(self):
                self.calls = []

            def record(self, *args):
                self.calls.append(args)

        metrics = RecordingMetrics()
        reward_fn = make_env_reward_fn(
            {"math": CompetitionMathEnv()}, metrics  # type: ignore[arg-type]
        )

        rewards = reward_fn(
            prompts=["Solve the problem"],
            completions=["<answer>1</answer>"],
            task=["math"],
            entry=[self.row],
        )

        self.assertEqual(rewards, [1.0])
        self.assertEqual(len(metrics.calls), 1)

    def test_wrong_answer_gets_zero(self):
        env = CompetitionMathEnv()
        env.reset(self.row)
        result = env.step("<answer>2</answer>")
        self.assertEqual(result["reward"], 0.0)
        self.assertEqual(result["severity"], "end_episode")

    def test_missing_answer_tags_gets_zero(self):
        env = CompetitionMathEnv()
        env.reset(self.row)
        result = env.step("The answer is 1.")
        self.assertEqual(result["reward"], 0.0)
        self.assertEqual(result["check"]["error"], "missing <answer>...</answer>")

    def test_empty_response_is_unscored(self):
        env = CompetitionMathEnv()
        env.reset(self.row)
        result = env.step("")
        self.assertIsNone(result["reward"])
        self.assertEqual(result["severity"], "end_rollout")

    def test_empty_boxed_reference_is_rejected(self):
        row = dict(self.row, solution=r"The result is \boxed{}.")
        with self.assertRaisesRegex(ValueError, "non-empty balanced"):
            CompetitionMathEnv().reset(row)

    def test_step_before_reset_is_unscored(self):
        result = CompetitionMathEnv().step("<answer>1</answer>")
        self.assertIsNone(result["reward"])
        self.assertEqual(result["check"]["error"], "step before reset")


if __name__ == "__main__":
    unittest.main()
