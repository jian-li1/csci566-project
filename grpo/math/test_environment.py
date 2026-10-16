"""Focused tests for the Competition Math environment and binary reward."""

import unittest

from competition_math import CompetitionMathEnv, check_answer, extract_boxed_answer


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
