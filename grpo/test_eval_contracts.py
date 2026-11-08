"""
Contract tests for held-out eval (no GPU / no full model generate).

Covers:
  - canonical prompt invariant (fingerprinted text == generation text)
  - env.step wrong vs unscored outcomes for Math + RG wrappers
  - is_unresolved_verifier_result / answer field token-count safety
  - coding task refused
"""

from __future__ import annotations

import unittest

from grpo.countdown import CountdownEnv
from grpo.eval import (
    canonical_prompt,
    is_unresolved_verifier_result,
    make_env_for_task,
    verify_response,
)
from grpo.knights_knaves import KnightsKnavesEnv
from grpo.math.environment import CompetitionMathEnv
from grpo.shortest_path import ShortestPathEnv


class CanonicalPromptTests(unittest.TestCase):
    def test_stored_prompt_must_match_rebuild(self):
        row = {
            "problem": "What is 1+1?",
            "solution": r"Clearly \boxed{2}.",
            "prompt": "WRONG PROMPT",
            "entry": {
                "problem": "What is 1+1?",
                "solution": r"Clearly \boxed{2}.",
            },
            "task": "math",
        }
        with self.assertRaisesRegex(ValueError, "Prompt mismatch"):
            canonical_prompt(row, "math")

    def test_missing_prompt_falls_back_to_rebuild(self):
        row = {
            "problem": "What is 1+1?",
            "solution": r"Clearly \boxed{2}.",
        }
        text = canonical_prompt(row, "math")
        self.assertIsInstance(text, str)
        self.assertTrue(len(text) > 0)
        row["prompt"] = text
        self.assertEqual(canonical_prompt(row, "math"), text)


class MathVerifierContractTests(unittest.TestCase):
    def setUp(self):
        self.entry = {
            "problem": "Compute one half plus one half.",
            "level": "Level 1",
            "type": "Prealgebra",
            "solution": r"The sum is $\frac12+\frac12=\boxed{1}$.",
        }

    def test_correct_is_scored_accept(self):
        out = verify_response("math", {"entry": self.entry}, "<answer>1</answer>")
        self.assertEqual(out["reward"], 1.0)
        self.assertTrue(out["acceptor"])
        self.assertFalse(is_unresolved_verifier_result(out))
        self.assertIsInstance(out["answer"], str)

    def test_wrong_is_scored_reject(self):
        out = verify_response("math", {"entry": self.entry}, "<answer>2</answer>")
        self.assertEqual(out["reward"], 0.0)
        self.assertFalse(out["acceptor"])
        self.assertEqual(out["severity"], "end_episode")
        self.assertFalse(is_unresolved_verifier_result(out))

    def test_empty_response_is_unresolved(self):
        out = verify_response("math", {"entry": self.entry}, "")
        self.assertIsNone(out["reward"])
        self.assertEqual(out["severity"], "end_rollout")
        self.assertTrue(is_unresolved_verifier_result(out))

    def test_step_before_reset_is_unresolved(self):
        out = CompetitionMathEnv().step("<answer>1</answer>")
        self.assertIsNone(out["reward"])
        self.assertTrue(is_unresolved_verifier_result(out))

    def test_answer_none_safe_for_token_count(self):
        out = verify_response("math", {"entry": self.entry}, "")
        answer = out.get("answer")
        if isinstance(answer, str) and answer:
            n = len(answer)
        else:
            n = 0
        self.assertEqual(n, 0)


class ReasoningGymVerifierContractTests(unittest.TestCase):
    """
    Format-path contracts shared by Countdown / KnightsKnaves / ShortestPath.

    These paths return before Reasoning Gym score_answer(), so they do not need
    a fully valid RG metadata blob — only load_problem(entry) binding.
    """

    ENTRY = {
        "question": "synthetic question for contract tests",
        "answer": "synthetic-gold",
        "metadata": {},
    }

    def _check_env(self, env_cls):
        env = env_cls()
        env.load_problem(dict(self.ENTRY))

        wrong_format = env.step("I forgot the tags.")
        self.assertEqual(wrong_format["reward"], 0.0)
        self.assertFalse(wrong_format["acceptor"])
        self.assertEqual(wrong_format["severity"], "end_episode")
        self.assertFalse(is_unresolved_verifier_result(wrong_format))

        env2 = env_cls()
        env2.load_problem(dict(self.ENTRY))
        empty = env2.step("")
        self.assertIsNone(empty["reward"])
        self.assertEqual(empty["severity"], "end_rollout")
        self.assertTrue(is_unresolved_verifier_result(empty))

        before = env_cls().step("<answer>x</answer>")
        self.assertIsNone(before["reward"])
        self.assertTrue(is_unresolved_verifier_result(before))

    def test_countdown_format_contracts(self):
        self._check_env(CountdownEnv)

    def test_knights_knaves_format_contracts(self):
        self._check_env(KnightsKnavesEnv)

    def test_shortest_path_format_contracts(self):
        self._check_env(ShortestPathEnv)


class UnresolvedHelperTests(unittest.TestCase):
    def test_synthetic_unresolved_shapes(self):
        self.assertTrue(
            is_unresolved_verifier_result(
                {"reward": None, "severity": "end_rollout", "acceptor": False}
            )
        )
        self.assertTrue(
            is_unresolved_verifier_result(
                {"reward": None, "severity": "stop_training", "acceptor": False}
            )
        )
        self.assertFalse(
            is_unresolved_verifier_result(
                {"reward": 0.0, "severity": "end_episode", "acceptor": False}
            )
        )
        self.assertFalse(
            is_unresolved_verifier_result(
                {"reward": 1.0, "severity": None, "acceptor": True}
            )
        )


class CodingRejectedTests(unittest.TestCase):
    def test_coding_env_refused(self):
        with self.assertRaisesRegex(NotImplementedError, "coding eval is not available"):
            make_env_for_task("coding")


if __name__ == "__main__":
    unittest.main()
