"""
OpenMathReasoning environment

Goal:
  Solve one mathematics problem and report the final answer.

Action space:

  Agent submits one completion string per episode.

Observation space:
  Agent sees the problem statement, source metadata, and a text prompt.
  The expected answer is never shown to the agent.

Internal state:
  Stores the completion, extracted answer, reference answer, acceptor flag, and reward.

Step / reward:
  Reward is assigned when the agent submits a completion.
  Verifier is check_answer(). Success → 1, scored failure → 0,
  unusable action / verifier crash → None (do not train on this sample).
  Episode terminates after the submit either way.

Failure handling (by severity; set explicitly, not via error-string matching):
  end_episode — bad/wrong answer for this problem (reward 0)
  end_rollout — empty action / verifier crash (reward None)
  stop_training — fatal infra error (reward None; caller should halt)

Dependencies:
  _extract_answer() — extract the final answer from the agent completion
  check_answer()    — compare a submitted answer with the expected answer

Randomness:
  None. The caller binds a row from nvidia/OpenMathReasoning with load_problem().
"""

from __future__ import annotations

import re
from typing import Any

from grpo.math.environment import check_answer

# Error severity → what the training loop should do.
SEVERITY_END_EPISODE = "end_episode"
SEVERITY_END_ROLLOUT = "end_rollout"
SEVERITY_STOP_TRAINING = "stop_training"


def _extract_answer(response: str) -> tuple[str | None, str | None]:
    """
    Pull the final standalone <answer>...</answer> body from a completion.

    OpenMathReasoning answers can contain literal inequality signs, so this keeps
    the shared final-tag contract while allowing ``<`` and ``>`` in the body.
    Returns (answer, error). error is None on success.
    """
    if response is None or not str(response).strip():
        return None, "empty response"

    text = str(response)
    openings = list(re.finditer(r"(?is)<answer>", text))
    if not openings:
        return None, "missing <answer>...</answer>"

    # Use the last opening tag so a mention inside <think> cannot steal the body.
    match = re.fullmatch(
        r"(?is)(?P<body>.*?)</answer>\s*", text[openings[-1].end() :]
    )
    if match is None:
        return None, "missing <answer>...</answer>"
    answer = match.group("body").strip()
    if not answer:
        return None, "empty <answer>"
    return answer, None


class OpenMathReasoningEnv:
    name = "openmathreasoning"

    def __init__(self) -> None:
        # Goal / observation: solve this mathematics problem.
        self.problem: str | None = None

        # Observation: dataset metadata shown to the agent.
        self.problem_source: str | None = None
        self.problem_type: str | None = None

        # Hidden verifier data from the dataset row.
        self.reference_answer: str | None = None

        # Text prompt for the model (built when a problem is loaded).
        self.prompt: str | None = None

        # Action / internal state.
        self.last_response: str | None = None  # raw completion from the agent
        self.answer: str | None = None  # extracted <answer> body (scored answer)

        # Episode progress.
        self.current_step: int = 0

        # Internal episode values.
        self.acceptor: bool = False  # whether the submitted answer was accepted
        self.reward: float | None = None  # 1 / 0 / None (unscored)
        self.terminated: bool = False
        self.severity: str | None = None

        self.check_answer = check_answer

    def build_prompt(self) -> str:
        """Turn the current problem into a model prompt without leaking the answer."""
        return (
            f"Solve the following mathematics problem.\n\n{self.problem}\n\n"
            "Put step-by-step reasoning in <think>...</think> and only the final "
            "answer in <answer>...</answer>."
        )

    def state(self) -> dict[str, Any]:
        return {
            "problem": self.problem,
            "problem_source": self.problem_source,
            "problem_type": self.problem_type,
            "current_step": self.current_step,
            "last_response": self.last_response,
            "terminated": self.terminated,
            "prompt": self.prompt,
        }

    def load_problem(
        self,
        problem: str | dict[str, Any],
        expected_answer: str | None = None,
        problem_source: str | None = None,
        problem_type: str | None = None,
    ) -> dict[str, Any]:
        """
        Bind one dataset row and reset all per-episode values.

        The preferred form matches ``ReasoningGymEnv.load_problem(entry)``::

            env.load_problem({
                "problem": "...",
                "expected_answer": "...",
                "problem_source": "...",
                "problem_type": "...",
            })

        The positional form matches the Competition Math environment::

            env.load_problem(problem, expected_answer, problem_source, problem_type)
        """
        if isinstance(problem, dict):
            if (
                expected_answer is not None
                or problem_source is not None
                or problem_type is not None
            ):
                raise TypeError(
                    "when problem is a dict, do not pass expected_answer, "
                    "problem_source, or problem_type"
                )
            row = problem
            problem = row["problem"]
            expected_answer = row["expected_answer"]
            problem_source = row.get("problem_source")
            problem_type = row.get("problem_type")

        if not isinstance(problem, str) or not problem.strip():
            raise ValueError("problem must be a non-empty string")
        if not isinstance(expected_answer, str) or not expected_answer.strip():
            raise ValueError("expected_answer must be a non-empty string")

        self.problem = problem.strip()
        self.problem_source = problem_source
        self.problem_type = problem_type
        self.reference_answer = expected_answer.strip()
        self.prompt = self.build_prompt()

        # Reset internal episode values.
        self.last_response = None
        self.answer = None
        self.current_step = 0
        self.acceptor = False
        self.reward = None
        self.terminated = False
        self.severity = None

        # Give the agent the start of the problem (metadata + prompt).
        # Note: expected_answer / reference_answer are intentionally not returned.
        return {
            "problem": self.problem,
            "problem_source": self.problem_source,
            "problem_type": self.problem_type,
            "prompt": self.prompt,
        }

    def reset(self, row: dict[str, Any]) -> dict[str, Any]:
        """
        Start a new episode from one OpenMathReasoning dataset row.

        1. Read problem / expected_answer / source metadata from the row.
        2. Store the expected answer as hidden verifier data.
        3. Reset internal values: acceptor, reward, answer.
        4. Return the starting observation (metadata + prompt).
        """
        if not isinstance(row, dict):
            raise TypeError(f"row must be a dict, got {type(row)}")
        return self.load_problem(row)

    def step(self, response: str) -> dict[str, Any]:
        """
        Agent submits a completion (reward timing = submit).

        Extracts <answer>...</answer>, then scores with check_answer().
        reward: 1 correct, 0 scored failure, None unscored (skip train).
        severity: set explicitly for failures (not via error-string matching).

        Rejects step-before-reset and step-after-terminate (reward None).
        """

        def _out(
            result: dict[str, Any], *, reward: float | None | object = ...
        ) -> dict[str, Any]:
            return {
                "observation": {
                    "problem": self.problem,
                    "problem_source": self.problem_source,
                    "problem_type": self.problem_type,
                },
                "answer": self.answer,
                "reward": self.reward if reward is ... else reward,
                "acceptor": self.acceptor,
                "terminated": self.terminated,
                "severity": self.severity,
                "check": result,
                "state": self.state(),
            }

        # Episode lifecycle guards — check before mutating scored state.
        if self.problem is None or self.reference_answer is None:
            result = {
                "correct": False,
                "answer": "" if response is None else str(response),
                "reference_answer": None,
                "verifier": None,
                "error": "step before reset",
            }
            self.reward = None
            self.severity = SEVERITY_END_ROLLOUT
            self.terminated = True
            return _out(result)

        if self.terminated:
            result = {
                "correct": False,
                "answer": "" if response is None else str(response),
                "reference_answer": self.reference_answer,
                "verifier": None,
                "error": "step after terminate",
            }
            # Keep prior self.reward/answer; this return is unscored.
            self.severity = SEVERITY_END_ROLLOUT
            return _out(result, reward=None)

        self.current_step += 1
        self.last_response = response
        self.answer = response
        self.acceptor = False
        self.reward = None
        self.severity = None
        self.terminated = True
        result: dict[str, Any]

        if response is None or str(response).strip() == "":
            result = {
                "correct": False,
                "answer": "" if response is None else str(response),
                "reference_answer": self.reference_answer,
                "verifier": None,
                "error": "empty response",
            }
            self.severity = SEVERITY_END_ROLLOUT
            return _out(result)

        # Prompt asks for a final standalone <answer>...</answer>; score only that body.
        answer, error = _extract_answer(str(response))
        if error is not None:
            result = {
                "correct": False,
                "answer": str(response),
                "reference_answer": self.reference_answer,
                "verifier": None,
                "error": error,
            }
            self.reward = 0.0
            self.severity = SEVERITY_END_EPISODE
            return _out(result)

        try:
            self.answer = answer
            result = self.check_answer(answer, self.reference_answer)
        except NotImplementedError:
            raise
        except Exception as exc:  # noqa: BLE001
            result = {
                "correct": False,
                "answer": answer,
                "reference_answer": self.reference_answer,
                "verifier": None,
                "error": str(exc),
            }
            # Typed fatal vs rollout crash — no substring matching on messages.
            if isinstance(exc, (MemoryError, SystemError)):
                self.severity = SEVERITY_STOP_TRAINING
            else:
                self.severity = SEVERITY_END_ROLLOUT
            return _out(result)  # reward stays None

        correct = bool(result.get("correct", False))
        self.acceptor = correct
        if correct:
            self.reward = 1.0
            self.severity = None
        else:
            self.reward = 0.0
            self.severity = SEVERITY_END_EPISODE
        return _out(result)
