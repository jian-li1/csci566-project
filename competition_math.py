"""
Competition Math environment

Goal:
  Solve one competition mathematics problem and report the final answer.

Action space:

  Agent submits one completion string per episode.

Observation space:
  Agent sees the problem statement, difficulty level, subject, and a text prompt.
  The worked reference solution and its boxed answer are never shown to the agent.

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
  extract_boxed_answer() — extract the last boxed answer from a gold solution
  check_answer()         — compare a submitted answer with the gold answer

Randomness:
  None. The caller binds a row from qwedsacf/competition_math with load_problem().
"""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Any

try:
    from math_verify import parse as _parse_math
    from math_verify import verify as _verify_math
except ImportError:  # The exact/numeric fallback keeps the environment usable alone.
    _parse_math = None
    _verify_math = None

MATH_VERIFY_AVAILABLE = _parse_math is not None and _verify_math is not None

# Error severity → what the training loop should do.
SEVERITY_END_EPISODE = "end_episode"
SEVERITY_END_ROLLOUT = "end_rollout"
SEVERITY_STOP_TRAINING = "stop_training"


def extract_boxed_answer(solution: str) -> str | None:
    """Extract the body of the last balanced ``\\boxed{...}`` in a solution."""
    if not isinstance(solution, str):
        return None

    matches = list(re.finditer(r"\\boxed\s*", solution))
    for match in reversed(matches):
        start = match.end()
        if start >= len(solution):
            continue

        # The MATH dataset uses braces, including nested LaTeX such as \frac{}{}.
        if solution[start] == "{":
            depth = 0
            for index in range(start, len(solution)):
                char = solution[index]
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        return solution[start + 1 : index].strip()
            continue

        # Be permissive for the uncommon ``\boxed 4`` form.
        token = re.match(r"[^\s$.,;]+", solution[start:])
        if token:
            return token.group(0).strip()
    return None


def _unwrap_boxed(answer: str) -> str:
    """Use a boxed body when the submitted answer contains one."""
    boxed = extract_boxed_answer(answer)
    return boxed if boxed is not None else answer


def _normalize_answer(answer: str) -> str:
    """Normalize harmless formatting differences before exact comparison."""
    value = _unwrap_boxed(str(answer)).strip()
    value = value.replace("\u2212", "-").replace("\u00d7", r"\times")
    value = value.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    value = value.replace(r"\left", "").replace(r"\right", "")
    value = value.replace(r"\!", "").replace(r"\,", "").replace(r"\;", "")
    value = value.strip("$ \n\t")
    value = re.sub(r"\.(?=\s*$)", "", value)

    # Keep words when the entire answer is textual; otherwise discard unit prose.
    text_only = re.fullmatch(r"\\(?:text|mathrm)\{([^{}]*)\}", value)
    if text_only:
        value = text_only.group(1)
    else:
        value = re.sub(r"\\(?:text|mathrm)\{[^{}]*\}", "", value)

    # Gold answers sometimes include an assignment (for example, ``x = 2``).
    assignment = re.fullmatch(r"[A-Za-z](?:_\{?\d+\}?)?\s*=\s*(.+)", value)
    if assignment:
        value = assignment.group(1)

    value = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", value)
    value = re.sub(r"\s+", "", value)
    return value.lower()


def _as_fraction(answer: str) -> Fraction | None:
    """Parse a plain number or nested LaTeX fraction for a numeric fallback."""
    value = _normalize_answer(answer)
    while value.startswith("(") and value.endswith(")"):
        value = value[1:-1]

    match = re.fullmatch(r"\\frac\{(.+)\}\{(.+)\}", value)
    if match:
        numerator = _as_fraction(match.group(1))
        denominator = _as_fraction(match.group(2))
        if numerator is None or denominator in {None, Fraction(0)}:
            return None
        return numerator / denominator

    percent = value.endswith(r"\%") or value.endswith("%")
    if percent:
        value = value[:-2] if value.endswith(r"\%") else value[:-1]
    try:
        parsed = Fraction(value)
    except (ValueError, ZeroDivisionError):
        return None
    return parsed / 100 if percent else parsed


def check_answer(answer: str, reference_answer: str) -> dict[str, Any]:
    """
    Check an answer submitted by the agent.

    1) compare normalized strings for deterministic exact matches
    2) use math-verify, when installed, for LaTeX / symbolic equivalence
    3) compare simple numeric values exactly as Fractions
    """
    candidate = _unwrap_boxed((answer or "").strip())
    reference = _unwrap_boxed((reference_answer or "").strip())
    fail = lambda error, verifier=None: {
        "correct": False,
        "answer": candidate,
        "reference_answer": reference,
        "verifier": verifier,
        "error": error,
    }

    if not candidate:
        return fail("empty answer")
    if not reference:
        raise ValueError("reference solution has no boxed answer")

    candidate_normalized = _normalize_answer(candidate)
    reference_normalized = _normalize_answer(reference)
    if candidate_normalized and candidate_normalized == reference_normalized:
        return {
            "correct": True,
            "answer": candidate,
            "reference_answer": reference,
            "verifier": "normalized_exact",
            "error": None,
        }

    if MATH_VERIFY_AVAILABLE:
        try:
            # After tag / box extraction, wrap both values in a LaTeX environment.
            gold = _parse_math(f"${reference}$")
            prediction = _parse_math(f"${candidate}$")
            if gold and prediction and bool(_verify_math(gold, prediction)):
                return {
                    "correct": True,
                    "answer": candidate,
                    "reference_answer": reference,
                    "verifier": "math_verify",
                    "error": None,
                }
        except Exception:  # noqa: BLE001
            # Malformed model output is a scored failure; still try numeric fallback.
            pass

    candidate_number = _as_fraction(candidate)
    reference_number = _as_fraction(reference)
    if candidate_number is not None and candidate_number == reference_number:
        return {
            "correct": True,
            "answer": candidate,
            "reference_answer": reference,
            "verifier": "numeric_exact",
            "error": None,
        }

    return fail("answers are not equivalent", verifier="binary_correctness")


class CompetitionMathEnv:
    name = "competition_math"

    def __init__(self) -> None:
        # Goal / observation: solve this competition problem.
        self.problem: str | None = None

        # Observation: dataset metadata shown to the agent.
        self.level: str | None = None
        self.problem_type: str | None = None

        # Hidden verifier data from the dataset row.
        self.solution: str | None = None
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

        self.extract_boxed_answer = extract_boxed_answer
        self.check_answer = check_answer

    def build_prompt(self) -> str:
        """Turn the current problem into a model prompt without leaking the solution."""
        return (
            f"Solve the following competition mathematics problem.\n\n{self.problem}\n\n"
            "Put step-by-step reasoning in <think>...</think> and only the final "
            "answer in <answer>...</answer>."
        )

    def state(self) -> dict[str, Any]:
        return {
            "problem": self.problem,
            "level": self.level,
            "type": self.problem_type,
            "current_step": self.current_step,
            "last_response": self.last_response,
            "terminated": self.terminated,
            "prompt": self.prompt,
        }

    def load_problem(
        self,
        problem: str,
        solution: str,
        level: str | None = None,
        problem_type: str | None = None,
    ) -> dict[str, Any]:
        """Bind one dataset row and reset all per-episode values."""
        if not isinstance(problem, str) or not problem.strip():
            raise ValueError("problem must be a non-empty string")
        if not isinstance(solution, str) or not solution.strip():
            raise ValueError("solution must be a non-empty string")

        reference_answer = self.extract_boxed_answer(solution)
        if reference_answer is None or not reference_answer.strip():
            raise ValueError(
                "solution does not contain a non-empty balanced \\boxed{...} answer"
            )

        self.problem = problem.strip()
        self.level = level
        self.problem_type = problem_type
        self.solution = solution
        self.reference_answer = reference_answer
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
        # Note: solution / reference_answer are intentionally not returned.
        return {
            "problem": self.problem,
            "level": self.level,
            "type": self.problem_type,
            "prompt": self.prompt,
        }

    def reset(self, row: dict[str, Any]) -> dict[str, Any]:
        """
        Start a new episode from one Competition Math dataset row.

        1. Read problem / solution / level / type from the row.
        2. Extract the final boxed answer from the hidden solution.
        3. Reset internal values: acceptor, reward, answer.
        4. Return the starting observation (metadata + prompt).
        """
        if not isinstance(row, dict):
            raise TypeError(f"row must be a dict, got {type(row)}")
        return self.load_problem(
            problem=row["problem"],
            solution=row["solution"],
            level=row.get("level"),
            problem_type=row.get("type"),
        )

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
                    "level": self.level,
                    "type": self.problem_type,
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

        # Prompt asks for <answer>...</answer>; score only that body.
        tags = re.findall(
            r"<answer>(.*?)</answer>", str(response), flags=re.DOTALL | re.IGNORECASE
        )
        if not tags:
            error = "missing <answer>...</answer>"
            answer = ""
        else:
            answer = tags[-1].strip()
            error = "empty <answer>" if not answer else None
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
