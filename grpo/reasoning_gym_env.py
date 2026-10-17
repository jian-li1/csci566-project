"""
Shared Reasoning Gym env wrapper.

Subclassed by countdown / knights_knaves / shortest_path.

Reasoning Gym already provides:
  - create_dataset(name, ...)  → puzzles as {question, answer, metadata}
  - score_answer(answer, entry) → float reward in [0, 1]

This file adds the one-shot episode API used by GRPO (same shape as math/mbpp):
  reset / load_problem → model sees prompt → step(completion) → reward + done

Typical train loop:
  env.load_problem(entry)   # or env.reset(index=i)
  out = env.step(completion)
  reward = out["reward"]    # float, or None if unscored
"""

from __future__ import annotations

import re
from typing import Any

import reasoning_gym
from reasoning_gym import get_score_answer_fn

# ---------------------------------------------------------------------------
# Severity — what the training loop should do after a failure
# (same names as math/mbpp; set explicitly, not via error-string matching)
# ---------------------------------------------------------------------------
SEVERITY_END_EPISODE = "end_episode"  # scored wrong / bad format; keep training
SEVERITY_END_ROLLOUT = "end_rollout"  # empty / crash / bad call order; unscored
SEVERITY_STOP_TRAINING = "stop_training"  # fatal infra error; caller should halt

# RG scores at/above this count as a fully accepted solve.
ACCEPT_THRESHOLD = 0.99


def extract_answer(response: str) -> tuple[str | None, str | None]:
    """
    Pull the last <answer>...</answer> body from a model completion.

    Does NOT check correctness — only format/presence.
    Returns (answer, error). error is None on success.
    """
    # Empty / whitespace-only completion → unscored later in step().
    if response is None or not str(response).strip():
        return None, "empty response"

    # DOTALL so answers can span lines; IGNORECASE for <Answer> etc.
    tags = re.findall(
        r"<answer>(.*?)</answer>", str(response), flags=re.DOTALL | re.IGNORECASE
    )
    if not tags:
        return None, "missing <answer>...</answer>"

    # If the model wrote several <answer> blocks, grade the last one.
    answer = tags[-1].strip()
    if not answer:
        return None, "empty <answer>"
    return answer, None


class ReasoningGymEnv:
    """
    One-shot episode wrapper around one Reasoning Gym procedural dataset.

    Subclasses only pass a different dataset_name (countdown, knights_knaves, ...).
    """

    def __init__(self, dataset_name: str, **kwargs: Any) -> None:
        """
        Build the RG dataset + scorer and clear episode state.

        kwargs:
          size, seed, accept_threshold — consumed here
          anything else — forwarded to reasoning_gym.create_dataset(...)
        """
        if not isinstance(dataset_name, str) or not dataset_name.strip():
            raise ValueError("dataset_name must be a non-empty string")

        size = int(kwargs.pop("size", 500))
        seed = int(kwargs.pop("seed", 0))
        accept_threshold = float(kwargs.pop("accept_threshold", ACCEPT_THRESHOLD))

        self.dataset_name = dataset_name.strip()
        self.name = self.dataset_name
        self.accept_threshold = accept_threshold

        # Procedural puzzle generator for this task.
        self.dataset = reasoning_gym.create_dataset(
            self.dataset_name, size=size, seed=seed, **kwargs
        )
        # Task-specific grader (same as dataset.score_answer).
        self._score_fn = get_score_answer_fn(self.dataset_name)

        # Bound RG entry — full dict stays hidden; only question → prompt is shown.
        self.entry: dict[str, Any] | None = None
        self.question: str | None = None
        self.prompt: str | None = None

        # Per-episode fields cleared on every load_problem().
        self.last_response: str | None = None  # raw model completion
        self.answer: str | None = None  # extracted <answer> body (or raw while parsing)
        self.current_step: int = 0
        self.acceptor: bool = False  # True if reward >= accept_threshold
        self.reward: float | None = None  # RG float, 0.0, or None (unscored)
        self.terminated: bool = False
        self.severity: str | None = None

    def build_prompt(self) -> str:
        """RG question + instructions to put the final answer in <answer> tags."""
        return (
            f"{self.question}\n\n"
            "Put step-by-step reasoning in <think>...</think> and only the final "
            "answer in <answer>...</answer>."
        )

    def load_problem(self, entry: dict[str, Any]) -> dict[str, Any]:
        """
        Bind one RG entry and start a fresh episode.

        entry must look like: {question, answer, metadata?}
        Returns an observation the agent may see (no gold answer).
        """
        if not isinstance(entry, dict):
            raise TypeError(f"entry must be a dict, got {type(entry)}")

        question = entry.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("entry['question'] must be a non-empty string")
        if "answer" not in entry:
            raise ValueError("entry must include an 'answer' field for scoring")

        metadata = entry.get("metadata")
        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError("entry['metadata'] must be a dict when provided")

        # Store full entry for scoring; never put answer/metadata gold in the obs.
        self.entry = entry
        self.question = question.strip()
        self.prompt = self.build_prompt()

        # Wipe leftover state from the previous episode.
        self.last_response = None
        self.answer = None
        self.current_step = 0
        self.acceptor = False
        self.reward = None
        self.terminated = False
        self.severity = None

        return {
            "dataset_name": self.dataset_name,
            "question": self.question,
            "prompt": self.prompt,
        }

    def reset(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """
        Start a new episode, then delegate to load_problem.

        Ways to choose the puzzle:
          reset() / reset(index=i)     — sample from self.dataset
          reset(entry=...) / reset(row=...) / reset({...}) — bind a ready-made entry
        """
        entry = kwargs.pop("entry", None)
        if entry is None and "row" in kwargs:
            entry = kwargs.pop("row")

        index = kwargs.pop("index", None)
        if args:
            if len(args) != 1:
                raise TypeError("reset expects at most one positional argument")
            if entry is not None or index is not None:
                raise ValueError("pass only one of index, entry/row, or a positional arg")
            arg = args[0]
            if isinstance(arg, dict):
                entry = arg
            else:
                index = arg

        if kwargs:
            raise TypeError(f"unexpected keyword arguments: {sorted(kwargs)}")
        if entry is not None and index is not None:
            raise ValueError("pass only one of index or entry/row")

        if entry is not None:
            return self.load_problem(entry)
        if index is None:
            index = 0
        return self.load_problem(self.dataset[int(index)])

    def score_answer(self, answer: str) -> float:
        """
        Grade an already-extracted answer string with Reasoning Gym.

        Requires load_problem/reset first so self.entry is set.
        """
        if self.entry is None:
            raise RuntimeError("score_answer called before reset/load_problem")
        return float(self._score_fn(answer, self.entry))

    def state(self) -> dict[str, Any]:
        """Public snapshot of episode bookkeeping (no gold answer)."""
        return {
            "dataset_name": self.dataset_name,
            "question": self.question,
            "current_step": self.current_step,
            "last_response": self.last_response,
            "terminated": self.terminated,
            "prompt": self.prompt,
        }

    def _step_result(
        self, check: dict[str, Any], *, reward: float | None | object = ...
    ) -> dict[str, Any]:
        """
        Pack the standard step() return dict so every exit path matches.

        reward=... (default) → use self.reward
        reward=None          → override for this return only (e.g. step-after-terminate)
        """
        return {
            "observation": {
                "dataset_name": self.dataset_name,
                "question": self.question,
            },
            "answer": self.answer,
            "reward": self.reward if reward is ... else reward,
            "acceptor": self.acceptor,
            "terminated": self.terminated,
            "severity": self.severity,
            "check": check,
            "state": self.state(),
        }

    def step(self, response: str) -> dict[str, Any]:
        """
        Score one model completion and end the episode (one-shot).

        Flow:
          1. Guards (step-before-reset / step-after-terminate)
          2. extract_answer — on format error, set severity and return
          3. score_answer via Reasoning Gym
          4. Return reward / acceptor / terminated / severity / check / state
        """
        # --- Guard: nothing loaded yet ---
        if self.entry is None or self.question is None:
            result = {
                "correct": False,
                "answer": "" if response is None else str(response),
                "score": None,
                "error": "step before reset",
            }
            self.reward = None
            self.severity = SEVERITY_END_ROLLOUT
            self.terminated = True
            return self._step_result(result)

        # --- Guard: episode already finished ---
        # Keep self.reward from the real scored step; this call itself is unscored.
        if self.terminated:
            result = {
                "correct": False,
                "answer": "" if response is None else str(response),
                "score": None,
                "error": "step after terminate",
            }
            self.severity = SEVERITY_END_ROLLOUT
            return self._step_result(result, reward=None)

        # One submit ends the episode either way.
        self.current_step += 1
        self.last_response = response
        self.answer = response
        self.acceptor = False
        self.reward = None
        self.severity = None
        self.terminated = True

        # --- Parse <answer> tags ---
        answer, error = extract_answer("" if response is None else str(response))
        if error == "empty response":
            # Unscored — not a normal wrong answer.
            result = {
                "correct": False,
                "answer": "" if response is None else str(response),
                "score": None,
                "error": error,
            }
            self.severity = SEVERITY_END_ROLLOUT
            return self._step_result(result)

        if error is not None:
            # Missing / empty tags → scored failure (reward 0).
            result = {
                "correct": False,
                "answer": "" if response is None else str(response),
                "score": 0.0,
                "error": error,
            }
            self.reward = 0.0
            self.severity = SEVERITY_END_EPISODE
            return self._step_result(result)

        # --- Grade with Reasoning Gym ---
        try:
            self.answer = answer
            score = self.score_answer(answer)
        except NotImplementedError:
            raise
        except Exception as exc:  # noqa: BLE001
            result = {
                "correct": False,
                "answer": answer,
                "score": None,
                "error": f"verifier crash: {exc}",
            }
            self.reward = None
            if isinstance(exc, (MemoryError, SystemError)):
                self.severity = SEVERITY_STOP_TRAINING
            else:
                self.severity = SEVERITY_END_ROLLOUT
            return self._step_result(result)

        self.reward = float(score)
        self.acceptor = self.reward >= self.accept_threshold
        self.severity = None if self.acceptor else SEVERITY_END_EPISODE
        result = {
            "correct": self.acceptor,
            "answer": answer,
            "score": self.reward,
            "error": None,
        }
        return self._step_result(result)
