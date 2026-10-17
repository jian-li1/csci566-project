"""
Shared Reasoning Gym env wrapper.

Subclassed by countdown / knights_knaves / shortest_path.
RG gives: create_dataset + score_answer.
This file should add: reset/load, step, parse <answer>, return reward/etc.
"""

from __future__ import annotations

import re
from typing import Any

import reasoning_gym
from reasoning_gym import get_score_answer_fn

# severity constants (same names as math/mbpp)
SEVERITY_END_EPISODE = "end_episode"
SEVERITY_END_ROLLOUT = "end_rollout"
SEVERITY_STOP_TRAINING = "stop_training"

ACCEPT_THRESHOLD = 0.99


def extract_answer(response: str) -> tuple[str | None, str | None]:
    """Pull the last <answer>...</answer> body. Returns (answer, error)."""
    if response is None or not str(response).strip():
        return None, "empty response"

    tags = re.findall(
        r"<answer>(.*?)</answer>", str(response), flags=re.DOTALL | re.IGNORECASE
    )
    if not tags:
        return None, "missing <answer>...</answer>"

    answer = tags[-1].strip()
    if not answer:
        return None, "empty <answer>"
    return answer, None


class ReasoningGymEnv:
    def __init__(self, dataset_name: str, **kwargs: Any) -> None:
        if not isinstance(dataset_name, str) or not dataset_name.strip():
            raise ValueError("dataset_name must be a non-empty string")

        size = int(kwargs.pop("size", 500))
        seed = int(kwargs.pop("seed", 0))
        accept_threshold = float(kwargs.pop("accept_threshold", ACCEPT_THRESHOLD))

        self.dataset_name = dataset_name.strip()
        self.name = self.dataset_name
        self.accept_threshold = accept_threshold
        self.dataset = reasoning_gym.create_dataset(
            self.dataset_name, size=size, seed=seed, **kwargs
        )
        self._score_fn = get_score_answer_fn(self.dataset_name)

        self.entry: dict[str, Any] | None = None
        self.question: str | None = None
        self.prompt: str | None = None

        self.last_response: str | None = None
        self.answer: str | None = None
        self.current_step: int = 0
        self.acceptor: bool = False
        self.reward: float | None = None
        self.terminated: bool = False
        self.severity: str | None = None

    def build_prompt(self) -> str:
        return (
            f"{self.question}\n\n"
            "Put step-by-step reasoning in <think>...</think> and only the final "
            "answer in <answer>...</answer>."
        )

    def load_problem(self, entry: dict[str, Any]) -> dict[str, Any]:
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

        self.entry = entry
        self.question = question.strip()
        self.prompt = self.build_prompt()

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
        ...

    def score_answer(self, answer: str) -> float:
        ...

    def step(self, response: str) -> dict[str, Any]:
        ...
