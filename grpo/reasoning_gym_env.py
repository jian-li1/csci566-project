"""
Shared Reasoning Gym env wrapper.

Subclassed by countdown / knights_knaves / shortest_path.
RG gives: create_dataset + score_answer.
This file should add: reset/load, step, parse <answer>, return reward/etc.
"""

from __future__ import annotations

from typing import Any

# severity constants (same names as math/mbpp)
SEVERITY_END_EPISODE = "end_episode"
SEVERITY_END_ROLLOUT = "end_rollout"
SEVERITY_STOP_TRAINING = "stop_training"

# accept threshold for score_answer


def extract_answer(response: str) -> tuple[str | None, str | None]:
    # parse last <answer>...</answer> -> (answer, error)
    ...


class ReasoningGymEnv:
    def __init__(self, dataset_name: str, **kwargs: Any) -> None:
        # create_dataset(dataset_name, ...)
        # get_score_answer_fn(dataset_name)
        ...

    def build_prompt(self) -> str:
        # question + <think>/<answer> instructions
        ...

    def load_problem(self, entry: dict[str, Any]) -> dict[str, Any]:
        # bind {question, answer, metadata}, return obs with prompt
        ...

    def reset(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        # by index into dataset, or by entry/row dict
        ...

    def score_answer(self, answer: str) -> float:
        # RG score_answer / get_score_answer_fn
        ...

    def step(self, response: str) -> dict[str, Any]:
        # parse answer -> score -> return
        # {observation, answer, reward, acceptor, terminated, severity, check, state}
        ...
