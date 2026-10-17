"""
Shared Reasoning Gym env wrapper.

Subclassed by countdown / knights_knaves / shortest_path.
RG gives: create_dataset + score_answer.
This file should add: reset/load, step, parse <answer>, return reward/etc.
"""

from __future__ import annotations

import re
from typing import Any

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
        ...

    def build_prompt(self) -> str:
        ...

    def load_problem(self, entry: dict[str, Any]) -> dict[str, Any]:
        ...

    def reset(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        ...

    def score_answer(self, answer: str) -> float:
        ...

    def step(self, response: str) -> dict[str, Any]:
        ...
