"""
Countdown environment (Reasoning Gym wrapper).

Thin episode adapter around reasoning_gym's ``countdown`` dataset.
Generation and verification are delegated to Reasoning Gym; this module only
owns the one-shot reset/step episode API used by the rest of the project.
"""

from __future__ import annotations

from typing import Any

from ..reasoning_gym_env import (
    ACCEPT_THRESHOLD,
    SEVERITY_END_EPISODE,
    SEVERITY_END_ROLLOUT,
    SEVERITY_STOP_TRAINING,
    ReasoningGymEnv,
    extract_answer,
)

DATASET_NAME = "countdown"


class CountdownEnv(ReasoningGymEnv):
    """Countdown puzzles via Reasoning Gym."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(DATASET_NAME, **kwargs)


__all__ = [
    "ACCEPT_THRESHOLD",
    "DATASET_NAME",
    "CountdownEnv",
    "SEVERITY_END_EPISODE",
    "SEVERITY_END_ROLLOUT",
    "SEVERITY_STOP_TRAINING",
    "extract_answer",
]
