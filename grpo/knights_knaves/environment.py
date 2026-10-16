"""
Knights and Knaves environment (Reasoning Gym wrapper).

Thin episode adapter around reasoning_gym's ``knights_knaves`` dataset.
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

DATASET_NAME = "knights_knaves"


class KnightsKnavesEnv(ReasoningGymEnv):
    """Knights-and-knaves logic puzzles via Reasoning Gym."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(DATASET_NAME, **kwargs)


__all__ = [
    "ACCEPT_THRESHOLD",
    "DATASET_NAME",
    "KnightsKnavesEnv",
    "SEVERITY_END_EPISODE",
    "SEVERITY_END_ROLLOUT",
    "SEVERITY_STOP_TRAINING",
    "extract_answer",
]
