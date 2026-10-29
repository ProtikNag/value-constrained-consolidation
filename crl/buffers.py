"""Per-task trajectory storage.

Stores behavior log-probabilities alongside actions so future off-policy
constraint estimators (importance-weighted replay, frozen-reference
surrogates) can be added without re-collecting data. The Monte-Carlo
estimator deposits its most recent batches here as a side effect.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch


@dataclass
class Trajectory:
    """One episode collected under a behavior policy."""

    obs: torch.Tensor  # [T, obs_dim]
    actions: torch.Tensor  # [T]
    rewards: torch.Tensor  # [T]
    behavior_logps: torch.Tensor  # [T] log pi_b(a_t | s_t) at collection time
    terminated: bool  # environment terminal state reached (not truncation)


class TaskBuffer:
    """Bounded FIFO of trajectories for a single task. Space O(capacity * T)."""

    def __init__(self, capacity: int = 256) -> None:
        self._trajectories: deque[Trajectory] = deque(maxlen=capacity)

    def add(self, trajectory: Trajectory) -> None:
        self._trajectories.append(trajectory)

    def __len__(self) -> int:
        return len(self._trajectories)

    def all(self) -> list[Trajectory]:
        return list(self._trajectories)


class BufferSet:
    """One :class:`TaskBuffer` per task id, created lazily."""

    def __init__(self, capacity: int = 256) -> None:
        self._capacity = capacity
        self._buffers: dict[int, TaskBuffer] = {}

    def for_task(self, task_id: int) -> TaskBuffer:
        if task_id not in self._buffers:
            self._buffers[task_id] = TaskBuffer(self._capacity)
        return self._buffers[task_id]
