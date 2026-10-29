"""Task and task-family abstractions.

A *task* is one MDP ``M_i = (S, A, P_i, r_i, rho_i, gamma)`` (eq 1); a
*family* is the ordered sequence of tasks presented to the learner. State
and action spaces are shared across the family; only dynamics, rewards and
initial distributions vary.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import torch


@dataclass
class TaskSpec:
    """Identity and parameters of one task in a family."""

    task_id: int  # 0-based arrival index
    name: str
    params: dict[str, Any] = field(default_factory=dict)


class Task(ABC):
    """One MDP. Environments returned by :meth:`make_env` are fresh copies."""

    # What counts as "solving" an episode in the rollout performance metric.
    # True  -> the episode TERMINATES at an absorbing success state (e.g. the
    #          gridworld reaching its goal).
    # False -> success is SURVIVING to the horizon without terminating (e.g.
    #          CartPole, where termination means the pole fell = failure).
    success_on_termination: bool = True

    def __init__(self, spec: TaskSpec, gamma: float) -> None:
        self.spec = spec
        self.gamma = gamma

    @abstractmethod
    def make_env(self) -> gym.Env:
        """Instantiate a Gymnasium environment for this task."""

    def make_vector_env(self, num_envs: int, async_mode: bool = False, **kwargs):
        """A vectorized bank of ``num_envs`` copies of :meth:`make_env`.

        Generic default for flat-observation families (e.g. gridworld) so they can
        run through the PPO backend's :class:`RolloutCollector`, which is otherwise
        Atari-specific. Uses ``SAME_STEP`` autoreset to match the collector's GAE
        bootstrap convention. Atari overrides this with its wrapper-aware version;
        Atari-only kwargs (``clip_rewards``/``noop_override``/...) are accepted and
        ignored here.
        """
        from gymnasium.vector import AsyncVectorEnv, AutoresetMode, SyncVectorEnv

        fns = [self.make_env for _ in range(int(num_envs))]
        cls = AsyncVectorEnv if async_mode else SyncVectorEnv
        return cls(fns, autoreset_mode=AutoresetMode.SAME_STEP)


class TabularTask(Task):
    """A task that additionally exposes exact MDP tensors.

    Attributes:
        transition: ``[S, A, S]`` next-state distribution ``P_i``.
        reward: ``[S, A]`` expected immediate reward ``r_i``.
        initial_dist: ``[S]`` initial state distribution ``rho_i``.

    Exact (dynamic-programming) estimators require these; sampled
    estimators only need :meth:`make_env`.
    """

    transition: torch.Tensor
    reward: torch.Tensor
    initial_dist: torch.Tensor


class TaskFamily(ABC):
    """Ordered task sequence with shared observation/action spaces."""

    tasks: list[Task]
    obs_dim: int
    num_actions: int
    is_tabular: bool = False

    def __len__(self) -> int:
        return len(self.tasks)
