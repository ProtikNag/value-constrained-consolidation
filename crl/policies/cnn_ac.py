"""Actor-critic CNN policies for Atari (the PPO backend).

Standard Nature-DQN trunk (Mnih et al., 2015): conv(32,8,s4) -> conv(64,4,s2)
-> conv(64,3,s1) -> FC(512). Two policy variants share this trunk design:

* :class:`AtariActorCriticPolicy`          -- one shared actor head + one critic
                                              head (unified action set).
* :class:`AtariMultiHeadActorCriticPolicy` -- one actor head AND one critic head
                                              per task over the shared trunk
                                              (the standard multi-head continual
                                              setup; forgetting lives in the
                                              shared trunk, which the constraint
                                              protects).

Both optionally append a task one-hot to the trunk features
(``task_conditioned``), and both satisfy the :class:`~crl.policies.base.Policy`
interface via ``dist(obs, task_id) -> Categorical`` while adding
``value(obs, task_id)`` and a fused ``dist_value``. uint8 frames are normalized
to ``[0, 1]`` inside the forward pass. Only the ACTOR is ever constrained (in the
global phase); the critic head(s) are optimized by standard PPO value regression.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from crl.policies.base import Policy


def _ortho(layer: nn.Module, gain: float) -> nn.Module:
    nn.init.orthogonal_(layer.weight, gain=gain)
    if getattr(layer, "bias", None) is not None:
        nn.init.zeros_(layer.bias)
    return layer


class _NatureTrunk(nn.Module):
    """Nature-CNN body -> ``[B, hidden_size]`` features (optionally task-conditioned)."""

    def __init__(self, obs_shape: tuple[int, int, int], hidden_size: int,
                 num_tasks: int, task_conditioned: bool) -> None:
        super().__init__()
        c, h, w = obs_shape
        self.task_conditioned = task_conditioned
        self.num_tasks = num_tasks
        self.conv = nn.Sequential(
            _ortho(nn.Conv2d(c, 32, kernel_size=8, stride=4), gain=2**0.5),
            nn.ReLU(),
            _ortho(nn.Conv2d(32, 64, kernel_size=4, stride=2), gain=2**0.5),
            nn.ReLU(),
            _ortho(nn.Conv2d(64, 64, kernel_size=3, stride=1), gain=2**0.5),
            nn.ReLU(),
        )
        with torch.no_grad():
            conv_out = self.conv(torch.zeros(1, c, h, w)).flatten(1).shape[1]
        fc_in = conv_out + (num_tasks if task_conditioned else 0)
        self.fc = _ortho(nn.Linear(fc_in, hidden_size), gain=2**0.5)

    def forward(self, obs: torch.Tensor, task_id: int) -> torch.Tensor:
        x = obs.float() / 255.0  # uint8 [0,255] -> [0,1]
        x = self.conv(x).flatten(start_dim=1)
        if self.task_conditioned:
            one_hot = torch.zeros(x.shape[0], self.num_tasks, device=x.device,
                                  dtype=x.dtype)
            one_hot[:, task_id] = 1.0
            x = torch.cat([x, one_hot], dim=-1)
        return torch.relu(self.fc(x))


def _actor_head(hidden_size: int, num_actions: int) -> nn.Linear:
    # Small actor head -> near-uniform initial policy (standard PPO init).
    return _ortho(nn.Linear(hidden_size, num_actions), gain=0.01)


def _critic_head(hidden_size: int) -> nn.Linear:
    return _ortho(nn.Linear(hidden_size, 1), gain=1.0)


class AtariActorCriticPolicy(Policy):
    """Shared Nature-CNN trunk with a single actor head + critic head."""

    def __init__(self, obs_shape, num_actions, hidden_size=512, num_tasks=0,
                 task_conditioned=False) -> None:
        super().__init__()
        self.trunk = _NatureTrunk(obs_shape, hidden_size, num_tasks, task_conditioned)
        self.actor = _actor_head(hidden_size, num_actions)
        self.critic = _critic_head(hidden_size)

    def dist(self, obs: torch.Tensor, task_id: int) -> Categorical:
        return Categorical(logits=self.actor(self.trunk(obs, task_id)))

    def value(self, obs: torch.Tensor, task_id: int) -> torch.Tensor:
        return self.critic(self.trunk(obs, task_id)).squeeze(-1)

    def dist_value(self, obs: torch.Tensor, task_id: int):
        feats = self.trunk(obs, task_id)
        return Categorical(logits=self.actor(feats)), self.critic(feats).squeeze(-1)


class AtariMultiHeadActorCriticPolicy(Policy):
    """Shared Nature-CNN trunk with per-task actor AND critic heads.

    The convolutional trunk is shared across all games (where cross-game
    interference / forgetting lives, protected by the constraint); each task
    routes to its own actor and critic head via ``task_id``.
    """

    def __init__(self, obs_shape, num_actions, hidden_size=512, num_tasks=0,
                 task_conditioned=False) -> None:
        super().__init__()
        if num_tasks <= 0:
            raise ValueError("AtariMultiHeadActorCriticPolicy requires num_tasks > 0.")
        self.num_tasks = num_tasks
        self.trunk = _NatureTrunk(obs_shape, hidden_size, num_tasks, task_conditioned)
        self.actors = nn.ModuleList(
            [_actor_head(hidden_size, num_actions) for _ in range(num_tasks)])
        self.critics = nn.ModuleList(
            [_critic_head(hidden_size) for _ in range(num_tasks)])

    def _check(self, task_id: int) -> None:
        if not 0 <= task_id < self.num_tasks:
            raise IndexError(f"task_id {task_id} out of range [0, {self.num_tasks})")

    def dist(self, obs: torch.Tensor, task_id: int) -> Categorical:
        self._check(task_id)
        return Categorical(logits=self.actors[task_id](self.trunk(obs, task_id)))

    def value(self, obs: torch.Tensor, task_id: int) -> torch.Tensor:
        self._check(task_id)
        return self.critics[task_id](self.trunk(obs, task_id)).squeeze(-1)

    def dist_value(self, obs: torch.Tensor, task_id: int):
        self._check(task_id)
        feats = self.trunk(obs, task_id)
        return (Categorical(logits=self.actors[task_id](feats)),
                self.critics[task_id](feats).squeeze(-1))
