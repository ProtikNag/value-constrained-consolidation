"""Convolutional MinAtar policies: shared conv trunk + action head(s).

The standard MinAtar architecture (Young & Tian, 2019): one 3x3 conv (16
channels) over the 10x10 image, then a fully-connected hidden layer, then the
action logits. There is NO value head -- the method is pure policy gradient
(value estimated from returns), so no critic network is introduced.

Two variants:
    MinAtarCNNPolicy            single shared action head (unified action set)
    MinAtarMultiHeadCNNPolicy   shared conv+FC trunk, one action head per task

The shared conv trunk is where cross-game interference (and thus forgetting)
lives; the constraint protects it.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from crl.policies.base import Policy


def _init_head(head: nn.Linear) -> None:
    """Small final layer -> near-uniform initial policy (low-variance start)."""
    nn.init.orthogonal_(head.weight, gain=0.01)
    nn.init.zeros_(head.bias)


class _CNNTrunk(nn.Module):
    """Conv(→16,3x3) -> ReLU -> flatten -> Linear(hidden) -> ReLU."""

    def __init__(self, in_channels: int, height: int, width: int,
                 hidden_size: int, num_tasks: int, task_conditioned: bool) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)
        conv_out = 16 * (height - 2) * (width - 2)
        self.task_conditioned = task_conditioned
        self.num_tasks = num_tasks
        fc_in = conv_out + (num_tasks if task_conditioned else 0)
        self.fc = nn.Linear(fc_in, hidden_size)

    def forward(self, obs: torch.Tensor, task_id: int) -> torch.Tensor:
        x = torch.relu(self.conv(obs))
        x = x.flatten(start_dim=1)
        if self.task_conditioned:
            one_hot = torch.zeros(x.shape[0], self.num_tasks, device=x.device,
                                  dtype=x.dtype)
            one_hot[:, task_id] = 1.0
            x = torch.cat([x, one_hot], dim=-1)
        return torch.relu(self.fc(x))


class MinAtarCNNPolicy(Policy):
    """Shared conv trunk + a single shared action head (unified action set)."""

    def __init__(self, obs_shape: tuple[int, int, int], num_actions: int,
                 hidden_size: int = 128, num_tasks: int = 0,
                 task_conditioned: bool = False) -> None:
        super().__init__()
        c, h, w = obs_shape
        self.trunk = _CNNTrunk(c, h, w, hidden_size, num_tasks, task_conditioned)
        self.head = nn.Linear(hidden_size, num_actions)
        _init_head(self.head)

    def dist(self, obs: torch.Tensor, task_id: int) -> Categorical:
        return Categorical(logits=self.head(self.trunk(obs, task_id)))


class MinAtarMultiHeadCNNPolicy(Policy):
    """Shared conv+FC trunk with one action head per task (hard task routing)."""

    def __init__(self, obs_shape: tuple[int, int, int], num_actions: int,
                 hidden_size: int = 128, num_tasks: int = 0,
                 task_conditioned: bool = False) -> None:
        super().__init__()
        if num_tasks <= 0:
            raise ValueError("MinAtarMultiHeadCNNPolicy requires num_tasks > 0.")
        c, h, w = obs_shape
        self.num_tasks = num_tasks
        self.trunk = _CNNTrunk(c, h, w, hidden_size, num_tasks, task_conditioned)
        self.heads = nn.ModuleList(
            [nn.Linear(hidden_size, num_actions) for _ in range(num_tasks)])
        for head in self.heads:
            _init_head(head)

    def dist(self, obs: torch.Tensor, task_id: int) -> Categorical:
        if not 0 <= task_id < self.num_tasks:
            raise IndexError(f"task_id {task_id} out of range [0, {self.num_tasks})")
        return Categorical(logits=self.heads[task_id](self.trunk(obs, task_id)))
