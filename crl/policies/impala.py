"""IMPALA-CNN actor-critic policies (Espeholt et al. 2018) for Atari.

A deeper, residual convolutional trunk than the Nature-CNN, used for the
single-task expert models and the (larger) global model. "Large" variant:
three conv sequences with channels (32, 64, 64), each = conv 3x3 -> maxpool 3x3/2
-> 2 residual blocks, then ReLU -> flatten -> FC(hidden) -> ReLU. Optionally
appends a task one-hot before the FC (for the multi-head global). Same
`dist` / `value` / `dist_value` interface as the Nature-CNN policies, so it is a
drop-in via policy kinds "impala_ac" / "impala_ac_multihead".
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from crl.policies.base import Policy
from crl.policies.cnn_ac import _actor_head, _critic_head, _ortho


class _ResidualBlock(nn.Module):
    """ReLU -> conv3x3 -> ReLU -> conv3x3, with a skip connection."""

    def __init__(self, ch: int) -> None:
        super().__init__()
        self.c1 = _ortho(nn.Conv2d(ch, ch, 3, padding=1), gain=2 ** 0.5)
        self.c2 = _ortho(nn.Conv2d(ch, ch, 3, padding=1), gain=2 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.c1(torch.relu(x))
        y = self.c2(torch.relu(y))
        return x + y


class _ConvSequence(nn.Module):
    """conv3x3 -> maxpool3x3/2 -> 2 residual blocks (one IMPALA stage)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = _ortho(nn.Conv2d(in_ch, out_ch, 3, padding=1), gain=2 ** 0.5)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res1 = _ResidualBlock(out_ch)
        self.res2 = _ResidualBlock(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.conv(x))
        return self.res2(self.res1(x))


class _ImpalaTrunk(nn.Module):
    """IMPALA residual body -> ``[B, hidden_size]`` features (task-cond optional)."""

    def __init__(self, obs_shape: tuple[int, int, int], hidden_size: int,
                 num_tasks: int, task_conditioned: bool,
                 channels: tuple[int, ...] = (32, 64, 64)) -> None:
        super().__init__()
        c, h, w = obs_shape
        self.task_conditioned = task_conditioned
        self.num_tasks = num_tasks
        seqs = []
        in_ch = c
        for out_ch in channels:
            seqs.append(_ConvSequence(in_ch, out_ch))
            in_ch = out_ch
        self.convs = nn.Sequential(*seqs)
        with torch.no_grad():
            conv_out = self.convs(torch.zeros(1, c, h, w)).flatten(1).shape[1]
        fc_in = conv_out + (num_tasks if task_conditioned else 0)
        self.fc = _ortho(nn.Linear(fc_in, hidden_size), gain=2 ** 0.5)

    def forward(self, obs: torch.Tensor, task_id: int) -> torch.Tensor:
        x = obs.float() / 255.0
        x = torch.relu(self.convs(x)).flatten(start_dim=1)
        if self.task_conditioned:
            oh = torch.zeros(x.shape[0], self.num_tasks, device=x.device)
            oh[:, task_id] = 1.0
            x = torch.cat([x, oh], dim=1)
        return torch.relu(self.fc(x))


class ImpalaActorCriticPolicy(Policy):
    """IMPALA trunk with a single actor head + critic head (single-task expert)."""

    def __init__(self, obs_shape, num_actions, hidden_size=512, num_tasks=0,
                 task_conditioned=False, channels=(32, 64, 64)) -> None:
        super().__init__()
        self.trunk = _ImpalaTrunk(obs_shape, hidden_size, num_tasks,
                                  task_conditioned, channels)
        self.actor = _actor_head(hidden_size, num_actions)
        self.critic = _critic_head(hidden_size)

    def dist(self, obs, task_id=0) -> Categorical:
        return Categorical(logits=self.actor(self.trunk(obs, task_id)))

    def value(self, obs, task_id=0) -> torch.Tensor:
        return self.critic(self.trunk(obs, task_id)).squeeze(-1)

    def dist_value(self, obs, task_id=0):
        feats = self.trunk(obs, task_id)
        return Categorical(logits=self.actor(feats)), self.critic(feats).squeeze(-1)


class ImpalaMultiHeadActorCriticPolicy(Policy):
    """IMPALA trunk with per-task actor AND critic heads (the global model)."""

    def __init__(self, obs_shape, num_actions, hidden_size=512, num_tasks=0,
                 task_conditioned=False, channels=(32, 64, 64)) -> None:
        super().__init__()
        self.trunk = _ImpalaTrunk(obs_shape, hidden_size, num_tasks,
                                  task_conditioned, channels)
        self.actors = nn.ModuleList(
            [_actor_head(hidden_size, num_actions) for _ in range(num_tasks)])
        self.critics = nn.ModuleList(
            [_critic_head(hidden_size) for _ in range(num_tasks)])

    def dist(self, obs, task_id: int) -> Categorical:
        return Categorical(logits=self.actors[task_id](self.trunk(obs, task_id)))

    def value(self, obs, task_id: int) -> torch.Tensor:
        return self.critics[task_id](self.trunk(obs, task_id)).squeeze(-1)

    def dist_value(self, obs, task_id: int):
        feats = self.trunk(obs, task_id)
        return (Categorical(logits=self.actors[task_id](feats)),
                self.critics[task_id](feats).squeeze(-1))
