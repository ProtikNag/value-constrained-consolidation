"""CReLU MLP actor-critic: shared-head, fixed-capacity (loss-of-plasticity baseline).

Shared-head counterpart of :class:`crl.policies.mlp.MLPActorCriticPolicy` with the
Tanh trunk swapped for the Concatenated-ReLU (CReLU) activation of Abbas et al.
(2023, "Loss of Plasticity in Continual Deep RL"). CReLU is the loss-of-plasticity
mitigation baseline for the GridWorld fixed-capacity / no-growth comparison: same
shared trunk + a SINGLE shared actor head + single critic head, optional task
one-hot conditioning, NO per-task heads and NO capacity growth across tasks --
trained with the existing ``method=finetune`` (naive sequential).

CReLU(x) = concat(ReLU(x), ReLU(-x)) doubles the activation width of every layer it
follows while keeping that layer's parameter count unchanged. As the paper notes
(Sec. "invariant input dimension"), this doubling then doubles the input width of
the *following* layer -- so each subsequent Linear takes ``2 * prev_width`` inputs,
and the actor/critic heads take ``2 * last_hidden`` inputs. That plumbing is the
only structural difference from the Tanh trunk.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from crl.policies.base import Policy


class CReLU(nn.Module):
    """Concatenated ReLU: ``[ReLU(x), ReLU(-x)]`` along the feature dim (doubles width)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat((F.relu(x), F.relu(-x)), dim=-1)


def _crelu_trunk(input_dim: int, hidden_sizes: list[int]) -> tuple[nn.Sequential, int]:
    """CReLU MLP trunk; returns the module and its output width.

    Each ``Linear(last, width)`` is followed by CReLU, so its output width is
    ``2 * width`` and the next layer's input width ``last`` becomes ``2 * width``.
    The returned width is the CReLU-doubled width of the final hidden layer, which
    is what the actor/critic heads must consume.
    """
    layers: list[nn.Module] = []
    last = input_dim
    for width in hidden_sizes:
        layers += [nn.Linear(last, width), CReLU()]
        last = 2 * width  # CReLU doubling flows into the next layer's input dim
    return nn.Sequential(*layers), last


def _init_head(head: nn.Linear) -> None:
    """Small final layer keeps the initial policy near-uniform (low-variance
    early gradients)."""
    nn.init.orthogonal_(head.weight, gain=0.01)
    nn.init.zeros_(head.bias)


class MLPActorCriticCReLUPolicy(Policy):
    """CReLU MLP actor-critic with a SINGLE shared actor head + single critic head.

    Fixed-capacity shared-head regime: ALL weights (CReLU trunk + the one
    actor/critic head) are shared across tasks -- no per-task heads, no growth --
    so forgetting lives in the shared parameters and naive fine-tuning must
    overwrite them. Meant to be used with ``task_conditioned=True`` (task one-hot
    appended to the input) or a goal-in-obs family so the single critic stays
    well-posed while all tasks contend for one shared head.
    """

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        hidden_sizes: list[int],
        num_tasks: int = 0,
        task_conditioned: bool = False,
    ) -> None:
        super().__init__()
        if task_conditioned and num_tasks <= 0:
            raise ValueError("task_conditioned=True requires num_tasks > 0.")
        self.task_conditioned = task_conditioned
        self.num_tasks = num_tasks
        input_dim = obs_dim + (num_tasks if task_conditioned else 0)
        self.trunk, last = _crelu_trunk(input_dim, hidden_sizes)
        self.actor = nn.Linear(last, num_actions)
        self.critic = nn.Linear(last, 1)
        _init_head(self.actor)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def _augment(self, obs: torch.Tensor, task_id: int) -> torch.Tensor:
        if not self.task_conditioned:
            return obs
        one_hot = torch.zeros(obs.shape[0], self.num_tasks, device=obs.device,
                              dtype=obs.dtype)
        one_hot[:, task_id] = 1.0
        return torch.cat([obs, one_hot], dim=-1)

    def dist(self, obs: torch.Tensor, task_id: int) -> Categorical:
        return Categorical(logits=self.actor(self.trunk(self._augment(obs, task_id))))

    def value(self, obs: torch.Tensor, task_id: int) -> torch.Tensor:
        return self.critic(self.trunk(self._augment(obs, task_id))).squeeze(-1)

    def dist_value(self, obs: torch.Tensor, task_id: int):
        feats = self.trunk(self._augment(obs, task_id))
        return Categorical(logits=self.actor(feats)), self.critic(feats).squeeze(-1)
