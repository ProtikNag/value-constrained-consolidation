"""Policy interface shared by tabular and neural policies."""

from __future__ import annotations

import copy
from abc import abstractmethod

import torch
import torch.nn as nn
from torch.distributions import Categorical


class Policy(nn.Module):
    """Stochastic discrete-action policy pi(a | s).

    ``task_id`` is threaded through so task-conditioned policies can append
    a task indicator; unconditioned policies ignore it.
    """

    @abstractmethod
    def dist(self, obs: torch.Tensor, task_id: int) -> Categorical:
        """Action distribution for a batch of observations ``[B, obs_dim]``."""

    @torch.no_grad()
    def act(self, obs: torch.Tensor, task_id: int) -> int:
        """Sample one action for a single observation ``[obs_dim]``."""
        return int(self.dist(obs.unsqueeze(0), task_id).sample().item())


def clone_policy(policy: Policy, trainable: bool) -> Policy:
    """Deep-copy a policy; frozen copies serve as the phase references
    (``theta^(0) = phi_bar``, eq 2, and the frozen counterparts in eqs 8/12).
    """
    clone = copy.deepcopy(policy)
    for param in clone.parameters():
        param.requires_grad_(trainable)
    if not trainable:
        clone.eval()
    return clone
