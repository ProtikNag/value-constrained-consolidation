"""Softmax tabular policy over one-hot state observations."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from crl.policies.base import Policy


class TabularPolicy(Policy):
    """One logit per (state, action); pi(a|s) = softmax over the state's row.

    Observations are one-hot state indicators, so ``obs @ logits`` selects
    the logit rows of the visited states while staying batched and
    differentiable. Space O(S * A).
    """

    def __init__(self, num_states: int, num_actions: int) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_states, num_actions))

    def dist(self, obs: torch.Tensor, task_id: int) -> Categorical:
        del task_id  # a single shared table; conditioning not supported here
        return Categorical(logits=obs @ self.logits)
