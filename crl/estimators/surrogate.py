"""Frozen-reference surrogate estimator (NOT YET IMPLEMENTED — by design).

Intended approach (CPO-style, Achiam et al. 2017): at the start of each
phase collect one batch per past task under the frozen reference policy,
fit advantages once, and evaluate the constraint as an expectation over
that fixed batch in which the learner enters only through likelihood
ratios. Exact at the phase start (theta^(0) = phi_bar, eq 2), valid nearby
under a trust region / clipping.

Whether to build this, the replay + truncated-importance-weight (V-trace)
variant, or per-task fitted critics is an open project decision with
paper-level consequences and is not settled here. The interface below fixes the plug-in point so the
trainer needs no changes when a choice lands.
"""

from __future__ import annotations

import torch

from crl.envs.base import Task
from crl.estimators.base import ValueEstimator
from crl.policies.base import Policy

_DECISION_NOTE = (
    "FrozenReferenceSurrogate is a placeholder: the past-task evaluation "
    "strategy (CPO surrogate vs importance-weighted replay vs fitted critics) "
    "is an open decision. The naive `monte_carlo` backend is "
    "the current default and is adequate for the gridworld/CartPole tiers."
)


class FrozenReferenceSurrogate(ValueEstimator):
    """Placeholder that documents the intended cheap constraint estimator."""

    def evaluate(
        self, policy: Policy, task: Task, num_episodes: int | None = None
    ) -> float:
        raise NotImplementedError(_DECISION_NOTE)

    def surrogate_objective(
        self, policy: Policy, task: Task
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        raise NotImplementedError(_DECISION_NOTE)
