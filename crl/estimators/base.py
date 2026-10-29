"""Value-estimation interface.

Everything the trainer needs from a backend is two operations:

* ``evaluate``           -- a scalar estimate of V_i^pi (no autograd graph),
                            used for constraint values F-hat (eqs 7, 11) and
                            frozen references.
* ``surrogate_objective``-- a differentiable scalar whose gradient is an
                            estimate of grad V_i^pi (eqs 17-19 / 21-23), used
                            to assemble the primal updates (eqs 20, 24).

Swapping backends (exact DP, Monte-Carlo rollouts, and later replay-based or
frozen-reference surrogates) changes nothing in the trainer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from crl.envs.base import Task
from crl.policies.base import Policy


class ValueEstimator(ABC):
    """Backend for value estimates and policy-gradient surrogates."""

    @abstractmethod
    def evaluate(
        self, policy: Policy, task: Task, num_episodes: int | None = None
    ) -> float:
        """Scalar estimate of V_task^policy (detached from autograd).

        ``num_episodes`` overrides the default sample size where sampling is
        involved (e.g. larger once-per-phase reference batches); exact
        backends ignore it.
        """

    def evaluate_return(
        self, policy: Policy, task: Task, num_episodes: int | None = None
    ) -> float:
        """Mean undiscounted episode return (task performance / score).

        Used only for reporting when ``trainer.report_return`` is set. Defaults
        to :meth:`evaluate` (the discounted value) so backends without a
        distinct notion -- e.g. the exact DP estimator -- still work.
        """
        return self.evaluate(policy, task, num_episodes)

    @abstractmethod
    def surrogate_objective(
        self, policy: Policy, task: Task
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        """Differentiable objective, differentiable entropy term, and stats.

        The first tensor's gradient w.r.t. the policy parameters estimates
        grad V_task^policy. The second is a differentiable policy-entropy
        scalar (over visited / occupied states) for optional entropy
        regularization. Stats are plain floats safe for JSON logging and
        always include ``value`` and ``entropy``.
        """
