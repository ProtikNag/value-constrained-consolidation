"""Exact dynamic-programming estimator for tabular tasks.

Computes V^pi by solving the Bellman linear system, so both values and
policy gradients are exact (autograd differentiates through the solve).
This is the zero-variance backend used to verify that the update rules
(eqs 20, 24) and the dual dynamics behave as derived, before any sampling
noise enters the picture.

Note on horizons: the derivation uses finite-T trajectories while this
backend evaluates the discounted infinite-horizon value with an absorbing
goal. For gamma^T << 1 the two coincide up to negligible truncation error;
``tests/test_rollout_estimator.py`` checks the agreement empirically.
"""

from __future__ import annotations

import torch

from crl.envs.base import TabularTask, Task
from crl.estimators.base import ValueEstimator
from crl.policies.base import Policy


class ExactEstimator(ValueEstimator):
    """Closed-form policy evaluation, O(S^3) per solve (trivial at S ~ 25)."""

    def _require_tabular(self, task: Task) -> TabularTask:
        if not isinstance(task, TabularTask):
            raise TypeError(
                f"ExactEstimator needs a TabularTask, got {type(task).__name__}; "
                "use the monte_carlo estimator for non-tabular families."
            )
        return task

    def _value_vector(self, policy: Policy, task: TabularTask) -> torch.Tensor:
        """Solve (I - gamma * P_pi) V = r_pi; differentiable w.r.t. policy.

        Works in whatever dtype the task tensors carry (float64 task tensors
        plus a double() policy give reference-grade gradients for tests).
        """
        num_states = task.transition.shape[0]
        dtype = task.transition.dtype
        all_states = torch.eye(num_states, dtype=dtype)
        action_probs = policy.dist(all_states, task.spec.task_id).probs.to(dtype)
        transition_pi = torch.einsum("sa,saj->sj", action_probs, task.transition)
        reward_pi = (action_probs * task.reward).sum(dim=-1)
        system = torch.eye(num_states, dtype=dtype) - task.gamma * transition_pi
        return torch.linalg.solve(system, reward_pi)

    def evaluate(
        self, policy: Policy, task: Task, num_episodes: int | None = None
    ) -> float:
        del num_episodes  # exact: sample size is meaningless
        tab = self._require_tabular(task)
        with torch.no_grad():
            value = self._value_vector(policy, tab)
            return float(tab.initial_dist @ value)

    def surrogate_objective(
        self, policy: Policy, task: Task
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        tab = self._require_tabular(task)
        value = self._value_vector(policy, tab)
        objective = tab.initial_dist @ value

        # Entropy term: policy entropy weighted by the exact discounted state
        # occupancy. The occupancy weight is detached (mirroring the sampled
        # backend, which does not differentiate through visitation either).
        num_states = tab.transition.shape[0]
        dtype = tab.transition.dtype
        all_states = torch.eye(num_states, dtype=dtype)
        dist = policy.dist(all_states, tab.spec.task_id)
        with torch.no_grad():
            transition_pi = torch.einsum(
                "sa,saj->sj", dist.probs.to(dtype), tab.transition
            )
            system = torch.eye(num_states, dtype=dtype) - tab.gamma * transition_pi
            occupancy = (1.0 - tab.gamma) * torch.linalg.solve(
                system.T, tab.initial_dist
            )
        entropy_term = occupancy.to(dist.probs.dtype) @ dist.entropy()

        stats = {
            "value": float(objective.detach()),
            "entropy": float(entropy_term.detach()),
        }
        return objective, entropy_term, stats
