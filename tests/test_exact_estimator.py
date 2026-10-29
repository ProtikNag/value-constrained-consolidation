"""Correctness of the exact DP estimator: values and autodiff gradients."""

import torch

from crl.config import EnvConfig
from crl.envs import make_family
from crl.estimators.exact import ExactEstimator
from crl.policies.tabular import TabularPolicy
from crl.seeding import set_seed


def _setup():
    set_seed(0)
    cfg = EnvConfig(
        family="gridworld",
        params={"size": 4, "slip": 0.1, "gamma": 0.9, "max_steps": 50},
        tasks=[{"goal": [3, 3]}],
    )
    family = make_family(cfg)
    policy = TabularPolicy(family.obs_dim, family.num_actions)
    with torch.no_grad():
        policy.logits.add_(0.5 * torch.randn_like(policy.logits))
    return family.tasks[0], policy


def test_value_matches_power_iteration():
    """Linear-solve value equals long Bellman backup iteration."""
    task, policy = _setup()
    estimator = ExactEstimator()
    value_solve = estimator.evaluate(policy, task)

    with torch.no_grad():
        probs = policy.dist(torch.eye(task.transition.shape[0]), 0).probs
        transition_pi = torch.einsum("sa,saj->sj", probs, task.transition)
        reward_pi = (probs * task.reward).sum(-1)
        value = torch.zeros_like(reward_pi)
        for _ in range(2000):
            value = reward_pi + task.gamma * (transition_pi @ value)
        value_iter = float(task.initial_dist @ value)

    assert abs(value_solve - value_iter) < 1e-5


def test_gradient_matches_finite_differences():
    task, policy = _setup()
    # Float64 throughout: float32 round-off in the linear solve (~2e-4 at
    # h=1e-4) would otherwise dominate the comparison.
    task.transition = task.transition.double()
    task.reward = task.reward.double()
    task.initial_dist = task.initial_dist.double()
    policy = policy.double()
    estimator = ExactEstimator()

    objective, _, _ = estimator.surrogate_objective(policy, task)
    objective.backward()
    autodiff_grad = policy.logits.grad.clone()

    # Central finite differences on a handful of coordinates.
    step = 1e-5
    for state, action in [(0, 0), (5, 2), (10, 3), (14, 1)]:
        with torch.no_grad():
            policy.logits[state, action] += step
        value_plus = estimator.evaluate(policy, task)
        with torch.no_grad():
            policy.logits[state, action] -= 2 * step
        value_minus = estimator.evaluate(policy, task)
        with torch.no_grad():
            policy.logits[state, action] += step
        numeric = (value_plus - value_minus) / (2 * step)
        assert abs(numeric - float(autodiff_grad[state, action])) < 1e-8
