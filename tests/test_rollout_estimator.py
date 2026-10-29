"""Monte-Carlo estimator agreement with the exact backend on the gridworld.

These are statistical tests with fixed seeds: values must agree within
sampling tolerance, and the REINFORCE gradient must point in the same
direction as the exact policy gradient.
"""

import torch

from crl.config import EnvConfig, PolicyConfig
from crl.envs import make_family
from crl.estimators.exact import ExactEstimator
from crl.estimators.rollout import MonteCarloEstimator
from crl.policies import make_policy
from crl.policies.tabular import TabularPolicy
from crl.seeding import set_seed


def _setup():
    set_seed(1)
    cfg = EnvConfig(
        family="gridworld",
        params={"size": 4, "slip": 0.1, "gamma": 0.9, "max_steps": 60},
        tasks=[{"goal": [3, 3]}],
    )
    family = make_family(cfg)
    policy = TabularPolicy(family.obs_dim, family.num_actions)
    with torch.no_grad():
        policy.logits.add_(0.3 * torch.randn_like(policy.logits))
    return family.tasks[0], policy


def test_monte_carlo_value_matches_exact():
    task, policy = _setup()
    exact_value = ExactEstimator().evaluate(policy, task)
    mc_value = MonteCarloEstimator(episodes_per_eval=600).evaluate(policy, task)
    # Value scale here is ~0.2-0.5; 0.05 absolute is ~3 sigma at 600 episodes.
    assert abs(mc_value - exact_value) < 0.05


def test_monte_carlo_gradient_direction_matches_exact():
    task, policy = _setup()

    objective, _, _ = ExactEstimator().surrogate_objective(policy, task)
    objective.backward()
    exact_grad = policy.logits.grad.flatten().clone()
    policy.logits.grad = None

    estimator = MonteCarloEstimator(
        episodes_per_grad=600, baseline="batch_mean", time_discount_weighting=True
    )
    mc_objective, _, _ = estimator.surrogate_objective(policy, task)
    mc_objective.backward()
    mc_grad = policy.logits.grad.flatten().clone()

    cosine = torch.nn.functional.cosine_similarity(exact_grad, mc_grad, dim=0)
    assert float(cosine) > 0.5, f"gradient cosine too low: {float(cosine):.3f}"


def test_vector_rollout_matches_gym_env_semantics():
    """The batched fast path samples the same MDP as the gym TabularEnv:
    reward only on entering the goal, and terminated iff the goal was reached."""
    set_seed(2)
    cfg = EnvConfig(
        family="gridworld",
        params={"size": 4, "slip": 0.1, "gamma": 0.9, "max_steps": 40,
                "goal_reward": 1.0, "step_penalty": 0.0},
        tasks=[{"goal": [3, 3]}],
    )
    task = make_family(cfg).tasks[0]
    policy = TabularPolicy(task.transition.shape[0], 4)
    episodes = task.vector_rollout(policy, num_episodes=200)

    assert len(episodes) == 200
    for ep in episodes:
        assert len(ep.rewards) <= 40  # never exceeds the horizon
        # Reward is 1.0 exactly once (goal entry) for a terminated episode,
        # and 0.0 everywhere else; a truncated episode collects no reward.
        total = float(ep.rewards.sum())
        assert total in (0.0, 1.0)
        assert ep.terminated == (total == 1.0)
        if ep.terminated:
            assert float(ep.rewards[-1]) == 1.0  # reward lands on the final step


def test_vector_rollout_value_matches_exact_multihead():
    """Vectorized Monte-Carlo agrees with exact DP for a multi-head neural
    policy across every task in a 3-task family (unbiasedness at scale)."""
    set_seed(4)
    cfg = EnvConfig(
        family="gridworld",
        params={"size": 5, "slip": 0.1, "gamma": 0.95, "max_steps": 100},
        tasks=[{"goal": [0, 4]}, {"goal": [4, 4]}, {"goal": [4, 0]}],
    )
    family = make_family(cfg)
    policy = make_policy(
        PolicyConfig(kind="multihead", hidden_sizes=[64, 64], task_conditioned=True),
        family,
    )
    exact = ExactEstimator()
    mc = MonteCarloEstimator(episodes_per_eval=800)
    for task in family.tasks:
        assert abs(mc.evaluate(policy, task) - exact.evaluate(policy, task)) < 0.05
