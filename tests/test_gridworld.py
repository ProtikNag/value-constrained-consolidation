"""Sanity checks on the gridworld MDP tensors and their sampled environment."""

import numpy as np
import torch

from crl.config import EnvConfig
from crl.envs import make_family


def _family(**overrides):
    params = {"size": 4, "slip": 0.2, "gamma": 0.9, "max_steps": 50}
    params.update(overrides)
    cfg = EnvConfig(family="gridworld", params=params,
                    tasks=[{"goal": [0, 3]}, {"goal": [3, 0]}])
    return make_family(cfg)


def test_transition_tensor_is_stochastic():
    family = _family()
    for task in family.tasks:
        row_sums = task.transition.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)


def test_goal_is_absorbing_with_zero_reward():
    family = _family()
    task = family.tasks[0]
    goal = 0 * 4 + 3
    assert torch.allclose(task.transition[goal, :, goal],
                          torch.ones(family.num_actions))
    assert torch.allclose(task.reward[goal], torch.zeros(family.num_actions))


def test_initial_dist_excludes_goal():
    family = _family()
    task = family.tasks[0]
    goal = 0 * 4 + 3
    assert task.initial_dist[goal] == 0.0
    assert abs(float(task.initial_dist.sum()) - 1.0) < 1e-6


def test_env_rewards_only_on_goal_entry():
    family = _family()
    task = family.tasks[0]
    env = task.make_env()
    rng = np.random.default_rng(0)
    obs, _ = env.reset(seed=0)
    for _ in range(200):
        action = int(rng.integers(4))
        obs, reward, terminated, truncated, _ = env.step(action)
        if reward != 0.0:
            # A nonzero reward must coincide with reaching the goal state.
            assert terminated
        if terminated or truncated:
            obs, _ = env.reset()
