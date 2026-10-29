"""Unit tests for the PPO Atari backend (fast, CPU-only).

Covers the pieces whose correctness is not obvious by inspection: GAE(lambda)
against a hand-computed rollout (with and without a done mask), the actor-critic
policy interfaces (single- and multi-head), and one end-to-end PPO iteration
that actually reduces the value loss on a trivial environment.
"""

from __future__ import annotations

import torch

from crl.ppo.collector import compute_gae


def test_gae_no_done_matches_hand_computation():
    gamma, lam = 0.9, 0.5
    values = torch.tensor([[1.0], [2.0], [3.0]])
    rewards = torch.tensor([[1.0], [1.0], [1.0]])
    dones = torch.zeros(3, 1)
    next_value = torch.tensor([4.0])
    next_done = torch.tensor([0.0])
    adv, ret = compute_gae(rewards, values, dones, next_value, next_done, gamma, lam)
    # t2: 1 + 0.9*4 - 3 = 1.6
    # t1: (1 + 0.9*3 - 2) + 0.45*1.6 = 1.7 + 0.72 = 2.42
    # t0: (1 + 0.9*2 - 1) + 0.45*2.42 = 1.8 + 1.089 = 2.889
    assert torch.allclose(adv, torch.tensor([[2.889], [2.42], [1.6]]), atol=1e-5)
    assert torch.allclose(ret, adv + values, atol=1e-6)


def test_gae_done_masks_bootstrap():
    # dones[t] flags that obs[t] starts a fresh episode; it gates the bootstrap
    # out of step t-1. With dones[1]=1, step 0 must not bootstrap from step 1.
    gamma, lam = 0.99, 0.95
    values = torch.tensor([[5.0], [7.0]])
    rewards = torch.tensor([[1.0], [1.0]])
    dones = torch.tensor([[0.0], [1.0]])
    next_value = torch.tensor([9.0])
    next_done = torch.tensor([0.0])
    adv, _ = compute_gae(rewards, values, dones, next_value, next_done, gamma, lam)
    # t1: 1 + 0.99*9 - 7 = 2.91
    # t0: nextnonterminal = 1 - dones[1] = 0 -> delta = 1 - 5 = -4, no bootstrap
    assert torch.allclose(adv, torch.tensor([[-4.0], [2.91]]), atol=1e-5)


def _obs(n):
    return torch.randint(0, 256, (n, 4, 84, 84), dtype=torch.uint8)


def test_single_head_policy_shapes():
    from crl.policies.cnn_ac import AtariActorCriticPolicy

    pol = AtariActorCriticPolicy((4, 84, 84), num_actions=18, num_tasks=3)
    dist, value = pol.dist_value(_obs(5), task_id=1)
    assert dist.logits.shape == (5, 18)
    assert value.shape == (5,)
    assert pol.value(_obs(2), 0).shape == (2,)


def test_multi_head_routes_by_task():
    from crl.policies.cnn_ac import AtariMultiHeadActorCriticPolicy

    pol = AtariMultiHeadActorCriticPolicy((4, 84, 84), num_actions=18, num_tasks=3,
                                          task_conditioned=True)
    d0, _ = pol.dist_value(_obs(4), 0)
    d2, _ = pol.dist_value(_obs(4), 2)
    assert d0.logits.shape == (4, 18) and d2.logits.shape == (4, 18)
    # Different heads -> generally different logits for the same input.
    x = _obs(4)
    assert not torch.allclose(pol.dist(x, 0).logits, pol.dist(x, 2).logits)
    try:
        pol.dist(_obs(1), 3)
        assert False, "expected IndexError for out-of-range task_id"
    except IndexError:
        pass


def test_one_ppo_iteration_reduces_value_loss():
    """A full collect+update iteration on a trivial constant-obs env should
    drive the critic toward the observed returns (value loss decreases)."""
    import gymnasium as gym
    import numpy as np
    from gymnasium.vector import AutoresetMode, SyncVectorEnv

    from crl.config import PPOConfig
    from crl.envs.base import Task, TaskSpec
    from crl.policies.cnn_ac import AtariActorCriticPolicy
    from crl.ppo.collector import RolloutCollector
    from crl.ppo.trainer import PPOTrainer

    class _ConstEnv(gym.Env):
        def __init__(self):
            self.observation_space = gym.spaces.Box(0, 255, (4, 84, 84), np.uint8)
            self.action_space = gym.spaces.Discrete(18)
            self._t = 0

        def reset(self, *, seed=None, options=None):
            self._t = 0
            return np.zeros((4, 84, 84), np.uint8), {}

        def step(self, action):
            self._t += 1
            done = self._t >= 8
            return np.zeros((4, 84, 84), np.uint8), 1.0, done, False, {}

    class _ConstTask(Task):
        success_on_termination = False

        def make_env(self):
            return _ConstEnv()

        def make_vector_env(self, num_envs, clip_rewards=None):
            return SyncVectorEnv([_ConstEnv for _ in range(num_envs)],
                                 autoreset_mode=AutoresetMode.SAME_STEP)

    task = _ConstTask(TaskSpec(0, "const"), gamma=0.99)
    device = torch.device("cpu")
    cfg = PPOConfig(n_envs=2, n_steps=16, ppo_epochs=4, num_minibatches=2)
    policy = AtariActorCriticPolicy((4, 84, 84), 18, num_tasks=1)
    trainer = PPOTrainer(cfg, device, logger=None, log_every=1)
    optimizer = trainer._new_optimizer(policy)
    collector = RolloutCollector(task, cfg.n_envs, cfg.n_steps, device, seed=0)
    batch = collector.collect(policy, cfg.gae_lambda)
    before = float(0.5 * (policy.value(batch.obs, 0) - batch.returns).pow(2).mean())
    for _ in range(5):
        batch = collector.collect(policy, cfg.gae_lambda)
        trainer.optimize_batches(policy, optimizer, [batch], [1.0])
    batch = collector.collect(policy, cfg.gae_lambda)
    after = float(0.5 * (policy.value(batch.obs, 0) - batch.returns).pow(2).mean())
    collector.close()
    assert after < before, f"value loss did not decrease: {before} -> {after}"
