"""Checks for the separate actor/critic policy and Bellman-residual consolidation.

These cover what would fail silently rather than loudly: the actor and critic
sharing parameters after all, the collector ignoring the critic it was handed,
and above all the telescoping identity that the whole formulation rests on --
that GAE at lambda=1 over whole episodes really does return ``G_t - V(s_t)``,
so that ``-mean(A_0)`` is the value gap exactly rather than approximately.
"""

from __future__ import annotations

import pytest
import torch

from crl.policies.mlp import MLPActorCriticPolicy, MLPSeparateActorCriticPolicy


def _sep(obs_dim: int = 12, num_actions: int = 4, num_tasks: int = 3):
    return MLPSeparateActorCriticPolicy(
        obs_dim=obs_dim, num_actions=num_actions, hidden_sizes=[16],
        num_tasks=num_tasks, task_conditioned=True,
    )


def test_actor_and_critic_share_no_parameters() -> None:
    """The whole point of mlp_ac_sep: disjoint parameter sets."""
    policy = _sep()
    actor_ids = {id(p) for p in policy.actor_parameters()}
    critic_ids = {id(p) for p in policy.critic_parameters()}
    assert actor_ids and critic_ids
    assert actor_ids.isdisjoint(critic_ids)
    # and together they account for every parameter
    assert actor_ids | critic_ids == {id(p) for p in policy.parameters()}


def test_shared_trunk_policy_does_couple_them() -> None:
    """Contrast case: mlp_ac's value loss DOES reach actor-facing parameters."""
    policy = MLPActorCriticPolicy(
        obs_dim=12, num_actions=4, hidden_sizes=[16], num_tasks=3,
        task_conditioned=True)
    obs = torch.randn(8, 12)
    policy.value(obs, 0).sum().backward()
    assert policy.trunk[0].weight.grad is not None
    assert policy.trunk[0].weight.grad.abs().sum() > 0


def test_value_loss_leaves_actor_untouched() -> None:
    """With separate trunks a critic update cannot move the policy."""
    policy = _sep()
    obs = torch.randn(8, 12)
    policy.value(obs, 0).sum().backward()
    for param in policy.actor_parameters():
        assert param.grad is None or param.grad.abs().sum() == 0
    grads = [p.grad for p in policy.critic_parameters() if p.grad is not None]
    assert grads and sum(float(g.abs().sum()) for g in grads) > 0


def test_policy_gradient_leaves_critic_untouched() -> None:
    """...and the converse."""
    policy = _sep()
    obs = torch.randn(8, 12)
    policy.dist(obs, 0).logits.sum().backward()
    for param in policy.critic_parameters():
        assert param.grad is None or param.grad.abs().sum() == 0


def test_dist_value_matches_the_separate_calls() -> None:
    """The fused accessor must agree with the individual ones."""
    policy = _sep()
    obs = torch.randn(8, 12)
    with torch.no_grad():
        dist, value = policy.dist_value(obs, 1)
        assert torch.allclose(value, policy.value(obs, 1))
        assert torch.allclose(dist.logits, policy.dist(obs, 1).logits)


def test_collector_uses_the_critic_it_is_handed() -> None:
    """An external critic must actually supply the values, and flag the batch."""
    from crl.ppo.collector import RolloutCollector
    from crl.config import EnvConfig
    from crl.envs import make_family

    family = make_family(EnvConfig(
        family="biggrid",
        params={"size": 6, "gamma": 0.99, "max_steps": 20, "slip": 0.0,
                "goal_in_obs": False, "obstacle_seed": 1},
        tasks=[{"goal": [5, 5], "slip": 0.0, "obstacle_density": 0.0,
                "threshold": 0.85}],
    ))
    task = family.tasks[0]
    policy = _sep(obs_dim=family.obs_dim, num_actions=family.num_actions,
                  num_tasks=len(family))
    expert = _sep(obs_dim=family.obs_dim, num_actions=family.num_actions,
                  num_tasks=len(family))
    # Make the expert's critic unmistakably different from the policy's.
    with torch.no_grad():
        expert.critic.bias.fill_(50.0)

    device = torch.device("cpu")
    own = RolloutCollector(task, 2, 8, device, seed=0).collect(policy, 0.95)
    ext = RolloutCollector(task, 2, 8, device, seed=0).collect(
        policy, 0.95, critic=expert)

    assert own.external_critic is False
    assert ext.external_critic is True
    assert float(ext.values.mean()) > float(own.values.mean()) + 10.0
    assert not torch.allclose(ext.advantages, own.advantages)


def _tiny_family(size: int = 6):
    from crl.config import EnvConfig
    from crl.envs import make_family
    return make_family(EnvConfig(
        family="biggrid",
        params={"size": size, "gamma": 0.99, "max_steps": 25, "slip": 0.0,
                "goal_in_obs": False, "obstacle_seed": 1},
        tasks=[{"goal": [size - 1, size - 1], "slip": 0.0,
                "obstacle_density": 0.0, "threshold": 0.85}],
    ))


def test_lambda_one_advantage_is_return_minus_baseline() -> None:
    """The identity the whole formulation rests on.

    At lambda=1, with the bootstrap masked at episode ends, GAE must return
    exactly ``G_t - V(s_t)`` where ``G_t`` is the discounted reward to the end of
    that episode. If it does not, ``-mean(A_0)`` is not the value gap and the
    constraint is measuring something else.

    Checked against an independently computed ``G_t`` on a synthetic two-episode
    rollout with a deliberately WRONG baseline, since the claim is that the
    baseline's error cannot leak into the advantage.
    """
    from crl.ppo.collector import compute_gae

    gamma, T, n = 0.9, 6, 1
    rewards = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    # dones[t] marks that obs[t] STARTED a fresh episode, so the episode break
    # sits between t=2 and t=3 (the collector's CleanRL convention).
    dones = torch.tensor([[0.0], [0.0], [0.0], [1.0], [0.0], [0.0]])
    values = torch.tensor([[10.0], [-7.0], [0.5], [99.0], [-1.0], [3.0]])  # nonsense
    next_value = torch.tensor([42.0])          # masked out: next_done = 1
    next_done = torch.tensor([1.0])

    adv, ret = compute_gae(rewards, values, dones, next_value, next_done,
                           gamma, gae_lambda=1.0)

    # Independent ground truth: discounted reward to the end of each episode.
    # Episode A covers t=0..2, episode B covers t=3..5.
    G = torch.zeros(T, n)
    for lo, hi in ((0, 3), (3, 6)):
        acc = 0.0
        for t in reversed(range(lo, hi)):
            acc = float(rewards[t, 0]) + gamma * acc
            G[t, 0] = acc

    assert torch.allclose(adv, G - values, atol=1e-5), (
        f"lambda=1 advantage is not G_t - V(s_t)\n"
        f"got      {adv.flatten().tolist()}\n"
        f"expected {(G - values).flatten().tolist()}")
    # and returns = advantages + values must recover G exactly
    assert torch.allclose(ret, G, atol=1e-5)


def test_lambda_one_advantage_ignores_the_baselines_error() -> None:
    """Any baseline shifts the advantage by exactly its own value and no more.

    This is why a frozen specialist's critic is safe at lambda=1 and unsafe as a
    bootstrap: its error enters additively and differentiates away, rather than
    propagating through the return.
    """
    from crl.ppo.collector import compute_gae

    gamma, T, n = 0.99, 5, 2
    rewards = torch.randn(T, n)
    dones = torch.zeros(T, n)
    next_done = torch.ones(n)                  # episode closes inside the window
    next_value = torch.zeros(n)
    good = torch.randn(T, n)
    bad = good + 17.0                          # wildly miscalibrated baseline

    a_good, _ = compute_gae(rewards, good, dones, next_value, next_done, gamma, 1.0)
    a_bad, _ = compute_gae(rewards, bad, dones, next_value, next_done, gamma, 1.0)

    assert torch.allclose(a_good - a_bad, torch.full((T, n), 17.0), atol=1e-4)


def test_lambda_below_one_needs_a_separate_scalar() -> None:
    """At lambda<1 the specialist bootstraps, so mean(G_0) stops being the gap.

    Reading the constraint off the rollout anyway would silently measure
    ``(1-gamma^H)`` of the shortfall, which is 26% at H=30 and 1% at H=1. That
    is the bug that kept mu pinned at zero for the whole project, so the
    combination must be refused rather than quietly under-reported.
    """
    from crl.config import PPOConfig
    from crl.ppo.trainer import GlobalTrainer

    cfg = PPOConfig()
    cfg.value_mode = "residual"
    cfg.global_gae_lambda = 0.95
    cfg.residual_scalar = "rollout"
    trainer = GlobalTrainer.__new__(GlobalTrainer)
    trainer.ppo = cfg
    with pytest.raises(ValueError, match="no longer the value gap"):
        GlobalTrainer.train(
            trainer, global_policy=_sep(), task_k=None, past_tasks=[],
            ref_current=0.0, mu_ctrl=None, omega=[], eps=1e-4,
            num_iters=1, seed=0, current_task=1, local_policy=_sep())


def test_lambda_below_one_is_allowed_with_an_mc_scalar() -> None:
    """...but a bootstrapped GRADIENT with an honest constraint is legitimate."""
    from crl.config import PPOConfig
    from crl.ppo.trainer import GlobalTrainer

    cfg = PPOConfig()
    cfg.value_mode = "residual"
    cfg.global_gae_lambda = 0.95
    cfg.residual_scalar = "mc"
    trainer = GlobalTrainer.__new__(GlobalTrainer)
    trainer.ppo = cfg
    # Must get PAST the lambda guard; it then fails on the missing specialist.
    with pytest.raises(ValueError, match="needs the frozen specialist"):
        GlobalTrainer.train(
            trainer, global_policy=_sep(), task_k=None, past_tasks=[],
            ref_current=0.0, mu_ctrl=None, omega=[], eps=1e-4,
            num_iters=1, seed=0, current_task=1, local_policy=None)


def test_residual_requires_one_expert_per_past_task() -> None:
    """A miscounted expert list must fail loudly, not score the wrong task."""
    from crl.config import PPOConfig
    from crl.ppo.trainer import GlobalTrainer

    cfg = PPOConfig()
    cfg.value_mode = "residual"
    cfg.global_gae_lambda = 1.0
    trainer = GlobalTrainer.__new__(GlobalTrainer)
    trainer.ppo = cfg
    with pytest.raises(ValueError, match="one frozen expert per"):
        GlobalTrainer.train(
            trainer, global_policy=_sep(), task_k=None, past_tasks=[None, None],
            ref_current=0.0, mu_ctrl=None, omega=[0.5, 0.5], eps=1e-4,
            num_iters=1, seed=0, current_task=3,
            local_policy=_sep(), past_experts=[_sep()])


def test_residual_requires_a_specialist() -> None:
    from crl.config import PPOConfig
    from crl.ppo.trainer import GlobalTrainer

    cfg = PPOConfig()
    cfg.value_mode = "residual"
    cfg.global_gae_lambda = 1.0
    trainer = GlobalTrainer.__new__(GlobalTrainer)
    trainer.ppo = cfg
    with pytest.raises(ValueError, match="needs the frozen specialist"):
        GlobalTrainer.train(
            trainer, global_policy=_sep(), task_k=None, past_tasks=[],
            ref_current=0.0, mu_ctrl=None, omega=[], eps=1e-4,
            num_iters=1, seed=0, current_task=1, local_policy=None)


def _multi(obs_dim: int = 12, num_actions: int = 4, num_tasks: int = 3):
    from crl.policies.mlp import MLPSharedActorMultiCriticPolicy
    return MLPSharedActorMultiCriticPolicy(
        obs_dim=obs_dim, num_actions=num_actions, hidden_sizes=[16],
        num_tasks=num_tasks, task_conditioned=True)


def test_per_task_critics_do_not_interfere() -> None:
    """The whole point: training task 0's critic must not move task 1's."""
    policy = _multi()
    obs = torch.randn(8, 12)
    before = [p.clone() for p in policy.critic_trunks[1].parameters()]
    policy.value(obs, 0).sum().backward()
    # only task 0's critic has gradient
    assert all(p.grad is not None and p.grad.abs().sum() > 0
               for p in policy.critic_trunks[0].parameters())
    assert all(p.grad is None or p.grad.abs().sum() == 0
               for p in policy.critic_trunks[1].parameters())
    assert all(torch.equal(a, b)
               for a, b in zip(before, policy.critic_trunks[1].parameters()))


def test_multicritic_actor_is_shared_across_tasks() -> None:
    """One actor head for every task -- that is the headline architecture."""
    policy = _multi()
    obs = torch.randn(8, 12)
    policy.dist(obs, 0).logits.sum().backward()
    # the actor took gradient...
    assert all(p.grad is not None and p.grad.abs().sum() > 0
               for p in policy.actor_parameters())
    # ...and NO critic did
    assert all(p.grad is None or p.grad.abs().sum() == 0
               for p in policy.critic_parameters())
    # same head serves every task: task_id only enters via the one-hot input
    assert len(list(policy.actor.parameters())) == 2
