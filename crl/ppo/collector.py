"""Vectorized rollout collection and GAE(lambda).

A :class:`RolloutCollector` owns one task's vectorized environment bank and a
persistent ``(next_obs, next_done)`` stream, so successive ``collect`` calls form
one continuous PPO rollout (the standard streaming pattern). Each call gathers
``n_steps`` transitions per env under the *current* policy (which serves as
``pi_old``: its log-probs and value predictions are frozen into the batch), then
computes GAE(lambda) advantages and bootstrapped returns.

The env uses ``SAME_STEP`` autoreset, so masking the bootstrap on ``done`` is the
exact classic formulation -- there are no dummy autoreset transitions to filter.
Truncation is treated like termination for bootstrapping (no value bootstrap on
either); with long Atari horizons this bias is negligible and matches CleanRL.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from crl.envs.base import Task
from crl.policies.base import Policy


@dataclass
class RolloutBatch:
    """One flattened on-policy batch (``N = n_steps * n_envs`` transitions)."""

    obs: torch.Tensor  # [N, C, H, W] uint8 (policy normalizes to [0,1])
    actions: torch.Tensor  # [N] long
    logprobs: torch.Tensor  # [N] float  log pi_old(a|s)
    advantages: torch.Tensor  # [N] float  GAE
    returns: torch.Tensor  # [N] float   advantages + values (critic target)
    values: torch.Tensor  # [N] float    V_old(s)
    task_id: int
    # Diagnostics from episodes that finished during collection (clipped-reward
    # scale -- a cheap training-progress proxy, not the reported game score).
    ep_returns: list[float]
    # True when ``values``/``returns`` came from a critic OTHER than the acting
    # policy's own (a frozen specialist). The global critic must not be regressed
    # on those targets, so the trainer gates its value loss off for this stream.
    external_critic: bool = False
    # Mean DISCOUNTED RETURN from the rollout's start states. With
    # ``gae_lambda=1`` and episodes finishing inside the window this is exactly
    # ``mean(G_0)``, i.e. an on-policy Monte-Carlo estimate of V^{pi}(rho) that
    # costs nothing, since the rollout was being collected for the gradient anyway.
    #
    # It deliberately does NOT go through a critic. In ``biggrid`` the reward is
    # best-so-far proximity shaping, so a cell's value depends on how close the
    # agent has already been -- hidden state the observation does not carry. A
    # critic therefore CANNOT represent V(s) there and reads ~0.12 where the truth
    # is ~0.83. That mis-specification is harmless in a baseline, which stays
    # unbiased however wrong it is, and fatal in a value estimate. So the gradient
    # may use the critic and the constraint scalar must not.
    # Only meaningful when the rollout was collected with ``reset=True``.
    start_return: float = float("nan")


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    next_done: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE(lambda) over a ``[T, n_envs]`` rollout.

    ``dones[t]`` marks whether ``obs[t]`` began a fresh episode (post-reset), i.e.
    the CleanRL convention where ``dones[t+1]`` gates the bootstrap out of step t.
    Returns ``(advantages, returns)`` each ``[T, n_envs]``.
    """
    num_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(next_value)
    for t in reversed(range(num_steps)):
        if t == num_steps - 1:
            next_nonterminal = 1.0 - next_done
            next_values = next_value
        else:
            next_nonterminal = 1.0 - dones[t + 1]
            next_values = values[t + 1]
        delta = rewards[t] + gamma * next_values * next_nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


class RolloutCollector:
    """Persistent vectorized-env stream for one task."""

    def __init__(
        self,
        task: Task,
        n_envs: int,
        n_steps: int,
        device: torch.device,
        seed: int,
        async_mode: bool = False,
    ) -> None:
        self.task_id = task.spec.task_id
        self.gamma = task.gamma
        self.n_envs = int(n_envs)
        self.n_steps = int(n_steps)
        self.device = device
        self.venv = task.make_vector_env(self.n_envs, async_mode=async_mode)
        self._reset_seed = int(seed)
        obs, _ = self.venv.reset(seed=seed)
        self.next_obs = torch.as_tensor(obs, device=device)  # uint8 [n_envs,C,H,W]
        self.next_done = torch.zeros(self.n_envs, device=device)
        # Running clipped-return accumulator per env (for progress diagnostics).
        self._ep_ret = torch.zeros(self.n_envs)

    def close(self) -> None:
        self.venv.close()

    @torch.no_grad()
    def collect(self, policy: Policy, gae_lambda: float,
                critic: Policy | None = None,
                reset: bool = False) -> RolloutBatch:
        """Roll out ``n_steps`` under ``policy`` (as pi_old); return a batch.

        ``critic`` optionally supplies the value predictions GAE bootstraps with.
        The default (``None``) is ordinary PPO, where the acting policy values its
        own states. Passing a frozen specialist instead makes every TD error
        ``r_t + gamma V_L(s_{t+1}) - V_L(s_t)``. By the performance-difference
        lemma the average of those over the rollout is exactly ``(1-gamma)`` times
        the expert-anchored value gap, so the advantages built from them point
        along the gradient of that gap rather than along the acting policy's own
        value. The ``values``/``returns`` returned then belong to THAT critic.

        ``reset`` restarts every environment before collecting, so the batch holds
        whole episodes from the task's start distribution rather than a slice of an
        ongoing stream. Needed whenever ``start_return`` is to be read, since that
        quantity is only V^{pi}(rho) if t=0 really is a start state.
        """
        device, tid = self.device, self.task_id
        if reset:
            obs, _ = self.venv.reset(seed=self._reset_seed)
            self._reset_seed += 1          # fresh start states each time, not a replay
            self.next_obs = torch.as_tensor(obs, device=device)
            self.next_done = torch.zeros(self.n_envs, device=device)
            self._ep_ret = torch.zeros(self.n_envs)
        T, n = self.n_steps, self.n_envs
        # Obs shape/dtype are taken from the live stream, so this works for both
        # Atari (uint8 [C,H,W] frames) and flat-vector envs (float [obs_dim]).
        obs_shape = tuple(self.next_obs.shape[1:])
        obs_dtype = self.next_obs.dtype

        obs_buf = torch.empty((T, n, *obs_shape), dtype=obs_dtype, device=device)
        act_buf = torch.empty((T, n), dtype=torch.long, device=device)
        logp_buf = torch.empty((T, n), device=device)
        rew_buf = torch.empty((T, n), device=device)
        done_buf = torch.empty((T, n), device=device)
        val_buf = torch.empty((T, n), device=device)
        ep_returns: list[float] = []

        for t in range(T):
            obs_buf[t] = self.next_obs
            done_buf[t] = self.next_done
            if critic is None:
                dist, value = policy.dist_value(self.next_obs, tid)
            else:
                dist = policy.dist(self.next_obs, tid)
                value = critic.value(self.next_obs, tid)
            action = dist.sample()
            act_buf[t] = action
            logp_buf[t] = dist.log_prob(action)
            val_buf[t] = value

            action_cpu = action.to("cpu").numpy()
            obs, reward, term, trunc, _ = self.venv.step(action_cpu)
            reward_t = torch.as_tensor(reward, dtype=torch.float32)
            rew_buf[t] = reward_t.to(device)
            done_np = term | trunc
            self.next_obs = torch.as_tensor(obs, device=device)
            self.next_done = torch.as_tensor(
                done_np, dtype=torch.float32, device=device
            )
            # Track finished episodes (clipped-reward return) for logging.
            self._ep_ret += reward_t
            for i in range(n):
                if bool(done_np[i]):
                    ep_returns.append(float(self._ep_ret[i]))
                    self._ep_ret[i] = 0.0

        next_value = (policy if critic is None else critic).value(self.next_obs, tid)
        advantages, returns = compute_gae(
            rew_buf, val_buf, done_buf, next_value, self.next_done,
            self.gamma, gae_lambda,
        )
        return RolloutBatch(
            # returns is [T, n_envs]; row 0 is every env's start state, so at
            # lambda=1 with the episode closing inside the window this is mean(G_0).
            start_return=float(returns[0].mean()),
            obs=obs_buf.reshape(T * n, *obs_shape),
            actions=act_buf.reshape(T * n),
            logprobs=logp_buf.reshape(T * n),
            advantages=advantages.reshape(T * n),
            returns=returns.reshape(T * n),
            values=val_buf.reshape(T * n),
            task_id=tid,
            ep_returns=ep_returns,
            external_critic=critic is not None,
        )
