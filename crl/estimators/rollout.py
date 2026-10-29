"""On-policy Monte-Carlo estimator (REINFORCE with reward-to-go).

Implements the sample-mean gradient estimators of eqs 19 and 23, upgraded to
the standard lower-variance form: per-timestep reward-to-go with a per-
timestep batch-mean baseline (state-independent baselines leave the gradient
unbiased; using the same batch to fit the baseline introduces an O(1/N) bias
that is negligible at the batch sizes used here).

With ``time_discount_weighting`` on, each log-prob term carries its gamma^t
weight so the estimator targets the *discounted* policy gradient — the same
quantity the exact backend computes; the two are cross-checked in
``tests/test_rollout_estimator.py``.

Cost per call: O(num_episodes * horizon) environment steps.
"""

from __future__ import annotations

import torch

from crl.buffers import BufferSet, Trajectory
from crl.envs.base import Task
from crl.estimators.base import ValueEstimator
from crl.policies.base import Policy


class MonteCarloEstimator(ValueEstimator):
    """Fresh-rollout evaluation and gradients (the naive baseline backend)."""

    def __init__(
        self,
        episodes_per_eval: int = 16,
        episodes_per_grad: int = 16,
        episodes_per_ref: int = 64,
        baseline: str = "batch_mean",
        time_discount_weighting: bool = True,
        buffer_set: BufferSet | None = None,
    ) -> None:
        if baseline not in ("batch_mean", "none"):
            raise ValueError(f"Unknown baseline '{baseline}'")
        self.episodes_per_eval = episodes_per_eval
        self.episodes_per_grad = episodes_per_grad
        self.episodes_per_ref = episodes_per_ref
        self.baseline = baseline
        self.time_discount_weighting = time_discount_weighting
        self.buffer_set = buffer_set
        self._envs: dict[int, object] = {}  # task_id -> cached env instance

    def _env_for(self, task: Task):
        if task.spec.task_id not in self._envs:
            self._envs[task.spec.task_id] = task.make_env()
        return self._envs[task.spec.task_id]

    def _run_episodes(
        self, policy: Policy, task: Task, num_episodes: int
    ) -> list[Trajectory]:
        """Collect episodes under ``policy``; actions sampled without grad.

        Environment stochasticity is seeded from the global torch RNG so a
        single ``set_seed`` call makes whole runs reproducible.

        Tasks that expose a batched ``vector_rollout`` (e.g. the tabular
        gridworld, whose transition tensor lets N episodes step in lockstep
        with one policy forward pass each timestep) take a fast path; the
        estimate is identical in expectation, only much cheaper. Any other task
        falls back to the per-episode gym loop below.
        """
        vector_rollout = getattr(task, "vector_rollout", None)
        if vector_rollout is not None:
            episodes = vector_rollout(policy, num_episodes)
            if self.buffer_set is not None:
                buffer = self.buffer_set.for_task(task.spec.task_id)
                for trajectory in episodes:
                    buffer.add(trajectory)
            return episodes

        env = self._env_for(task)
        episodes: list[Trajectory] = []
        # One reset-seed per collection; later resets continue the env RNG.
        collect_seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        for episode_idx in range(num_episodes):
            obs, _ = env.reset(seed=collect_seed if episode_idx == 0 else None)
            obs_list, act_list, rew_list, logp_list = [], [], [], []
            terminated = truncated = False
            while not (terminated or truncated):
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
                with torch.no_grad():
                    dist = policy.dist(obs_tensor.unsqueeze(0), task.spec.task_id)
                    action = int(dist.sample().item())
                    logp = float(dist.log_prob(torch.tensor([action])).item())
                next_obs, reward, terminated, truncated, _ = env.step(action)
                obs_list.append(obs_tensor)
                act_list.append(action)
                rew_list.append(float(reward))
                logp_list.append(logp)
                obs = next_obs
            episodes.append(
                Trajectory(
                    obs=torch.stack(obs_list),
                    actions=torch.tensor(act_list, dtype=torch.long),
                    rewards=torch.tensor(rew_list, dtype=torch.float32),
                    behavior_logps=torch.tensor(logp_list, dtype=torch.float32),
                    terminated=terminated,
                )
            )
        if self.buffer_set is not None:
            buffer = self.buffer_set.for_task(task.spec.task_id)
            for trajectory in episodes:
                buffer.add(trajectory)
        return episodes

    @staticmethod
    def _reward_to_go(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
        """G_t = sum_{t'>=t} gamma^(t'-t) r_t'  (reverse scan, O(T))."""
        returns = torch.empty_like(rewards)
        running = 0.0
        for t in range(len(rewards) - 1, -1, -1):
            running = float(rewards[t]) + gamma * running
            returns[t] = running
        return returns

    def evaluate(
        self, policy: Policy, task: Task, num_episodes: int | None = None
    ) -> float:
        n = num_episodes or self.episodes_per_eval
        episodes = self._run_episodes(policy, task, n)
        discounted = [
            float(self._reward_to_go(ep.rewards, task.gamma)[0]) for ep in episodes
        ]
        return sum(discounted) / len(discounted)

    def evaluate_return(
        self, policy: Policy, task: Task, num_episodes: int | None = None
    ) -> float:
        """Mean *undiscounted* episode return -- the reported task performance
        (e.g. MinAtar game score), as opposed to the discounted value the
        objective optimizes. Reward scaling (if any) is already applied."""
        n = num_episodes or self.episodes_per_eval
        episodes = self._run_episodes(policy, task, n)
        return float(sum(float(ep.rewards.sum()) for ep in episodes) / len(episodes))

    def surrogate_objective(
        self, policy: Policy, task: Task
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        episodes = self._run_episodes(policy, task, self.episodes_per_grad)
        num_episodes = len(episodes)
        gamma = task.gamma
        rtg_per_episode = [self._reward_to_go(ep.rewards, gamma) for ep in episodes]

        # Per-timestep batch-mean baseline over episodes still alive at t.
        max_len = max(len(ep.rewards) for ep in episodes)
        padded = torch.zeros(num_episodes, max_len)
        mask = torch.zeros(num_episodes, max_len)
        for row, rtg in enumerate(rtg_per_episode):
            padded[row, : len(rtg)] = rtg
            mask[row, : len(rtg)] = 1.0
        if self.baseline == "batch_mean" and num_episodes > 1:
            baseline = padded.sum(dim=0) / mask.sum(dim=0).clamp(min=1.0)
            advantage = (padded - baseline.unsqueeze(0)) * mask
        else:
            advantage = padded * mask

        # Single batched log-prob pass over all visited states (differentiable).
        # Everything is moved to the policy's device so GPU policies (e.g. the
        # MinAtar CNN) work unchanged; on CPU these .to() calls are no-ops.
        device = next(policy.parameters()).device
        flat_obs = torch.cat([ep.obs for ep in episodes]).to(device)
        flat_actions = torch.cat([ep.actions for ep in episodes]).to(device)
        flat_advantage = torch.cat(
            [advantage[row, : len(ep.rewards)] for row, ep in enumerate(episodes)]
        ).to(device)
        if self.time_discount_weighting:
            flat_time_weight = torch.cat(
                [gamma ** torch.arange(len(ep.rewards), dtype=torch.float32)
                 for ep in episodes]
            ).to(device)
        else:
            flat_time_weight = torch.ones_like(flat_advantage)

        dist = policy.dist(flat_obs, task.spec.task_id)
        logps = dist.log_prob(flat_actions)
        objective = (flat_time_weight * logps * flat_advantage).sum() / num_episodes
        entropy_term = dist.entropy().mean()

        stats = {
            "value": float(sum(rtg[0] for rtg in rtg_per_episode) / num_episodes),
            "entropy": float(entropy_term.detach()),
            "mean_episode_len": float(
                sum(len(ep.rewards) for ep in episodes) / num_episodes
            ),
        }
        return objective, entropy_term, stats
