"""Task-performance metrics from real rollouts.

The trainer's value V^pi (expected discounted return) is what the algorithm
optimizes, but it is not directly interpretable as task performance. This
module rolls the final policy out in each task's environment and reports
concrete, human-readable metrics:

    success_rate  -- fraction of episodes that solve the task (reach the goal /
                     terminate rather than time out)
    mean_return   -- average UNDISCOUNTED episode return (for the gridworld
                     goal reward this equals the success rate)
    mean_steps    -- average steps per episode (path efficiency; lower is better
                     for goal-reaching tasks)

These are the numbers to put in the paper -- "% of the task solved" -- rather
than the raw value.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from crl.envs.base import Task
from crl.policies.base import Policy


@dataclass
class Performance:
    """Rollout performance of one policy on one task."""

    success_rate: float
    mean_return: float  # undiscounted
    mean_steps: float
    num_episodes: int


@torch.no_grad()
def rollout_performance(
    policy: Policy, task: Task, num_episodes: int = 200
) -> Performance:
    """Run ``num_episodes`` episodes of ``policy`` on ``task`` and summarize.

    Success is env-dependent (``task.success_on_termination``): for a goal task
    (gridworld) an episode succeeds if it TERMINATES at the goal; for a survival
    task (CartPole) it succeeds if it survives to the horizon (is TRUNCATED)
    rather than terminating early (the pole falling).
    """
    success_on_termination = getattr(task, "success_on_termination", True)
    env = task.make_env()
    successes = 0
    total_return = 0.0
    total_steps = 0
    for episode_idx in range(num_episodes):
        obs, _ = env.reset(seed=episode_idx)
        terminated = truncated = False
        while not (terminated or truncated):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
            action = policy.act(obs_tensor, task.spec.task_id)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_return += float(reward)
            total_steps += 1
        successes += int(terminated if success_on_termination else truncated)
    return Performance(
        success_rate=successes / num_episodes,
        mean_return=total_return / num_episodes,
        mean_steps=total_steps / num_episodes,
        num_episodes=num_episodes,
    )
