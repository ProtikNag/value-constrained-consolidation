"""MinAtar task family: five miniaturized Atari games as a continual sequence.

Each task is one MinAtar game (Breakout, SpaceInvaders, Freeway, Asterix,
Seaquest). Observations are 10x10 binary-channel images; all games share a
6-action space, and channels are zero-padded to a common width so a single
shared convolutional trunk can process every game (which is exactly the
shared-parameter interference that induces forgetting on genuinely different
games -- unlike the compatible CartPole physics tasks).

Performance is the raw game score (undiscounted episode return); there is no
"goal" success notion. The batched ``vector_rollout`` steps many game instances
in lockstep with one policy forward per timestep, the same fast path the
gridworld/CartPole envs expose, so the REINFORCE estimator is affordable.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch

from crl.buffers import Trajectory
from crl.envs.base import Task, TaskFamily, TaskSpec

MINATAR_GAMES = ("breakout", "space_invaders", "freeway", "asterix", "seaquest")


def _pad_state(state: np.ndarray, max_channels: int) -> np.ndarray:
    """(10, 10, C) bool -> (max_channels, 10, 10) float32, channel-padded."""
    chw = np.transpose(state.astype(np.float32), (2, 0, 1))  # (C, 10, 10)
    if chw.shape[0] < max_channels:
        pad = np.zeros((max_channels - chw.shape[0], *chw.shape[1:]), dtype=np.float32)
        chw = np.concatenate([chw, pad], axis=0)
    return chw


class _MinAtarGymEnv(gym.Env):
    """Minimal Gymnasium adapter (used by the rollout-performance metric)."""

    def __init__(self, game: str, max_channels: int, max_steps: int,
                 sticky_action_prob: float, difficulty_ramping: bool,
                 reward_scale: float) -> None:
        from minatar import Environment
        self._env = Environment(game, sticky_action_prob=sticky_action_prob,
                                difficulty_ramping=difficulty_ramping)
        self._max_channels = max_channels
        self._max_steps = max_steps
        self._reward_scale = reward_scale
        self._steps = 0
        self.action_space = gym.spaces.Discrete(self._env.num_actions())

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._env.seed(seed)
        self._env.reset()
        self._steps = 0
        return _pad_state(self._env.state(), self._max_channels), {}

    def step(self, action: int):
        reward, done = self._env.act(int(action))
        self._steps += 1
        truncated = self._steps >= self._max_steps
        obs = _pad_state(self._env.state(), self._max_channels)
        return (obs, float(reward) * self._reward_scale, bool(done),
                bool(truncated and not done), {})


class MinAtarTask(Task):
    """One MinAtar game. Success is not goal-based; the metric is game score."""

    success_on_termination = False  # no goal; report mean return (game score)

    def __init__(self, spec: TaskSpec, gamma: float, game: str, max_channels: int,
                 max_steps: int, sticky_action_prob: float,
                 difficulty_ramping: bool, reward_scale: float = 1.0) -> None:
        super().__init__(spec, gamma)
        self._game = game
        self._max_channels = max_channels
        self._max_steps = max_steps
        self._sticky = sticky_action_prob
        self._ramping = difficulty_ramping
        self._reward_scale = reward_scale

    def make_env(self) -> gym.Env:
        return _MinAtarGymEnv(self._game, self._max_channels, self._max_steps,
                              self._sticky, self._ramping, self._reward_scale)

    @torch.no_grad()
    def vector_rollout(self, policy, num_episodes: int) -> list[Trajectory]:
        """Collect ``num_episodes`` episodes in lockstep: one batched policy
        forward per timestep across all still-alive game instances."""
        from minatar import Environment
        device = next(policy.parameters()).device
        task_id = self.spec.task_id
        base_seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        envs = []
        for i in range(num_episodes):
            e = Environment(self._game, sticky_action_prob=self._sticky,
                            difficulty_ramping=self._ramping)
            e.seed(base_seed + i)
            e.reset()
            envs.append(e)

        alive = [True] * num_episodes
        obs_hist: list[list] = [[] for _ in range(num_episodes)]
        act_hist: list[list] = [[] for _ in range(num_episodes)]
        rew_hist: list[list] = [[] for _ in range(num_episodes)]
        logp_hist: list[list] = [[] for _ in range(num_episodes)]
        terminated = [False] * num_episodes

        for _ in range(self._max_steps):
            active = [i for i in range(num_episodes) if alive[i]]
            if not active:
                break
            batch = np.stack([_pad_state(envs[i].state(), self._max_channels)
                              for i in active])
            obs = torch.as_tensor(batch, dtype=torch.float32, device=device)
            dist = policy.dist(obs, task_id)
            actions = dist.sample()
            logps = dist.log_prob(actions)
            actions_cpu = actions.to("cpu")
            for j, i in enumerate(active):
                a = int(actions_cpu[j])
                reward, done = envs[i].act(a)
                obs_hist[i].append(obs[j].to("cpu"))
                act_hist[i].append(a)
                rew_hist[i].append(float(reward) * self._reward_scale)
                logp_hist[i].append(float(logps[j]))
                if done:
                    alive[i] = False
                    terminated[i] = True

        episodes: list[Trajectory] = []
        for i in range(num_episodes):
            if not obs_hist[i]:
                continue
            episodes.append(
                Trajectory(
                    obs=torch.stack(obs_hist[i]),
                    actions=torch.tensor(act_hist[i], dtype=torch.long),
                    rewards=torch.tensor(rew_hist[i], dtype=torch.float32),
                    behavior_logps=torch.tensor(logp_hist[i], dtype=torch.float32),
                    terminated=terminated[i],
                )
            )
        return episodes


class MinAtarFamily(TaskFamily):
    """Sequence of MinAtar games.

    Family params (``env.params``):
        gamma: discount (default 0.99)
        max_steps: episode truncation horizon (default 1000)
        sticky_action_prob: MinAtar action-repeat probability (default 0.1)
        difficulty_ramping: MinAtar within-episode difficulty ramp (default True)

    Per-task params (``env.tasks[i]``):
        game: one of MINATAR_GAMES (required)
    """

    is_tabular = False

    def __init__(self, params: dict[str, Any], tasks: list[dict[str, Any]]) -> None:
        from minatar import Environment
        gamma = float(params.get("gamma", 0.99))
        max_steps = int(params.get("max_steps", 1000))
        sticky = float(params.get("sticky_action_prob", 0.1))
        ramping = bool(params.get("difficulty_ramping", True))
        if not tasks:
            raise ValueError("MinAtarFamily needs a non-empty env.tasks list.")

        games = [t["game"] for t in tasks]
        for g in games:
            if g not in MINATAR_GAMES:
                raise KeyError(f"Unknown MinAtar game '{g}'; available: {MINATAR_GAMES}")
        # Common channel width so one shared conv trunk fits every game.
        max_channels = max(Environment(g).state_shape()[2] for g in games)

        self.obs_shape = (max_channels, 10, 10)
        self.obs_dim = max_channels * 10 * 10
        self.num_actions = 6
        self.tasks = []
        for task_id, t in enumerate(tasks):
            g = t["game"]
            # Per-task reward scale so returns are ~O(1) across games (default 1);
            # normalizes the multi-task objective and the squared-value constraint.
            scale = float(t.get("reward_scale", 1.0))
            spec = TaskSpec(task_id, f"minatar-{g}", {"game": g})
            self.tasks.append(
                MinAtarTask(spec, gamma, g, max_channels, max_steps, sticky,
                            ramping, scale))
