"""Parametric gridworld family with exact tabular MDP tensors.

Tasks share an ``n x n`` grid and four movement actions; each task moves the
goal cell. Goals in different corners force different actions in shared
states, which is exactly the task-conflict regime the constraint is meant to
manage. Because every task exposes ``(P, r, rho)`` tensors, values and policy
gradients can be computed exactly (see ``crl/estimators/exact.py``), making
this the cheapest possible check that the update rules work.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch

from crl.buffers import Trajectory
from crl.envs.base import TabularTask, TaskFamily, TaskSpec

# Action index -> (row delta, col delta): up, down, left, right.
_MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))


class TabularEnv(gym.Env):
    """Gymnasium wrapper that samples transitions from exact MDP tensors.

    Observations are one-hot state indicators (``float32``, length S) so the
    same policies work for tabular and function-approximation settings.
    Sampling from the same tensors used by the exact estimator guarantees
    Monte-Carlo and dynamic-programming evaluations agree in expectation.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        transition: np.ndarray,
        initial_dist: np.ndarray,
        goal_state: int,
        goal_reward: float,
        step_penalty: float,
        max_steps: int,
    ) -> None:
        super().__init__()
        self._transition = transition  # [S, A, S]
        self._initial_dist = initial_dist  # [S]
        self._goal_state = goal_state
        self._goal_reward = goal_reward
        self._step_penalty = step_penalty
        self._max_steps = max_steps
        num_states, num_actions, _ = transition.shape
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(num_states,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(num_actions)
        self._state = 0
        self._steps = 0

    def _obs(self) -> np.ndarray:
        one_hot = np.zeros(self._transition.shape[0], dtype=np.float32)
        one_hot[self._state] = 1.0
        return one_hot

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._state = int(
            self.np_random.choice(len(self._initial_dist), p=self._initial_dist)
        )
        self._steps = 0
        return self._obs(), {}

    def step(self, action: int):
        probs = self._transition[self._state, action]
        next_state = int(self.np_random.choice(len(probs), p=probs))
        # Reward is granted on the transition INTO the goal; the goal itself
        # is absorbing with zero reward, matching the reward tensor.
        entered_goal = next_state == self._goal_state and self._state != self._goal_state
        reward = self._goal_reward if entered_goal else self._step_penalty
        self._state = next_state
        self._steps += 1
        terminated = next_state == self._goal_state
        truncated = self._steps >= self._max_steps
        return self._obs(), reward, terminated, truncated, {}


class GridWorldTask(TabularTask):
    """One gridworld goal-reaching task with exact tensors."""

    def __init__(
        self,
        spec: TaskSpec,
        gamma: float,
        transition: torch.Tensor,
        reward: torch.Tensor,
        initial_dist: torch.Tensor,
        goal_state: int,
        goal_reward: float,
        step_penalty: float,
        max_steps: int,
    ) -> None:
        super().__init__(spec, gamma)
        self.transition = transition
        self.reward = reward
        self.initial_dist = initial_dist
        self._goal_state = goal_state
        self._goal_reward = goal_reward
        self._step_penalty = step_penalty
        self._max_steps = max_steps

    def make_env(self) -> gym.Env:
        return TabularEnv(
            transition=self.transition.numpy(),
            initial_dist=self.initial_dist.numpy(),
            goal_state=self._goal_state,
            goal_reward=self._goal_reward,
            step_penalty=self._step_penalty,
            max_steps=self._max_steps,
        )

    @torch.no_grad()
    def vector_rollout(self, policy, num_episodes: int) -> list[Trajectory]:
        """Collect ``num_episodes`` episodes in lockstep from the exact tensors.

        Steps every episode of the batch with a *single* batched policy forward
        pass per timestep, instead of the per-episode Python loop in
        ``MonteCarloEstimator._run_episodes``. Because the gridworld exposes its
        ``[S, A, S]`` transition tensor, next states are sampled directly from
        it (``torch.multinomial``), so the sampled MDP is identical to the
        gym ``TabularEnv`` and the estimate stays unbiased -- only the wall-clock
        cost drops (one forward over N states beats N forwards over one state).

        Environment stochasticity is drawn from the global torch RNG, so a
        single ``set_seed`` makes whole runs reproducible, matching the scalar
        path. Returns the same ``list[Trajectory]`` that path returns, so every
        downstream estimator (value, REINFORCE surrogate) is unchanged.
        """
        transition = self.transition  # [S, A, S]
        num_states = transition.shape[0]
        goal = self._goal_state
        eye = torch.eye(num_states, dtype=torch.float32)

        state = torch.multinomial(self.initial_dist, num_episodes, replacement=True)
        alive = torch.ones(num_episodes, dtype=torch.bool)
        reached_goal = torch.zeros(num_episodes, dtype=torch.bool)
        lengths = torch.zeros(num_episodes, dtype=torch.long)

        obs_hist: list[torch.Tensor] = []
        act_hist: list[torch.Tensor] = []
        rew_hist: list[torch.Tensor] = []
        logp_hist: list[torch.Tensor] = []

        for _ in range(self._max_steps):
            obs = eye[state]  # [N, S] one-hot, matching TabularEnv._obs
            dist = policy.dist(obs, self.spec.task_id)
            action = dist.sample()  # [N]
            logp = dist.log_prob(action)  # [N]

            probs = transition[state, action]  # [N, S]
            next_state = torch.multinomial(probs, 1).squeeze(1)  # [N]
            # Reward granted on the transition INTO the goal (matches TabularEnv).
            entered_goal = (next_state == goal) & (state != goal)
            reward = torch.where(
                entered_goal,
                torch.full_like(obs[:, 0], self._goal_reward),
                torch.full_like(obs[:, 0], self._step_penalty),
            )

            obs_hist.append(obs)
            act_hist.append(action)
            rew_hist.append(reward)
            logp_hist.append(logp)

            lengths = torch.where(alive, lengths + 1, lengths)
            hit = alive & (next_state == goal)  # goal is terminal
            reached_goal = reached_goal | hit
            alive = alive & ~hit
            state = next_state
            if not bool(alive.any()):
                break

        obs_stack = torch.stack(obs_hist)  # [T, N, S]
        act_stack = torch.stack(act_hist)  # [T, N]
        rew_stack = torch.stack(rew_hist)  # [T, N]
        logp_stack = torch.stack(logp_hist)  # [T, N]

        episodes: list[Trajectory] = []
        for n in range(num_episodes):
            length = int(lengths[n])
            terminated = bool(reached_goal[n])  # else truncated at the horizon
            episodes.append(
                Trajectory(
                    obs=obs_stack[:length, n],
                    actions=act_stack[:length, n],
                    rewards=rew_stack[:length, n],
                    behavior_logps=logp_stack[:length, n],
                    terminated=terminated,
                )
            )
        return episodes


def _build_task_tensors(
    size: int,
    slip: float,
    goal_state: int,
    goal_reward: float,
    step_penalty: float,
    start: str | list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct ``(P, r, rho)`` for one goal placement.

    Complexity O(S * A * S) time and space with S = size**2, A = 4.
    """
    num_states = size * size
    num_actions = len(_MOVES)

    # Deterministic move table; bumping a wall keeps the agent in place.
    intended = np.zeros((num_states, num_actions), dtype=np.int64)
    for state in range(num_states):
        row, col = divmod(state, size)
        for action, (d_row, d_col) in enumerate(_MOVES):
            new_row = min(max(row + d_row, 0), size - 1)
            new_col = min(max(col + d_col, 0), size - 1)
            intended[state, action] = new_row * size + new_col

    transition = np.zeros((num_states, num_actions, num_states), dtype=np.float64)
    for state in range(num_states):
        if state == goal_state:
            transition[state, :, state] = 1.0  # absorbing goal
            continue
        for action in range(num_actions):
            # With prob slip the executed action is uniform over all actions.
            transition[state, action, intended[state, action]] += 1.0 - slip
            for other in range(num_actions):
                transition[state, action, intended[state, other]] += slip / num_actions

    # Expected immediate reward r(s, a) = E_{s'}[R(s, s')] with
    # R = goal_reward on entering the goal, step_penalty otherwise, 0 at goal.
    reward = np.zeros((num_states, num_actions), dtype=np.float64)
    for state in range(num_states):
        if state == goal_state:
            continue
        p_goal = transition[state, :, goal_state]
        reward[state] = goal_reward * p_goal + step_penalty * (1.0 - p_goal)

    initial = np.zeros(num_states, dtype=np.float64)
    if start == "uniform":
        initial[:] = 1.0
        initial[goal_state] = 0.0  # never start on the goal
    else:
        row, col = start
        start_state = row * size + col
        if start_state == goal_state:
            raise ValueError("Start cell coincides with the goal cell.")
        initial[start_state] = 1.0
    initial /= initial.sum()

    return (
        torch.tensor(transition, dtype=torch.float32),
        torch.tensor(reward, dtype=torch.float32),
        torch.tensor(initial, dtype=torch.float32),
    )


class GridWorldFamily(TaskFamily):
    """Goal-relocation gridworld family.

    Family params (``env.params``):
        size: grid side length (default 5)
        slip: probability the executed action is uniformly random (default 0.1)
        gamma: discount factor shared by all tasks (default 0.95)
        goal_reward: reward for entering the goal (default 1.0)
        step_penalty: reward for every other transition (default 0.0)
        start: "uniform" or an explicit [row, col] (default "uniform")
        max_steps: episode truncation horizon for sampled rollouts (default 100)

    Per-task params (``env.tasks[i]``):
        goal: [row, col] of that task's goal cell (required)
    """

    is_tabular = True

    def __init__(self, params: dict[str, Any], tasks: list[dict[str, Any]]) -> None:
        size = int(params.get("size", 5))
        slip = float(params.get("slip", 0.1))
        gamma = float(params.get("gamma", 0.95))
        goal_reward = float(params.get("goal_reward", 1.0))
        step_penalty = float(params.get("step_penalty", 0.0))
        start = params.get("start", "uniform")
        max_steps = int(params.get("max_steps", 100))
        if not tasks:
            raise ValueError("GridWorldFamily needs a non-empty env.tasks list.")

        self.obs_dim = size * size
        self.num_actions = len(_MOVES)
        self.tasks = []
        for task_id, task_params in enumerate(tasks):
            goal_row, goal_col = task_params["goal"]
            goal_state = goal_row * size + goal_col
            transition, reward, initial = _build_task_tensors(
                size, slip, goal_state, goal_reward, step_penalty, start
            )
            spec = TaskSpec(task_id, f"grid{size}x{size}-goal({goal_row},{goal_col})", task_params)
            self.tasks.append(
                GridWorldTask(
                    spec,
                    gamma,
                    transition,
                    reward,
                    initial,
                    goal_state,
                    goal_reward,
                    step_penalty,
                    max_steps,
                )
            )
