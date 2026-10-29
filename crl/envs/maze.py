"""Continual maze-navigation family: one maze per task, sampled REINFORCE.

Each task is a distinct MAZE (walls, dead-ends, and some loops so multiple paths
exist). Within a task the maze is FIXED; the start and goal are randomized every
episode (and verified reachable by BFS). Different tasks are different mazes, so
navigating them requires genuinely conflicting routing knowledge in the shared
network -- unlike open-grid goal-conditioned navigation (which has a single
general "move toward the goal" rule and therefore barely forgets), a maze route
cannot be generalized from a local view, so naive fine-tuning forgets old mazes.

Observation (goal-conditioned, one shared head, NO task-id):
    factored one-hot of the current cell   (2*size)
  ++ factored one-hot of the goal cell      (2*size)
  ++ a local K x K wall patch around the agent (K*K; out-of-bounds = wall)
So the agent knows where it is, which cell is the goal, and its immediate
surroundings (a partial map) -- but not the global layout, which it must encode
in its weights per maze.

Reward: potential-based shaping along the BFS shortest-path distance to the goal
(dense, and accounts for walls so it rewards genuine progress, penalizes moving
into dead-ends), plus a terminal goal reward. With discounting, reaching the goal
sooner -- i.e. via a SHORTER path -- yields a higher (discounted) value, so the
reported value is graded and prefers short paths. Slip adds action uncertainty.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from crl.buffers import Trajectory
from crl.envs.base import Task, TaskFamily, TaskSpec

_MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))


# --------------------------------------------------------------------------- #
# maze generation + graph utilities (numpy/int; done once per task)
# --------------------------------------------------------------------------- #
def _generate_maze(size: int, braid: float, rng: np.random.Generator) -> np.ndarray:
    """Recursive-backtracker maze (True = wall). ``braid`` in [0,1] removes that
    fraction of internal walls to create loops (multiple paths)."""
    wall = np.ones((size, size), dtype=bool)
    stack = [(0, 0)]
    wall[0, 0] = False
    while stack:
        r, c = stack[-1]
        nbrs = []
        for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size and wall[nr, nc]:
                nbrs.append((nr, nc, r + dr // 2, c + dc // 2))
        if not nbrs:
            stack.pop()
            continue
        nr, nc, wr, wc = nbrs[rng.integers(len(nbrs))]
        wall[wr, wc] = False
        wall[nr, nc] = False
        stack.append((nr, nc))
    # braid: open some internal walls that separate two free cells -> loops
    for r in range(1, size - 1):
        for c in range(1, size - 1):
            if wall[r, c] and rng.random() < braid:
                free_h = (not wall[r, c - 1]) and (not wall[r, c + 1])
                free_v = (not wall[r - 1, c]) and (not wall[r + 1, c])
                if free_h or free_v:
                    wall[r, c] = False
    return wall


def _next_cell_table(wall: np.ndarray) -> np.ndarray:
    """[S, 4] next-state table: moving into a wall / off-grid stays in place."""
    size = wall.shape[0]
    S = size * size
    nxt = np.zeros((S, 4), dtype=np.int64)
    for s in range(S):
        r, c = divmod(s, size)
        for a, (dr, dc) in enumerate(_MOVES):
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size and not wall[nr, nc]:
                nxt[s, a] = nr * size + nc
            else:
                nxt[s, a] = s
    return nxt


def _bfs_dist(nxt: np.ndarray, free: np.ndarray, goal: int) -> np.ndarray:
    """Shortest-path distance (in steps) from every cell to ``goal`` over the
    free-cell graph; unreachable/wall cells get a large value."""
    S = nxt.shape[0]
    dist = np.full(S, 1 << 20, dtype=np.int64)
    dist[goal] = 0
    q = deque([goal])
    while q:
        s = q.popleft()
        for a in range(4):
            p = int(nxt[s, a])          # neighbor that can step INTO s is symmetric
            if free[p] and dist[p] == (1 << 20):
                # p reaches s in one move iff nxt[p, a'] == s for some a'; moves
                # are reversible on a grid, so nxt[s,a]=p implies nxt[p,rev]=s
                dist[p] = dist[s] + 1
                q.append(p)
    return dist


def _local_view_table(wall: np.ndarray, k: int) -> np.ndarray:
    """[S, k*k] wall patch around each cell (out-of-bounds counts as wall)."""
    size = wall.shape[0]
    pad = k // 2
    padded = np.ones((size + 2 * pad, size + 2 * pad), dtype=np.float32)
    padded[pad:pad + size, pad:pad + size] = wall.astype(np.float32)
    S = size * size
    table = np.zeros((S, k * k), dtype=np.float32)
    for s in range(S):
        r, c = divmod(s, size)
        table[s] = padded[r:r + k, c:c + k].reshape(-1)
    return table


def _free_cells(wall: np.ndarray) -> np.ndarray:
    return (~wall).reshape(-1)


# --------------------------------------------------------------------------- #
# observation builder (torch, batched)
# --------------------------------------------------------------------------- #
def _obs_batch(pos: torch.Tensor, goal: torch.Tensor, size: int,
               view_table: torch.Tensor) -> torch.Tensor:
    n = pos.shape[0]
    idx = torch.arange(n)
    row = pos // size; col = pos % size
    grow = goal // size; gcol = goal % size
    pr = torch.zeros(n, size); pr[idx, row] = 1.0
    pc = torch.zeros(n, size); pc[idx, col] = 1.0
    gr = torch.zeros(n, size); gr[idx, grow] = 1.0
    gc = torch.zeros(n, size); gc[idx, gcol] = 1.0
    return torch.cat([pr, pc, gr, gc, view_table[pos]], dim=1)


# --------------------------------------------------------------------------- #
# task + env
# --------------------------------------------------------------------------- #
class MazeTask(Task):
    success_on_termination = True

    def __init__(self, spec, gamma, size, wall, slip, goal_reward, shaping,
                 view_k, max_steps, wall_penalty=0.0):
        super().__init__(spec, gamma)
        self.size = size
        self.wall = wall
        self.slip = slip
        self.goal_reward = goal_reward
        self.shaping = shaping
        self.view_k = view_k
        self.max_steps = max_steps
        # Cost of bumping a wall (attempting a blocked move); <= 0, CONSTANT within
        # this task but different across mazes (some 0, some a random negative).
        self.wall_penalty = wall_penalty
        self.nxt = _next_cell_table(wall)
        self.free = _free_cells(wall)
        self.free_idx = np.nonzero(self.free)[0]
        self._view = torch.from_numpy(_local_view_table(wall, view_k))
        self._norm = float(size)  # potential normalizer

    # ---- shared helpers ----
    def _sample_start_goal(self, rng):
        """Random distinct free (start, goal), guaranteed reachable (BFS)."""
        while True:
            start, goal = rng.choice(self.free_idx, size=2, replace=False)
            dist = _bfs_dist(self.nxt, self.free, int(goal))
            if dist[int(start)] < (1 << 20):
                return int(start), int(goal), dist

    def make_env(self):
        return _MazeEnv(self)

    @torch.no_grad()
    def vector_rollout(self, policy, num_episodes: int) -> list[Trajectory]:
        rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,)).item()))
        tid = self.spec.task_id
        size = self.size
        n = num_episodes
        nxt = torch.from_numpy(self.nxt)               # [S,4]
        starts, goals, dists = [], [], []
        for _ in range(n):
            s, g, d = self._sample_start_goal(rng)
            starts.append(s); goals.append(g); dists.append(d)
        pos = torch.tensor(starts)
        goal = torch.tensor(goals)
        dist = torch.from_numpy(np.stack(dists)).float()  # [n, S] BFS dist to goal
        phi = -dist / self._norm                          # potential per cell

        alive = torch.ones(n, dtype=torch.bool)
        reached = torch.zeros(n, dtype=torch.bool)
        lengths = torch.zeros(n, dtype=torch.long)
        rows = torch.arange(n)
        obs_h, act_h, rew_h, logp_h = [], [], [], []

        def phi_of(p):
            return phi[rows, p]

        for _ in range(self.max_steps):
            obs = _obs_batch(pos, goal, size, self._view)
            d = policy.dist(obs, tid)
            action = d.sample()
            logp = d.log_prob(action)
            slip = torch.rand(n) < self.slip
            exec_a = torch.where(slip, torch.randint(0, 4, (n,)), action)
            npos = nxt[pos, exec_a]
            hit = alive & (npos == goal)
            bump = npos == pos                       # blocked move (wall / edge)
            # potential shaping (dense) + terminal goal reward + wall-hit penalty
            reward = self.shaping * (self.gamma * phi_of(npos) - phi_of(pos))
            reward = reward + hit.float() * self.goal_reward
            reward = reward + bump.float() * self.wall_penalty

            obs_h.append(obs); act_h.append(action); rew_h.append(reward); logp_h.append(logp)
            lengths = torch.where(alive, lengths + 1, lengths)
            reached = reached | hit
            alive = alive & ~hit
            pos = npos
            if not bool(alive.any()):
                break

        O = torch.stack(obs_h); A = torch.stack(act_h)
        R = torch.stack(rew_h); L = torch.stack(logp_h)
        eps = []
        for i in range(n):
            ln = int(lengths[i])
            eps.append(Trajectory(obs=O[:ln, i], actions=A[:ln, i], rewards=R[:ln, i],
                                  behavior_logps=L[:ln, i], terminated=bool(reached[i])))
        return eps


class _MazeEnv(gym.Env):
    """Single-episode maze env (used by the rollout-performance metric)."""

    def __init__(self, task: MazeTask):
        self.t = task
        self.size = task.size
        obs_dim = 4 * task.size + task.view_k * task.view_k
        self.observation_space = gym.spaces.Box(0.0, 1.0, (obs_dim,), np.float32)
        self.action_space = gym.spaces.Discrete(4)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)
        self._pos, self._goal, dist = self.t._sample_start_goal(rng)
        self._phi = (-dist / self.t._norm).astype(np.float32)
        self._steps = 0
        return self._obs(), {}

    def _obs(self):
        pos = torch.tensor([self._pos]); goal = torch.tensor([self._goal])
        return _obs_batch(pos, goal, self.size, self.t._view)[0].numpy()

    def step(self, action):
        if np.random.random() < self.t.slip:
            action = np.random.randint(4)
        npos = int(self.t.nxt[self._pos, action])
        hit = npos == self._goal
        reward = self.t.shaping * (self.t.gamma * self._phi[npos] - self._phi[self._pos])
        if hit:
            reward += self.t.goal_reward
        if npos == self._pos:                        # bumped a wall / edge
            reward += self.t.wall_penalty
        self._pos = npos
        self._steps += 1
        truncated = self._steps >= self.t.max_steps
        return self._obs(), float(reward), bool(hit), bool(truncated and not hit), {}


class MazeFamily(TaskFamily):
    """Continual maze family: one distinct maze per task.

    Family params (``env.params``):
        size: grid side (odd recommended; default 21)
        braid: fraction of internal walls reopened for loops (default 0.08)
        slip: action-uncertainty probability (default 0.1)
        gamma: discount (default 0.99)
        goal_reward: terminal reward for reaching the goal (default 1.0)
        shaping: BFS-potential shaping scale (default 1.0)
        view_k: side length of the local wall patch (default 5)
        max_steps: episode horizon (default 200)
        maze_base_seed: base RNG seed for maze generation (default 0)

    Per-task params (``env.tasks[i]``): optional ``maze_seed`` (else base+index).
    """

    is_tabular = False

    def __init__(self, params: dict[str, Any], tasks: list[dict[str, Any]]) -> None:
        size = int(params.get("size", 21))
        braid = float(params.get("braid", 0.08))
        slip = float(params.get("slip", 0.1))
        gamma = float(params.get("gamma", 0.99))
        goal_reward = float(params.get("goal_reward", 1.0))
        shaping = float(params.get("shaping", 1.0))
        view_k = int(params.get("view_k", 5))
        max_steps = int(params.get("max_steps", 200))
        base_seed = int(params.get("maze_base_seed", 0))
        # Per-maze wall-hit penalty: with prob ``wall_penalty_prob`` a maze hits
        # with a random negative cost in [-wall_penalty_max, -0.02], else 0.
        wp_prob = float(params.get("wall_penalty_prob", 0.5))
        wp_max = float(params.get("wall_penalty_max", 0.3))
        if not tasks:
            raise ValueError("MazeFamily needs a non-empty env.tasks list.")

        self.obs_dim = 4 * size + view_k * view_k
        self.num_actions = 4
        self.tasks = []
        for task_id, t in enumerate(tasks):
            seed = int(t.get("maze_seed", base_seed + task_id))
            rng = np.random.default_rng(seed)
            wall = _generate_maze(size, braid, rng)
            if "wall_penalty" in t:
                wall_penalty = float(t["wall_penalty"])
            elif rng.random() < wp_prob:
                wall_penalty = -float(rng.uniform(0.02, wp_max))
            else:
                wall_penalty = 0.0
            spec = TaskSpec(task_id, f"maze{size}-seed{seed}", t)
            self.tasks.append(MazeTask(spec, gamma, size, wall, slip, goal_reward,
                                       shaping, view_k, max_steps, wall_penalty))
