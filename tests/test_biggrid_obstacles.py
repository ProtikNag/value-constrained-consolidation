"""BigGrid obstacle + per-task-dynamics checks (Lane A hard 50-task env).

Guards the three correctness properties the hard family relies on:
  * obstacle layouts always leave a Manhattan path (goal row+col reserved free),
  * agents never spawn in or step onto a blocked cell,
  * density 0 reproduces the original obstacle-free behaviour (back-compat).
"""

from collections import deque

import torch

from crl.envs.biggrid import BigGridFamily, _generate_blocked, _reachable_starts
from crl.policies.mlp import MultiHeadMLPPolicy


def _reaches_goal(blocked: torch.Tensor, start, goal) -> bool:
    S = blocked.shape[0]
    seen = torch.zeros_like(blocked)
    q = deque([tuple(start)])
    seen[tuple(start)] = True
    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            return True
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < S and 0 <= nc < S and not blocked[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return False


def test_obstacle_mask_density_and_every_start_solvable():
    size, goal = 30, (7, 11)
    for density in (0.05, 0.15, 0.30):
        b = _generate_blocked(size, density, goal, seed=1)
        assert not b[goal[0]].any() and not b[:, goal[1]].any()  # goal row/col free
        assert not b[goal]
        assert abs(b.float().mean().item() - density) < 0.02  # ~requested density
        starts = _reachable_starts(b, goal)
        assert len(starts) > 0
        # Every spawn cell can reach the goal (spot-check a sample for speed).
        idx = torch.randperm(len(starts))[:40]
        for s in starts[idx]:
            assert _reaches_goal(b, (int(s[0]), int(s[1])), goal)


def test_agents_never_occupy_walls_and_slip_override():
    fam = BigGridFamily(
        {"size": 20, "slip": 0.1, "max_steps": 60, "goal_in_obs": False,
         "obstacle_seed": 3, "obstacle_density": 0.12},
        [{"goal": [2, 2]}, {"goal": [17, 17], "slip": 0.3, "obstacle_density": 0.2}],
    )
    assert fam.tasks[1].slip == 0.3  # per-task override honoured
    pol = MultiHeadMLPPolicy(fam.obs_dim, fam.num_actions, [64], num_tasks=len(fam))
    for t in fam.tasks:
        for e in t.vector_rollout(pol, 32):
            if len(e.obs) == 0:
                continue
            rows = e.obs[:, :20].argmax(1)
            cols = e.obs[:, 20:40].argmax(1)
            assert not t.blocked[rows, cols].any()


def test_density_zero_is_backward_compatible():
    fam = BigGridFamily({"size": 20}, [{"goal": [2, 2]}])
    assert fam.tasks[0].blocked is not None and not fam.tasks[0].blocked.any()
