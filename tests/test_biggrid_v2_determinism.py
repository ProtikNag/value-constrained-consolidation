"""The v2 dataset must be FIXED. Only the learning algorithm may vary.

If any of these fail, two runs are not comparable, because they were not
solving the same tasks.
"""

from __future__ import annotations

import numpy as np
import torch

from crl.config import load_config
from experiments.generate_biggrid_tasks import VIEW_RADII
from crl.envs import make_family

CFG = "configs/biggrid50v2.yaml"


def _family():
    return make_family(load_config(CFG).env)


def test_grids_are_identical_across_constructions() -> None:
    """Obstacle layouts are a property of the task, not of when it was built."""
    a, b = _family(), _family()
    for ta, tb in zip(a.tasks, b.tasks):
        assert ta.goal == tb.goal
        assert ta.slip == tb.slip and ta.view_radius == tb.view_radius
        assert torch.equal(ta.blocked, tb.blocked)
        assert torch.equal(ta.goal_dist, tb.goal_dist)


def test_spawns_repeat_for_the_same_seed() -> None:
    """Same seed => same start cell, every time."""
    task = _family().tasks[0]
    e1, e2 = task.make_env(), task.make_env()
    for seed in (0, 7, 123):
        o1, _ = e1.reset(seed=seed)
        o2, _ = e2.reset(seed=seed)
        assert np.array_equal(o1, o2)
    e1.close(); e2.close()


def test_trajectories_repeat_for_the_same_seed() -> None:
    """Same seed and same actions => identical transitions, rewards and endings.

    This is the one that catches slip being drawn from an unseeded stream.
    """
    task = _family().tasks[22]          # a task with non-zero slip
    assert task.slip > 0
    runs = []
    for _ in range(2):
        env = task.make_env()
        obs, _ = env.reset(seed=99)
        rng = np.random.default_rng(4)   # identical action sequence both times
        trace = []
        for _ in range(160):
            o, r, term, trunc, info = env.step(int(rng.integers(4)))
            trace.append((float(r), bool(term), bool(trunc), tuple(sorted(info))))
            if term or trunc:
                break
        env.close()
        runs.append(trace)
    assert runs[0] == runs[1]


def test_visibility_is_a_fixed_per_task_property() -> None:
    """Radius comes from the config, never from a draw at episode time."""
    fam = _family()
    radii = [t.view_radius for t in fam.tasks]
    # Read the axis off the generator rather than hard-coding it, so easing a
    # range cannot silently leave this test asserting the previous design.
    assert set(radii) <= set(VIEW_RADII), f"{sorted(set(radii))} vs {VIEW_RADII}"
    assert radii == [t.view_radius for t in _family().tasks]


def test_spawns_respect_each_task_s_path_length_band() -> None:
    """Path length is the difficulty axis, so it must hold at runtime and not
    only in the generator's report."""
    fam = _family()
    for t in fam.tasks:
        lo, hi = t.spec.params["start_band"]
        starts = t.starts.numpy()
        d = t.goal_dist.numpy()[starts[:, 0], starts[:, 1]]
        assert d.min() >= lo and d.max() <= hi, (
            f"task {t.spec.task_id + 1}: spawn distances {d.min()}-{d.max()} "
            f"outside band {lo}-{hi}")


def test_generator_validates_the_grid_the_env_actually_builds() -> None:
    """The layout seed must match BigGridFamily's obstacle_seed + task_id. When
    it counted generator attempts instead, a rejected candidate desynchronised
    the two and solvability was asserted of grids never trained on."""
    import numpy as np
    from crl.envs.biggrid import _generate_blocked
    cfg = load_config(CFG)
    seed = cfg.env.params["obstacle_seed"]
    size = cfg.env.params["size"]
    fam = _family()                      # once: building it is the expensive part
    for i, t in enumerate(cfg.env.tasks):
        want = _generate_blocked(size, t["obstacle_density"],
                                 tuple(t["goal"]), seed + i).numpy()
        got = fam.tasks[i].blocked.numpy()
        assert np.array_equal(want, got), f"task {i + 1}: grid differs from env's" 


def test_every_task_is_solvable_within_the_horizon() -> None:
    """The generator's validation rule, re-checked against the shipped config."""
    fam = _family()
    horizon = load_config(CFG).env.params["max_steps"]
    for t in fam.tasks:
        starts = t.starts.numpy()
        d = t.goal_dist.numpy()[starts[:, 0], starts[:, 1]]
        assert d.min() > 0
        assert float(np.percentile(d, 95)) <= horizon * 0.5


def test_deliberate_wall_kills_and_slip_bump_does_not() -> None:
    """The asymmetry the benchmark rests on."""
    fam = _family()
    task = next(t for t in fam.tasks if t.slip > 0)
    env = task.make_env()
    deaths = bumps = 0
    rng = np.random.default_rng(0)
    for i in range(200):
        env.reset(seed=i)
        for _ in range(160):
            _, _, term, trunc, info = env.step(int(rng.integers(4)))
            bumps += bool(info.get("slip_bump"))
            if term or trunc:
                deaths += bool(info.get("wall_death"))
                break
    env.close()
    assert deaths > 0, "a random policy should walk into walls and die"
    assert bumps > 0, "slip should sometimes bump a wall without killing"
