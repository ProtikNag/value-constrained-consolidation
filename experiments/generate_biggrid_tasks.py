"""Generate and VALIDATE the BigGrid v2 task set (goal-conditioned revision).

    python -m experiments.generate_biggrid_tasks --out configs/biggrid50v2.yaml

The agent sees its own position, THE GOAL'S POSITION, and a small egocentric
window of the walls around it. That combination is the whole design: position
and goal make the direction computable, and the window is what lets the agent
route around the walls in between.

v2 hid the goal (``goal_in_obs: false``). With no goal in the input a direction
cannot be computed, only recalled, so a shared network had to memorise one
navigation field per task keyed off a task one-hot. That inverted every axis:
denser grids scored BETTER (r = +0.63 of density with joint score) because walls
shrink the reachable region and therefore the table to memorise, and widening
the view HURT because 961 patch dims swamped the 100 position dims. Joint
training peaked at 0.36 and no amount of task easing moved it.

Four axes vary independently:

    obstacle density  5% .. 30%       how much of the grid is wall
    slip              0.01 .. 0.10    chance the executed action is random
    path length       5 .. 80 steps   spawn distance band -- the real difficulty
    view radius       3 .. 5          how far the agent can see (egocentric)

Path length is the axis that grades difficulty in the right direction. Spawns
are restricted to a per-task shortest-path band, so a task's distance profile
is a property of the task rather than a by-product of its layout.

The axes are balanced AND verified pairwise near-uncorrelated. v1 shipped with
slip confounded (+0.552 with score) and the benchmark measured the confound, so
this is checked rather than assumed.

EVERYTHING is deterministic given ``--seed``: goals, layouts, slips, view radii
and bands. Re-running writes a byte-identical config.

Solvability is a shortest-path check, not a guess. Obstacles are generated with
the goal's row and column reserved free, a flood fill from the goal gives the
spawnable component, BFS gives the true distance from every spawnable cell, and
a task is rejected unless its band holds enough spawn cells and the
95th-percentile distance fits in half the episode horizon. That slack matters
under ``wall_death``, where one wasted step can end the episode.
"""

from __future__ import annotations

import argparse
import collections
import itertools
from pathlib import Path

import numpy as np
import yaml

from crl.envs.biggrid import _generate_blocked, _reachable_starts

DENSITIES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
SLIPS = [0.01, 0.02, 0.04, 0.06, 0.08, 0.10]
VIEW_RADII = [3, 4, 5]
# Shortest-path distance bands for spawn cells: the difficulty axis.
PATH_BANDS = [(5, 15), (15, 30), (30, 45), (45, 60), (60, 80)]
# A band with too few cells gives a near-deterministic start, which makes the
# task memorisable rather than navigable.
MIN_SPAWN_IN_BAND = 40

AXES = {"density": DENSITIES, "slip": SLIPS,
        "band": PATH_BANDS, "view": VIEW_RADII}


def bfs_distance(blocked: np.ndarray, goal: tuple[int, int]) -> np.ndarray:
    """Shortest collision-free distance from every cell to ``goal`` (-1 = cut off)."""
    size = blocked.shape[0]
    dist = np.full((size, size), -1, np.int32)
    dist[goal] = 0
    queue = collections.deque([goal])
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size and not blocked[nr, nc] \
                    and dist[nr, nc] < 0:
                dist[nr, nc] = dist[r, c] + 1
                queue.append((nr, nc))
    return dist


def pearson(xs, ys) -> float:
    """Pearson r; 0.0 if either series is constant."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    dx, dy = xs.std(), ys.std()
    return float(((xs - xs.mean()) * (ys - ys.mean())).mean() / (dx * dy)) \
        if dx and dy else 0.0


def assign_axes(n_tasks: int, rng, max_abs_r: float = 0.20,
                tries: int = 4000) -> tuple[dict[str, np.ndarray], float]:
    """One level index per task per axis: each axis balanced over its levels, and
    every PAIR of axes near-uncorrelated.

    Independent shuffles decorrelate in expectation, but a single draw can still
    land a confound, so we draw repeatedly and keep the flattest. Returns the
    indices and the worst pairwise |r| achieved.
    """
    def balanced(levels: list) -> list[int]:
        reps = -(-n_tasks // len(levels))      # ceil
        return [i for i in range(len(levels)) for _ in range(reps)][:n_tasks]

    best: tuple[float, dict] | None = None
    for _ in range(tries):
        cols = {k: rng.permutation(balanced(v)) for k, v in AXES.items()}
        worst = max(abs(pearson(cols[a], cols[b]))
                    for a, b in itertools.combinations(cols, 2))
        if best is None or worst < best[0]:
            best = (worst, cols)
        if worst <= max_abs_r:
            return cols, worst
    return best[1], best[0]


def generate(n_tasks: int, size: int, max_steps: int, seed: int,
             min_goal_sep: int = 4,
             easy_tasks: frozenset[int] = frozenset()) -> tuple[list[dict], list[dict], dict]:
    """Return (tasks, per-task validation report, axis diagnostics).

    ``easy_tasks`` holds 1-based task numbers forced to the gentlest density and
    slip. They are the ones a verified specialist could not learn even after the
    reward fix: a benchmark must not contain a task no policy reaches, so those
    instances are replaced rather than kept as permanent zeros. Their BAND is
    preserved, so the path-length axis stays balanced at 10 tasks per band.
    """
    rng = np.random.default_rng(seed)
    cols, worst_r = assign_axes(n_tasks, rng)
    limit = max_steps * 0.5

    tasks: list[dict] = []
    report: list[dict] = []
    goals: list[tuple[int, int]] = []
    for i in range(n_tasks):
        lo, hi = PATH_BANDS[cols["band"][i]]
        view = VIEW_RADII[cols["view"][i]]
        if (i + 1) in easy_tasks:
            density, slip = min(DENSITIES), min(SLIPS)
        else:
            density = DENSITIES[cols["density"][i]]
            slip = SLIPS[cols["slip"][i]]

        # A long band needs a goal far from most of the grid, so the goal is
        # searched FOR the band rather than drawn first and hoped over.
        for attempt in range(20_000):
            gr = int(rng.integers(2, size - 2))
            gc = int(rng.integers(2, size - 2))
            if any(abs(gr - a) + abs(gc - b) < min_goal_sep for a, b in goals):
                continue
            # MUST be the seed BigGridFamily will use (obstacle_seed + task_id),
            # or the grid validated here is not the grid trained on. Counting
            # attempts here instead silently desynchronised the two from the
            # first rejection onward, so solvability was asserted of layouts the
            # env never built. The goal still varies the grid (its row and column
            # are reserved free), so searching over goals is unaffected.
            blocked = _generate_blocked(size, density, (gr, gc), seed + i)
            starts = _reachable_starts(blocked, (gr, gc)).numpy()
            if len(starts) == 0:
                continue
            dist = bfs_distance(blocked.numpy(), (gr, gc))
            d_all = dist[starts[:, 0], starts[:, 1]]
            in_band = (d_all >= lo) & (d_all <= hi)
            if int(in_band.sum()) < MIN_SPAWN_IN_BAND:
                continue
            d = d_all[in_band]
            p95 = float(np.percentile(d, 95))
            if p95 > limit:        # cannot happen while hi <= limit; checked anyway
                continue
            break
        else:
            raise SystemExit(
                f"task {i + 1}: no goal gives {MIN_SPAWN_IN_BAND} spawn cells at "
                f"distance {lo}-{hi} with density {density:.0%} on {size}x{size}. "
                f"Widen the band or lower MIN_SPAWN_IN_BAND.")

        goals.append((gr, gc))
        tasks.append({"goal": [gr, gc], "slip": slip,
                      "obstacle_density": density, "view_radius": view,
                      "start_band": [lo, hi], "threshold": 0.60})
        report.append({"task": i + 1, "goal": (gr, gc), "slip": slip,
                       "density": density, "view": view, "band": [lo, hi],
                       "spawnable": int(in_band.sum()),
                       "reachable": int(len(starts)),
                       "blocked": int(blocked.sum()),
                       "mean_d": float(d.mean()), "p95_d": p95,
                       "max_d": int(d.max()), "min_d": int(d.min())})

    # The confound check, reported so a reader can audit it rather than trust it.
    obs = {"density": [r["density"] for r in report],
           "slip": [r["slip"] for r in report],
           "mean path": [r["mean_d"] for r in report],
           "view": [r["view"] for r in report],
           "spawn cells": [r["spawnable"] for r in report]}
    corr = {f"{a} vs {b}": round(pearson(obs[a], obs[b]), 3)
            for a, b in itertools.combinations(obs, 2)}
    return tasks, report, {"worst_pairwise_r_on_levels": round(worst_r, 3),
                           "realised_correlations": corr}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="configs/biggrid50v2.yaml")
    ap.add_argument("--base", default="configs/biggrid50v2.yaml",
                    help="config to inherit policy/ppo/trainer blocks from. Defaults to "
                         "the output itself: the generated config carries those "
                         "blocks, so regeneration is self-contained on this branch.")
    ap.add_argument("--n-tasks", type=int, default=50)
    ap.add_argument("--size", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=160)
    ap.add_argument("--view-max", type=int, default=max(VIEW_RADII),
                    help="patch radius. The patch is (2r+1)^2 dims and competes "
                         "with the 4*size position+goal dims; at view_max 15 it was "
                         "961 against 100 and swamped them.")
    ap.add_argument("--shape-w", type=float, default=0.05,
                    help="potential-based shaping weight. Its undiscounted sum "
                         "leaves a small POSITIVE residue on a successful episode, "
                         "which rides on top of the capped terminal bonus.")
    ap.add_argument("--slip-wall-penalty", type=float, default=0.0,
                    help="charged when SLIP pushes the agent into a wall (it survives). "
                         "0.0 by MEASUREMENT: at 0.05 a single slip bump cost ~8x the "
                         "whole episode's potential-based shaping on a long task "
                         "(the gamma=0.99 discount leak cancels most of the proximity "
                         "gain), so avoiding bumps dominated reaching the goal. "
                         "Removing it rescued 11 of 15 tasks no specialist could learn "
                         "on two seeds. Raising shape_w instead rescued fewer AND "
                         "pushed episode scores above 1. Deliberate wall entry is "
                         "still fatal -- wall_death is untouched.")
    ap.add_argument("--easy-tasks", type=int, nargs="*", default=[20, 21, 28, 33],
                    help="1-based tasks forced to the gentlest density and slip "
                         "because a verified specialist could not learn them. "
                         "Default is the measured set; pass nothing to disable.")
    ap.add_argument("--seed", type=int, default=20260930)
    args = ap.parse_args()

    tasks, report, diag = generate(args.n_tasks, args.size, args.max_steps, args.seed,
                                   easy_tasks=frozenset(args.easy_tasks or ()))

    print(f"{'task':>5}{'goal':>10}{'slip':>6}{'dens':>6}{'view':>6}{'band':>10}"
          f"{'spawns':>8}{'mean d':>8}{'p95 d':>7}{'max d':>7}")
    for r in report:
        print(f"{r['task']:>5}{str(r['goal']):>10}{r['slip']:>6}{r['density']:>6.2f}"
              f"{r['view']:>6}{str(r['band']):>10}{r['spawnable']:>8}"
              f"{r['mean_d']:>8.1f}{r['p95_d']:>7.0f}{r['max_d']:>7}")
    print()
    print(f"ALL {len(tasks)} tasks validated solvable "
          f"(p95 shortest path <= {args.max_steps // 2} of {args.max_steps} steps, "
          f">= {MIN_SPAWN_IN_BAND} spawn cells in band)")
    print(f"  density    {min(t['obstacle_density'] for t in tasks):.0%}"
          f" - {max(t['obstacle_density'] for t in tasks):.0%}")
    print(f"  slip       {min(t['slip'] for t in tasks)} - {max(t['slip'] for t in tasks)}")
    print(f"  view       {min(t['view_radius'] for t in tasks)}"
          f" - {max(t['view_radius'] for t in tasks)}  (patch "
          f"{(2 * args.view_max + 1) ** 2} dims vs {4 * args.size} position+goal)")
    print(f"  path len   {min(r['min_d'] for r in report)}"
          f" - {max(r['max_d'] for r in report)} steps, "
          f"mean-of-means {np.mean([r['mean_d'] for r in report]):.1f}")
    print(f"  distinct (slip, density, band): "
          f"{len({(t['slip'], t['obstacle_density'], tuple(t['start_band'])) for t in tasks})}")
    print(f"\n  axis independence (worst |r| on levels: "
          f"{diag['worst_pairwise_r_on_levels']})")
    for k, v in diag["realised_correlations"].items():
        flag = "  <-- CONFOUNDED" if abs(v) > 0.35 else ""
        print(f"    {k:<28}{v:>7.3f}{flag}")

    cfg = yaml.safe_load(Path(args.base).read_text())
    cfg["experiment"]["name"] = Path(args.out).stem
    cfg["env"]["params"] = {
        "size": args.size, "gamma": 0.99, "max_steps": args.max_steps,
        # The agent must be able to COMPUTE a direction, not recall one.
        "goal_in_obs": True,
        "obstacle_seed": args.seed,
        "reward_mode": "goal_time", "shape_w": args.shape_w,
        "wall_death": True, "view_max": args.view_max,
        "slip_wall_penalty": args.slip_wall_penalty,
    }
    cfg["env"]["tasks"] = tasks
    Path(args.out).write_text(
        "# === BigGrid v2 -- GENERATED, do not hand-edit ===\n"
        f"# python -m experiments.generate_biggrid_tasks --out {args.out}\n"
        "#\n"
        "# goal_in_obs TRUE: the observation is the agent's position one-hot, the\n"
        "# GOAL's position one-hot, and an egocentric wall patch. Direction is\n"
        "# computable from the input; v2 hid the goal and the task became memory.\n"
        "# reward_mode goal_time: terminal bonus (1 - steps/max_steps) for REACHING\n"
        "# the goal, plus potential-based shaping. The shaping telescopes, so a\n"
        "# random policy nets ~0; it is path-independent, so detours cost nothing.\n"
        "# wall_death: entering a wall ENDS the episode.\n"
        "# view_radius: egocentric window, per task. The patch is always\n"
        "# (2*view_max+1)^2 so the input width is shared; narrower tasks leave the\n"
        "# surplus at 0, a third symbol distinct from 'free'.\n"
        "# start_band: spawns are restricted to this shortest-path distance band,\n"
        "# which makes PATH LENGTH the difficulty axis. Density is not one: walls\n"
        "# shrink the reachable region, so dense grids are easier.\n"
        "#\n"
        "# Every task validated solvable at generation (see the module docstring).\n"
        + yaml.safe_dump(cfg, sort_keys=False))

    import hashlib, json
    cfg_text = Path(args.out).read_text()
    manifest = {
        "generator": "experiments/generate_biggrid_tasks.py",
        "command": f"python -m experiments.generate_biggrid_tasks --out {args.out} "
                   f"--seed {args.seed}",
        "seed": args.seed,
        "config": args.out,
        "config_sha256": hashlib.sha256(cfg_text.encode()).hexdigest(),
        "deterministic": "re-running with the same --seed writes a byte-identical "
                         "config; grids, goals, slips, view radii and bands never change",
        "env": cfg["env"]["params"],
        "axes": {"obstacle_density": DENSITIES, "slip": SLIPS,
                 "view_radius": VIEW_RADII, "start_band": PATH_BANDS},
        "axis_independence": diag,
        "replaced_tasks": {
            "tasks": sorted(args.easy_tasks or ()),
            "why": "no specialist reached 0.6 on these instances even with the "
                   "slip-penalty fix; regenerated at the gentlest density and slip "
                   "so every shipped task is learnable",
        },
        "n_tasks": len(tasks),
        "validation": {
            "rule": "95th-percentile collision-free shortest path from a spawnable "
                    "cell in the band must fit in half the episode horizon, and the "
                    "band must hold at least MIN_SPAWN_IN_BAND cells",
            "horizon": args.max_steps,
            "limit": args.max_steps // 2,
            "min_spawn_in_band": MIN_SPAWN_IN_BAND,
            "worst_case_shortest_path": max(r["max_d"] for r in report),
        },
        "tasks": report,
    }
    mpath = Path(args.out).with_suffix(".manifest.json")
    mpath.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    print(f"wrote {mpath}  (config sha256 {manifest['config_sha256'][:16]}...)")


if __name__ == "__main__":
    main()
