"""Appendix figures for GridWorld-50 (BigGrid v2): task gallery and greedy rollouts.

    .venv/bin/python reports/final/gridworld_v2/make_appendix_env_figures.py [gallery|traj]

Writes reports/final/gridworld_v2/appendix/{tasks_gallery,trajectories}.{pdf,png,svg}
and copies the PDFs to paper/figures/appendix/gridworld/. Specialists for the
trajectory figure are trained once (PPO, 700 iters, n_envs 32, n_steps 128,
seed 0, as in the original analysis script) and cached under appendix/cache/.
Every cell is a vector quad (pcolormesh), so the PDFs embed no raster images.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "report"))
sys.path.insert(0, str(REPO))
from paperfig import AC, TEXTWIDTH, save, style  # noqa: E402

from crl.config import load_config  # noqa: E402
from crl.envs import make_family  # noqa: E402

OUT = REPO / "reports/final/gridworld_v2/appendix"
CACHE = OUT / "cache"
PAPER = REPO / "paper/figures/appendix/gridworld"

GALLERY = [3, 6, 13, 21, 22, 47]   # 1-indexed task numbers, in sequence order
TRAJ = [(6, "(a) Long path, task 6"), (3, "(b) Short path, task 3")]
N_ROLLOUTS = 12

# Cell categories, drawn as flat vector quads. 0 (free, outside the spawn band)
# is masked and shows the white axes background.
WALL, BAND, POCKET = 1, 2, 3
CELL_COLORS = ListedColormap(["#FFFFFF", AC["axis"], "#DBEAFE", "#E9ECEF"])
WINDOW = AC["teal"]
# Failures drawn above successes so they are never buried.
Z = {"goal": 3.0, "timeout": 3.2, "wall": 3.4}
OUTCOME = {"goal": AC["green"], "wall": AC["red"], "timeout": AC["amber"]}

cfg = load_config(str(REPO / "configs/biggrid50v2.yaml"))
fam = make_family(cfg.env)
SIZE = cfg.env.params["size"]


def np_(x):
    """Tensor or array to numpy."""
    return x.numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def cell_map(task) -> np.ndarray:
    """[size, size] category map: walls, spawn band, unreachable free pockets."""
    blocked = np_(task.blocked).astype(bool)
    dist = np_(task.goal_dist)
    cat = np.zeros((SIZE, SIZE), dtype=int)
    cat[(dist < 0) & ~blocked] = POCKET
    starts = np_(task.starts)
    cat[starts[:, 0], starts[:, 1]] = BAND
    cat[blocked] = WALL
    return cat


def draw_cells(ax, task, extent=None) -> None:
    """Grid cells as a flat pcolormesh (vector); ``extent`` = (r0, r1, c0, c1)."""
    cat = cell_map(task)
    r0, r1, c0, c1 = extent or (0, SIZE, 0, SIZE)
    sub = np.ma.masked_equal(cat[r0:r1, c0:c1], 0)
    # edgecolors="face" with a hairline closes the seams viewers draw between quads.
    ax.pcolormesh(np.arange(c0, c1 + 1) - 0.5, np.arange(r0, r1 + 1) - 0.5, sub,
                  cmap=CELL_COLORS, vmin=-0.5, vmax=3.5, shading="flat",
                  edgecolors="face", linewidth=0.15, antialiased=False, zorder=1)
    ax.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), c1 - c0, r1 - r0, fill=False,
                           edgecolor=AC["axis"], linewidth=0.6, zorder=8))
    ax.set_xlim(c0 - 0.5, c1 - 0.5)
    ax.set_ylim(r1 - 0.5, r0 - 0.5)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_goal(ax, task, size=95) -> None:
    gr, gc = task.goal
    ax.scatter([gc], [gr], s=size, marker="*", c=AC["red"], edgecolor="white",
               linewidths=0.7, zorder=9)


def params_line(i: int) -> str:
    """Compact per-task parameter line."""
    t, s = fam.tasks[i - 1], cfg.env.tasks[i - 1]
    lo, hi = s["start_band"]
    return (f"r {t.view_radius}  ·  slip {s['slip']:.2f}  ·  "
            f"walls {s['obstacle_density']:.0%}  ·  path {lo}–{hi}")


# --------------------------------------------------------------------- gallery
def gallery() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(TEXTWIDTH, 4.85),
                             gridspec_kw=dict(wspace=0.07, hspace=0.26))
    rng = np.random.default_rng(0)
    for ax, i in zip(axes.flat, GALLERY):
        task = fam.tasks[i - 1]
        draw_cells(ax, task)
        starts = np_(task.starts)
        k = task.view_radius
        # The agent: a sampled start whose window lies fully inside the grid.
        inside = starts[(starts.min(1) >= k) & (starts.max(1) < SIZE - k)]
        agent = inside[rng.integers(len(inside))]
        pick = starts[rng.choice(len(starts), size=min(14, len(starts)), replace=False)]
        ax.scatter(pick[:, 1], pick[:, 0], s=7, facecolor="white",
                   edgecolor=AC["amber"], linewidths=0.75, zorder=5)
        ax.add_patch(Rectangle((agent[1] - k - 0.5, agent[0] - k - 0.5), 2 * k + 1,
                               2 * k + 1, facecolor=WINDOW + "2E", edgecolor=WINDOW,
                               linewidth=1.3, zorder=6))
        ax.scatter([agent[1]], [agent[0]], s=16, c=WINDOW, edgecolor="white",
                   linewidths=0.6, zorder=7)
        draw_goal(ax, task, size=120)
        ax.text(0.0, 1.075, f"Task {i}", transform=ax.transAxes, fontsize=9,
                fontweight="bold", color=AC["text"], ha="left", va="bottom")
        ax.text(0.0, 1.012, params_line(i), transform=ax.transAxes, fontsize=7,
                color=AC["muted"], ha="left", va="bottom")
    handles = [
        Patch(facecolor=AC["axis"], label="wall"),
        Patch(facecolor="#DBEAFE", edgecolor="#93C5FD", linewidth=0.5,
              label="distance band"),
        Patch(facecolor="#E9ECEF", edgecolor=AC["faint"], linewidth=0.5,
              label="unreachable"),
        Line2D([], [], marker="*", ls="", ms=9, mfc=AC["red"], mec="white",
               mew=0.6, label="goal"),
        Line2D([], [], marker="o", ls="", ms=3.6, mfc="white", mec=AC["amber"],
               mew=0.8, label="sampled start"),
        Line2D([], [], marker="o", ls="", ms=4.5, mfc=WINDOW, mec="white",
               mew=0.5, label="agent"),
        Patch(facecolor=WINDOW + "2E", edgecolor=WINDOW, linewidth=1.2,
              label="view window"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=7, bbox_to_anchor=(0.5, 0.0),
               handlelength=1.3, handletextpad=0.45, columnspacing=1.1,
               borderaxespad=0.0, fontsize=7.8)
    fig.subplots_adjust(left=0.0, right=1.0, top=0.95, bottom=0.075)
    save(fig, OUT / "tasks_gallery", paper_pdf=PAPER / "tasks_gallery.pdf")


# ---------------------------------------------------------------- trajectories
def specialist(i: int):
    """PPO specialist for task ``i`` (1-indexed), trained once and cached."""
    from crl.logging_utils import RunLogger
    from crl.policies import make_policy
    from crl.ppo.trainer import LocalTrainer
    from crl.seeding import set_seed

    dev = torch.device("cpu")
    set_seed(0)
    pol = make_policy(cfg.policy, fam).to(dev)
    path = CACHE / f"specialist_task{i:02d}.pt"
    if path.exists():
        pol.load_state_dict(torch.load(path, map_location=dev))
        return pol
    cfg.ppo.n_envs, cfg.ppo.n_steps, cfg.ppo.stop_eval_every = 32, 128, 0
    logger = RunLogger(str(CACHE / "logs"), f"task{i:02d}", {})
    LocalTrainer(cfg.ppo, dev, logger, 10**9).train(
        pol, fam.tasks[i - 1], num_iters=700, seed=0, current_task=1,
        phase_type="local")
    logger.close()
    CACHE.mkdir(parents=True, exist_ok=True)
    torch.save(pol.state_dict(), path)
    return pol


def rollouts(i: int, pol) -> list[tuple[np.ndarray, str]]:
    """Greedy episodes from seeds 500..511 (same seeds as the original figure)."""
    env = fam.tasks[i - 1].make_env()
    out = []
    for ep in range(N_ROLLOUTS):
        obs, _ = env.reset(seed=500 + ep)
        path, how = [(env._row, env._col)], "timeout"
        for _ in range(cfg.env.params["max_steps"]):
            with torch.no_grad():
                a = int(pol.dist(torch.as_tensor(obs).unsqueeze(0), i - 1).logits[0].argmax())
            obs, _, term, trunc, info = env.step(a)
            path.append((env._row, env._col))
            if term or trunc:
                how = "wall" if info.get("wall_death") else ("goal" if term else "timeout")
                break
        out.append((np.array(path, float), how))
    env.close()
    return out


def zoom_box(paths, goal, margin=3) -> tuple[int, int, int, int]:
    """Square (r0, r1, c0, c1) around all paths and the goal, clipped to the grid."""
    pts = np.vstack([p for p, _ in paths] + [np.array([goal], float)])
    lo, hi = pts.min(0) - margin, pts.max(0) + margin + 1
    side = min(SIZE, int(max(hi - lo)))
    centre = (lo + hi) / 2
    start = np.clip(np.round(centre - side / 2), 0, SIZE - side).astype(int)
    return start[0], start[0] + side, start[1], start[1] + side


def trajectories() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH * 0.86, 3.25),
                             gridspec_kw=dict(wspace=0.08))
    jitter = np.linspace(-0.17, 0.17, N_ROLLOUTS)
    order = np.random.default_rng(1).permutation(N_ROLLOUTS)
    for ax, (i, label) in zip(axes, TRAJ):
        task = fam.tasks[i - 1]
        paths = rollouts(i, specialist(i))
        draw_cells(ax, task, zoom_box(paths, task.goal))
        for n, (p, how) in enumerate(paths):
            off = jitter[order[n]]
            xy = np.column_stack([p[:, 1] + off, p[:, 0] + off])
            col = OUTCOME[how]
            ax.plot(xy[:, 0], xy[:, 1], "-", c=col, lw=0.9, alpha=0.9,
                    solid_capstyle="round", solid_joinstyle="round", zorder=Z[how])
            # Direction cue: one small arrowhead halfway along the episode.
            m = max(1, len(xy) // 2)
            if len(xy) > 2 and not np.allclose(xy[m - 1], xy[m]):
                ax.add_patch(FancyArrowPatch(xy[m - 1], xy[m], arrowstyle="-|>",
                                             mutation_scale=6, color=col, lw=0,
                                             zorder=Z[how] + 0.05))
            ax.scatter(xy[0, 0], xy[0, 1], s=10, c=AC["text"], edgecolor="white",
                       linewidths=0.5, zorder=5)
            if how == "wall":
                ax.scatter(xy[-1, 0], xy[-1, 1], s=22, marker="x", c=col,
                           linewidths=1.3, zorder=6)
            elif how == "timeout":
                ax.scatter(xy[-1, 0], xy[-1, 1], s=18, facecolor="white", edgecolor=col,
                           linewidths=1.1, zorder=6)
        draw_goal(ax, task, size=130)
        counts = {h: sum(how == h for _, how in paths) for h in OUTCOME}
        ax.annotate(f"{counts['goal']} reached · {counts['wall']} wall · "
                    f"{counts['timeout']} timeout", xy=(1.0, 1.0), xycoords="axes fraction",
                    xytext=(0, 4), textcoords="offset points", fontsize=7.2,
                    color=AC["muted"], ha="right", va="baseline")
        ax.set_title(label, fontsize=9, loc="left", pad=4, va="baseline")
        print(f"task {i}: {counts}")
    handles = [
        Line2D([], [], color=OUTCOME["goal"], lw=1.5, label="reached the goal"),
        Line2D([], [], color=OUTCOME["wall"], lw=1.5, marker="x", ms=4.5, mew=1.2,
               label="died on a wall"),
        Line2D([], [], color=OUTCOME["timeout"], lw=1.5, marker="o", ms=4, mfc="white",
               mew=1.0, label="ran out of time"),
        Line2D([], [], marker="o", ls="", ms=3.8, mfc=AC["text"], mec="white",
               mew=0.4, label="start"),
        Line2D([], [], marker="*", ls="", ms=9, mfc=AC["red"], mec="white", mew=0.6,
               label="goal"),
        Patch(facecolor=AC["axis"], label="wall"),
        Patch(facecolor="#DBEAFE", edgecolor="#93C5FD", linewidth=0.5,
              label="distance band"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=7, bbox_to_anchor=(0.5, 0.0),
               handlelength=1.5, handletextpad=0.4, columnspacing=0.9,
               borderaxespad=0.0, fontsize=7.5)
    fig.subplots_adjust(left=0.0, right=1.0, top=0.93, bottom=0.1)
    save(fig, OUT / "trajectories", paper_pdf=PAPER / "trajectories.pdf")


if __name__ == "__main__":
    style()
    which = sys.argv[1:] or ["gallery", "traj"]
    if which[0] == "train":          # e.g. `train 6`: populate the cache only
        specialist(int(which[1]))
    if "gallery" in which:
        gallery()
    if "traj" in which:
        trajectories()
