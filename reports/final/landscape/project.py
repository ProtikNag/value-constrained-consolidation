#!/usr/bin/env python3
"""Project one whole task boundary and measure the surface it sits on.

A task boundary is two phases, not one. The specialist is cloned from the
global policy carried in from the previous task and trained on the new task
alone; the global policy is then consolidated from that same starting point
under the value constraint. Both phases therefore leave from one point, and
this builds a single plane for both of them.

The plane is the span of the two leading principal components of the two
recorded paths taken together, about their shared origin. It is not chosen to
contain any policy of interest. The share of the paths' variance it carries is
reported, and so is how far off it each landmark sits, because a point far off
the plane must not be read as if it lay in it.

Two fields are measured over the plane. The mean score on the earlier tasks is
what consolidation maximizes. The shortfall against the specialist on the new
task is what the constraint bounds.

    python reports/final/landscape/project.py --task 5 --n 31

Writes ``data/surface_k<k>.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
sys.path.insert(0, str(REPO))

from crl.config import load_config               # noqa: E402
from crl.envs import make_family                 # noqa: E402
from crl.policies import make_policy             # noqa: E402
from crl.ppo.evaluate import evaluate_value_and_score  # noqa: E402

RUN = REPO / "results" / "duel_v2_matched_seed0"
DATA = HERE / "data"
N_ENVS, SEED = 8, 4242
N_EPISODES = 12   # overridden by --episodes
MARGIN = 0.22          # fraction of the spanned extent added around the paths


def principal_plane(points: np.ndarray, origin: np.ndarray):
    """Top two principal directions about ``origin``, and the variance share."""
    centred = points - origin
    # ~200 points in ~1.2M dimensions, so the Gram matrix is tiny and gives the
    # same directions as an SVD of the full matrix.
    gram = centred @ centred.T
    evals, evecs = np.linalg.eigh(gram)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    total = float(evals.clip(min=0).sum())
    dirs = []
    for j in (0, 1):
        d = centred.T @ evecs[:, j]
        dirs.append(d / np.linalg.norm(d))
    explained = float(evals[:2].clip(min=0).sum() / total) if total > 0 else 0.0
    return dirs[0], dirs[1], explained


def project(points: np.ndarray, origin: np.ndarray, d1: np.ndarray,
            d2: np.ndarray):
    """2D coordinates in the plane, and each point's distance off it."""
    centred = np.atleast_2d(points - origin)
    a, b = centred @ d1, centred @ d2
    resid = np.sqrt(np.clip((centred * centred).sum(axis=1) - a * a - b * b,
                            0, None))
    return np.stack([a, b], axis=1), resid


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", type=int, default=5)
    ap.add_argument("--n", type=int, default=31, help="grid resolution")
    ap.add_argument("--episodes", type=int, default=12,
                    help="greedy episodes per task per grid point")
    args = ap.parse_args()
    k = args.task
    global N_EPISODES
    N_EPISODES = args.episodes

    phases = {}
    for name in ("global", "local"):
        path = DATA / f"traj_k{k}_{name}.npz"
        if not path.exists():
            raise SystemExit(f"missing {path}; run run_trajectory.py --phase {name}")
        blob = np.load(path)
        phases[name] = dict(
            iterates=blob["iterates"].astype(np.float64),
            iters=blob["iters"],
            meta=json.loads(bytes(blob["meta"].tolist()).decode()),
            mu_step=blob["mu_step"], mu=blob["mu"], constraint=blob["constraint"])

    # Both phases leave from the global policy carried in from task k-1.
    origin = phases["global"]["iterates"][0]
    gap = np.linalg.norm(phases["local"]["iterates"][0] - origin)
    if gap > 1e-4:
        raise SystemExit(f"the two phases do not share a starting point "
                         f"(distance {gap:.3e}); check local_init_fresh")

    stacked = np.concatenate([phases["global"]["iterates"],
                              phases["local"]["iterates"]], axis=0)
    d1, d2, explained = principal_plane(stacked, origin)
    print(f"task {k}: {len(stacked)} iterates over both phases, "
          f"top two components carry {explained:.1%} of their variance")

    cfg = load_config(RUN / "config.yaml")
    family = make_family(cfg.env)
    ref = json.loads((RUN / "run.json").read_text())["reference"]
    device = torch.device("cpu")
    policy = make_policy(cfg.policy, family).to(device)
    base = policy.state_dict()
    keys = phases["global"]["meta"]["keys"]
    shapes = [(key, base[key].shape, base[key].numel()) for key in keys]
    ref_current = phases["global"]["meta"]["ref_current"]

    def set_params(vec: np.ndarray) -> None:
        out, i = {}, 0
        tensor = torch.from_numpy(np.ascontiguousarray(vec))
        for key, shape, n in shapes:
            out[key] = tensor[i:i + n].view(shape).float()
            i += n
        policy.load_state_dict(out, strict=False)

    current, past = k - 1, list(range(k - 1))

    def measure():
        """(shortfall against the specialist on task k, mean score on earlier)."""
        value, score, _, _ = evaluate_value_and_score(
            policy, family.tasks[current], num_episodes=N_EPISODES,
            n_envs=N_ENVS, device=device, seed=SEED, greedy=True)
        lo, hi = ref["random"][current], ref["ceiling"][current]
        new_score = (score - lo) / (hi - lo) if hi > lo else score
        scores = []
        for i in past:
            _, s, _, _ = evaluate_value_and_score(
                policy, family.tasks[i], num_episodes=N_EPISODES, n_envs=N_ENVS,
                device=device, seed=SEED, greedy=True)
            lo, hi = ref["random"][i], ref["ceiling"][i]
            scores.append((s - lo) / (hi - lo) if hi > lo else s)
        return max(0.0, ref_current - value), sum(scores) / len(scores), new_score

    coords = {name: project(p["iterates"], origin, d1, d2)
              for name, p in phases.items()}
    allc = np.concatenate([c for c, _ in coords.values()], axis=0)
    lo1, hi1 = allc[:, 0].min(), allc[:, 0].max()
    lo2, hi2 = allc[:, 1].min(), allc[:, 1].max()
    pad1, pad2 = (hi1 - lo1) * MARGIN, (hi2 - lo2) * MARGIN
    xs = np.linspace(lo1 - pad1, hi1 + pad1, args.n)
    ys = np.linspace(lo2 - pad2, hi2 + pad2, args.n)

    z_past, z_short, z_new = [], [], []
    start = time.time()
    for j, y in enumerate(ys):
        rp, rs, rn = [], [], []
        for x in xs:
            set_params(origin + x * d1 + y * d2)
            shortfall, past_score, new_score = measure()
            rs.append(shortfall)
            rp.append(past_score)
            rn.append(new_score)
        z_past.append(rp)
        z_short.append(rs)
        z_new.append(rn)
        done = time.time() - start
        print(f"  row {j + 1}/{args.n}  {done:.0f}s, "
              f"~{done / (j + 1) * (args.n - j - 1):.0f}s left", flush=True)

    out_paths = {}
    for name, p in phases.items():
        xy, resid = coords[name]
        pts = []
        for idx, vec in enumerate(p["iterates"]):
            set_params(vec)
            shortfall, past_score, new_score = measure()
            pts.append(dict(iter=int(p["iters"][idx]), x=float(xy[idx, 0]),
                            y=float(xy[idx, 1]), off_plane=float(resid[idx]),
                            shortfall=shortfall, past_score=past_score,
                            new_score=new_score))
        out_paths[name] = pts

    # Where the original run's own checkpoints fall, as landmarks. They are a
    # different run instance, so each one's distance off the plane is recorded
    # and the figure must not place one that does not lie in it.
    landmarks = {}
    for fname, label in ((f"global_after_task{k - 1}.pt", "incoming_global"),
                         (f"local_after_task{k}.pt", "specialist"),
                         (f"global_after_task{k}.pt", "consolidated_original")):
        sd = torch.load(RUN / fname, map_location=device, weights_only=False)
        vec = torch.cat([sd[key].flatten().double()
                         for key in keys]).numpy()
        xy, resid = project(vec, origin, d1, d2)
        set_params(vec)
        shortfall, past_score, new_score = measure()
        landmarks[label] = dict(x=float(xy[0, 0]), y=float(xy[0, 1]),
                                off_plane=float(resid[0]),
                                dist=float(np.linalg.norm(vec - origin)),
                                shortfall=shortfall, past_score=past_score,
                                new_score=new_score, file=fname)

    meta = phases["global"]["meta"]
    out = DATA / f"surface_k{k}.json"
    out.write_text(json.dumps(dict(
        task=k, n_past=meta["n_past"], eps=meta["eps"], ref_current=ref_current,
        explained_variance=explained, run=meta["run"], n_episodes=N_EPISODES,
        seed=SEED, grid_n=args.n,
        iters={name: p["meta"]["n_iters"] for name, p in phases.items()},
        xs=xs.tolist(), ys=ys.tolist(),
        Z_past_score=z_past, Z_shortfall=z_short, Z_new_score=z_new,
        paths=out_paths, landmarks=landmarks,
        mu_step=phases["global"]["mu_step"].tolist(),
        mu=phases["global"]["mu"].tolist(),
        constraint=[None if np.isnan(v) else float(v)
                    for v in phases["global"]["constraint"]],
    ), indent=1))
    print(f"wrote {out}")
    print("\nlandmarks (off-plane distance matters: a large one must not be drawn)")
    for label, m in landmarks.items():
        print(f"  {label:<24} ({m['x']:+7.2f}, {m['y']:+7.2f})  "
              f"off-plane {m['off_plane']:6.2f} of {m['dist']:6.2f}  "
              f"new {m['new_score']:+.2f}  earlier {m['past_score']:+.2f}")


if __name__ == "__main__":
    main()
