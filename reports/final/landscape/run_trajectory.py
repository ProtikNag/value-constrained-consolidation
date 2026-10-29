#!/usr/bin/env python3
"""Re-run one consolidation phase and record the optimizer's actual path.

The earlier plane figure was spanned by three hand-picked policies, which made
it circular: the consolidated policy lay in the plane by construction. This
does not pick a plane. It records every iterate the consolidation phase passes
through and takes the plane afterwards from the two leading principal
components of that path, which is the standard way a trajectory is projected
(Goodfellow et al. 2015; Li et al. 2018). The surface is then measured on that
plane, and the path is drawn on the surface it was actually produced on.

Nothing about the optimisation is changed. The phase is restarted from the
checkpoints the original run wrote, with that run's own hyperparameters, and
the only addition is a probe that snapshots parameters and reads the dual
state.

    python reports/final/landscape/run_trajectory.py --task 5

Writes ``data/traj_k<k>.npz`` (iterates, mu, shortfall) for ``project.py``.
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
from crl.duals import make_dual                  # noqa: E402
from crl.envs import make_family                 # noqa: E402
from crl.logging_utils import RunLogger          # noqa: E402
from crl.policies import make_policy             # noqa: E402
from crl.ppo.evaluate import evaluate_value_and_score  # noqa: E402
from crl.ppo.trainer import GlobalTrainer, LocalTrainer  # noqa: E402

RUN = REPO / "results" / "duel_v2_matched_seed0"
DATA = HERE / "data"


def clone_frozen(policy):
    """A detached copy that never receives gradients."""
    import copy
    out = copy.deepcopy(policy)
    for p in out.parameters():
        p.requires_grad_(False)
    out.eval()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", type=int, default=5, help="1-based task index")
    ap.add_argument("--stride", type=int, default=5,
                    help="snapshot every N iterations")
    ap.add_argument("--iters", type=int, default=0,
                    help="override the config's global_iters (0 keeps it)")
    ap.add_argument("--phase", choices=("global", "local"), default="global",
                    help="which phase of the task boundary to record")
    ap.add_argument("--log-every", type=int, default=5,
                    help="the dual is refreshed every constraint_every steps; "
                         "log at least that often or the trace is sparse")
    args = ap.parse_args()

    k = args.task
    cfg = load_config(RUN / "config.yaml")
    family = make_family(cfg.env)
    device = torch.device("cpu")
    DATA.mkdir(parents=True, exist_ok=True)

    # Rebuild the exact state the original phase started from.
    global_policy = make_policy(cfg.policy, family).to(device)
    global_policy.load_state_dict(
        torch.load(RUN / f"global_after_task{k - 1}.pt", map_location=device,
                   weights_only=False), strict=False)
    specialist = make_policy(cfg.policy, family).to(device)
    specialist.load_state_dict(
        torch.load(RUN / f"local_after_task{k}.pt", map_location=device,
                   weights_only=False), strict=False)
    specialist = clone_frozen(specialist)

    task_k = family.tasks[k - 1]
    if args.phase == "local" and cfg.ppo.local_init_fresh:
        raise SystemExit("local_init_fresh is set: the specialist does not start "
                         "from the incoming global, so the two phases share no "
                         "origin and must not be drawn on one plane.")
    past_tasks = [family.tasks[i] for i in range(k - 1)]
    n_iters = args.iters or (cfg.ppo.global_iters if args.phase == "global"
                             else cfg.ppo.local_iters)

    # The constraint reference is the specialist's own stochastic value, exactly
    # as the trainer computes it at the start of a consolidation phase.
    ref_current, _, _, _ = evaluate_value_and_score(
        specialist, task_k, cfg.ppo.constraint_episodes, cfg.ppo.n_envs,
        device, greedy=False)
    eps = float(cfg.trainer.eps if isinstance(cfg.trainer.eps, (int, float))
                else cfg.trainer.eps[0])
    omega = [1.0 / k] * (k - 1)
    print(f"task {k}: {len(past_tasks)} past tasks, {n_iters} iterations, "
          f"eps={eps}, ref V_k^L={ref_current:.4f}")

    logger = RunLogger(DATA / "_traj_runs", f"traj_k{k}",
                       {"note": "trajectory probe", "task": k})
    trainer = GlobalTrainer(cfg.ppo, device, logger, log_every=args.log_every)
    mu_ctrl = make_dual(cfg.duals)
    mu_ctrl.reset()

    keys = list(global_policy.state_dict().keys())
    snapshots: list[np.ndarray] = []
    iters: list[int] = []
    counter = {"n": 0}

    def probe(_phase: str, _task: int) -> None:
        """Snapshot the live parameters; called once per consolidation step."""
        i = counter["n"]
        counter["n"] = i + 1
        if i % args.stride and i != 0:
            return
        sd = global_policy.state_dict()
        snapshots.append(
            torch.cat([sd[key].detach().flatten() for key in keys])
            .cpu().numpy().astype(np.float32))
        iters.append(i)

    # The starting point is an iterate too, and it is the one the figure needs.
    sd0 = global_policy.state_dict()
    snapshots.append(torch.cat([sd0[key].detach().flatten() for key in keys])
                     .cpu().numpy().astype(np.float32))
    iters.append(-1)

    start = time.time()
    if args.phase == "global":
        summary = trainer.train(
            global_policy, task_k, past_tasks,
            ref_current=ref_current, mu_ctrl=mu_ctrl, omega=omega, eps=eps,
            num_iters=n_iters, seed=cfg.experiment.seed + 1000 * k + 13,
            current_task=k, probe=probe, local_policy=specialist,
            retention_refs=None, retention_frac=cfg.ppo.global_retention_frac,
            past_experts=None, clog=None)
    else:
        # Standard PPO on task k alone, from the same starting point.
        local_trainer = LocalTrainer(cfg.ppo, device, logger,
                                     log_every=args.log_every)
        summary = local_trainer.train(
            global_policy, task_k, num_iters=n_iters,
            seed=cfg.experiment.seed + 1000 * k + 13, current_task=k,
            phase_type="local", probe=probe, clog=None)
    logger.close()
    print(f"consolidation finished in {time.time() - start:.0f}s")

    # The dual trace comes from the trainer's own log, not from a second source.
    mu_trace, sf_trace, step_trace = [], [], []
    for line in (DATA / "_traj_runs" / f"traj_k{k}" / "logs.jsonl").read_text().splitlines():
        rec = json.loads(line)
        if rec.get("phase") == "global" and "mu" in rec:
            step_trace.append(rec["step"])
            mu_trace.append(rec["mu"])
            sf_trace.append(rec.get("F_G"))

    out = DATA / f"traj_k{k}_{args.phase}.npz"
    np.savez_compressed(
        out, iterates=np.stack(snapshots), iters=np.array(iters),
        mu_step=np.array(step_trace, dtype=np.int32),
        mu=np.array(mu_trace, dtype=np.float32),
        constraint=np.array([v if v is not None else np.nan for v in sf_trace],
                            dtype=np.float32),
        meta=np.frombuffer(json.dumps(dict(
            task=k, n_past=len(past_tasks), eps=eps, ref_current=ref_current,
            n_iters=n_iters, stride=args.stride, keys=keys,
            phase=args.phase,
            summary={key: val for key, val in summary.items()
                     if isinstance(val, (int, float, str, bool))},
            run=RUN.name)).encode(), dtype=np.uint8))
    print(f"wrote {out}  ({len(snapshots)} iterates of {snapshots[0].size} params)")


if __name__ == "__main__":
    main()
