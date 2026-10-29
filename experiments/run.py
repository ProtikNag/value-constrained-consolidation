"""Single-run entry point.

Usage:
    python -m experiments.run --config configs/gridworld_exact.yaml
    python -m experiments.run --config configs/minatar_multihead.yaml --seed 7
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch

from crl.buffers import BufferSet
from crl.config import Config, load_config
from crl.envs import make_family
from crl.estimators import make_estimator
from crl.logging_utils import RunLogger
from crl.policies import make_policy
from crl.seeding import set_seed
from crl.trainer import AlternationTrainer


def resolve_device(name: str) -> torch.device:
    """Resolve a config device string; 'auto'/'cuda' fall back to CPU if no GPU."""
    if name in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def find_latest_checkpoint(run_dir: str | Path) -> tuple[str | None, int]:
    """Highest-``k`` ``global_after_task{k}.pt`` in ``run_dir``.

    Returns ``(path, k)``, or ``(None, 0)`` when the directory holds no per-task
    checkpoint (a run that died inside task 1, or never started).
    """
    best, best_k = None, 0
    for f in Path(run_dir).glob("global_after_task*.pt"):
        m = re.fullmatch(r"global_after_task(\d+)\.pt", f.name)
        if m and int(m.group(1)) > best_k:
            best, best_k = str(f), int(m.group(1))
    return best, best_k


def run_from_config(config: Config, resume_from: str | None = None,
                    resume_after: int = 0,
                    auto_resume: bool = False) -> list[list[float]]:
    """Build every component from config and train; returns the eval matrix.

    ``resume_from`` (a ``global_after_task{k}.pt``) with ``resume_after=k`` continues
    a prior run instead of training from scratch: the trunk + already-trained heads
    are loaded (new heads stay fresh, so more games can be appended) and tasks
    1..k are skipped.

    ``auto_resume`` looks for the highest ``global_after_task{k}.pt`` already in
    this config's own run directory and continues from it, so a preempted job can
    be relaunched with an unchanged command line. It is a no-op on a fresh run."""
    set_seed(config.experiment.seed)
    device = resolve_device(config.experiment.device)
    family = make_family(config.env)
    policy = make_policy(config.policy, family).to(device)
    run_name = f"{config.experiment.name}_seed{config.experiment.seed}"
    run_dir = Path(config.experiment.results_dir) / run_name
    if auto_resume and not resume_from:
        resume_from, resume_after = find_latest_checkpoint(run_dir)
        if resume_from:
            print(f"[auto-resume] continuing after task {resume_after} "
                  f"from {resume_from}")
        else:
            print(f"[auto-resume] no checkpoint in {run_dir}; starting fresh")
    # A resumed run appends to logs.jsonl; a fresh one truncates it.
    logger = RunLogger(config.experiment.results_dir, run_name, config.to_dict(),
                       append=bool(resume_from))

    if config.trainer.kind == "ppo":
        # PPO backend: same continual-learning framework, PPO as the optimizer.
        from crl.ppo_continual import PPOAlternationTrainer

        print(
            f"[run] name={run_name} seed={config.experiment.seed} "
            f"env={config.env.family}({len(family)} tasks) "
            f"policy={config.policy.kind} backend=ppo method={config.ppo.method} "
            f"eps={config.trainer.eps} device={device}"
        )
        trainer = PPOAlternationTrainer(config, family, policy, logger)
        if resume_from:
            trainer.resume(resume_from, resume_after)
    else:
        if resume_from:
            raise SystemExit("--resume-from is only supported for the ppo backend")
        estimator = make_estimator(config.estimator, buffer_set=BufferSet())
        print(
            f"[run] name={run_name} seed={config.experiment.seed} "
            f"env={config.env.family}({len(family)} tasks) "
            f"policy={config.policy.kind} estimator={config.estimator.kind} "
            f"eps={config.trainer.eps} device={device}"
        )
        trainer = AlternationTrainer(config, family, policy, estimator, logger)
    try:
        matrix = trainer.run()
    finally:
        logger.close()
    print(f"[run] done; results in {logger.run_dir}")
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument("--seed", type=int, default=None, help="Override seed.")
    parser.add_argument(
        "--results-dir", default=None, help="Override results directory."
    )
    parser.add_argument("--resume-from", default=None,
                        help="global_after_task{k}.pt to continue from (ppo backend).")
    parser.add_argument("--resume-after", type=int, default=0,
                        help="Skip training tasks 1..this (the k in the checkpoint).")
    parser.add_argument("--auto-resume", action="store_true",
                        help="Continue from the latest global_after_task{k}.pt in "
                             "this config's own run dir. No-op if there is none, so "
                             "the same command line works for launch and relaunch.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.seed is not None:
        config.experiment.seed = args.seed
    if args.results_dir is not None:
        config.experiment.results_dir = args.results_dir
    run_from_config(config, resume_from=args.resume_from,
                    resume_after=args.resume_after,
                    auto_resume=args.auto_resume)


if __name__ == "__main__":
    main()
