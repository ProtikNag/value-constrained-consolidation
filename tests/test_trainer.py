"""End-to-end acceptance test on the exact-gradient gridworld.

Core claim of the method: as the global policy consolidates the past, the
one-sided squared constraint F_G (eqs 11-13) stops it from discarding the
current task. The check compares a constrained run against an unconstrained
baseline (duals disabled via lr = 0). Correct primal-dual updates must yield:
(a) the constrained global learns the new task (task 2), and (b) it retains
task 2 far better than the baseline, which over-consolidates the past and
forgets task 2. Exact estimation removes sampling noise, so failures here
indicate real bugs, not variance.
"""

import json
import math

from crl.buffers import BufferSet
from crl.config import config_from_dict
from crl.estimators import make_estimator
from crl.envs import make_family
from crl.logging_utils import RunLogger
from crl.policies import make_policy
from crl.seeding import set_seed
from crl.trainer import AlternationTrainer

EPS = 0.0016  # squared-value units: tolerated value gap ~ 0.04


def _config(tmp_path, name, eps, dual_lr):
    return config_from_dict(
        {
            "experiment": {"name": name, "seed": 3,
                           "results_dir": str(tmp_path), "log_every": 20},
            "env": {
                "family": "gridworld",
                "params": {"size": 4, "slip": 0.1, "gamma": 0.9, "max_steps": 60},
                "tasks": [{"goal": [0, 3]}, {"goal": [3, 3]}],
            },
            "policy": {"kind": "tabular"},
            "estimator": {"kind": "exact"},
            "duals": {"kind": "projected_ascent", "lr": dual_lr,
                      "max_value": 200.0, "warm_start": True},
            "trainer": {
                "cycles_per_task": 3,
                "local_steps": 200,
                "global_steps": 200,
                "task1_steps": 150,
                "lr_local": 0.5,
                "lr_global": 0.5,
                "optimizer": "sgd",
                "eps": eps,
                "eval_episodes": 1,
            },
        }
    )


def _run(config, name):
    set_seed(config.experiment.seed)
    family = make_family(config.env)
    policy = make_policy(config.policy, family)
    estimator = make_estimator(config.estimator, buffer_set=BufferSet())
    logger = RunLogger(config.experiment.results_dir, name, config.to_dict())
    matrix = AlternationTrainer(config, family, policy, estimator, logger).run()
    logger.close()
    return matrix, logger.run_dir


def test_alternation_learns_and_retains(tmp_path):
    # Constrained run vs unconstrained baseline (duals disabled: lr 0 -> mu=0).
    matrix, run_dir = _run(_config(tmp_path, "constrained", EPS, 8.0), "constrained")
    base_matrix, _ = _run(
        _config(tmp_path, "unconstrained", EPS, 0.0), "unconstrained"
    )

    assert len(matrix) == 2 and all(len(row) == 2 for row in matrix)
    assert all(math.isfinite(v) for row in matrix for v in row)

    v1_after_task1 = matrix[0][0]
    v2_after_task1 = matrix[0][1]
    v2_after_task2 = matrix[1][1]

    # Task 1 was actually learned before the sequence moved on.
    assert v1_after_task1 > 0.15, f"task 1 never learned: {v1_after_task1:.3f}"
    # (a) the constrained global learned task 2 across the alternation.
    assert v2_after_task2 > v2_after_task1 + 0.05, (
        f"task 2 not learned: {v2_after_task1:.3f} -> {v2_after_task2:.3f}"
    )
    # (b) the constraint retains task 2 far better than the unconstrained
    # baseline, which over-consolidates the past and forgets it.
    base_v2_after_task2 = base_matrix[1][1]
    assert v2_after_task2 > base_v2_after_task2 + 0.1, (
        f"constraint gave no retention benefit: task-2 value "
        f"{v2_after_task2:.3f} (constrained) vs {base_v2_after_task2:.3f} (baseline)"
    )

    # Run directory is self-describing.
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "eval_matrix.json").exists()
    with open(run_dir / "logs.jsonl") as handle:
        records = [json.loads(line) for line in handle]
    phases = {record["phase"] for record in records}
    assert {"task1", "local", "global", "gaps", "eval"} <= phases
