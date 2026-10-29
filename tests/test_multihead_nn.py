"""Multi-head neural policy with the exact estimator on a 3-task gridworld.

This is the acceptance test for the neural, multi-task setting: separate task
heads plus the min-max constraint must retain all three tasks, where the
unconstrained baseline forgets the newest one.
"""

import json

import torch

from crl.buffers import BufferSet
from crl.config import EnvConfig, PolicyConfig, config_from_dict
from crl.envs import make_family
from crl.estimators import make_estimator
from crl.estimators.exact import ExactEstimator
from crl.logging_utils import RunLogger
from crl.policies import make_policy
from crl.seeding import set_seed
from crl.trainer import AlternationTrainer


def _family():
    return make_family(EnvConfig(
        family="gridworld",
        params={"size": 5, "slip": 0.1, "gamma": 0.95, "max_steps": 100},
        tasks=[{"goal": [0, 4]}, {"goal": [4, 4]}, {"goal": [4, 0]}],
    ))


def test_multihead_routes_by_task_and_is_differentiable():
    set_seed(0)
    family = _family()
    policy = make_policy(
        PolicyConfig(kind="multihead", hidden_sizes=[64, 64], task_conditioned=True),
        family,
    )
    obs = torch.eye(family.obs_dim)
    # Different heads -> generally different distributions for the same states.
    probs0 = policy.dist(obs, 0).probs
    probs2 = policy.dist(obs, 2).probs
    assert not torch.allclose(probs0, probs2)

    # Exact estimator differentiates through the network.
    objective, _, _ = ExactEstimator().surrogate_objective(policy, family.tasks[0])
    objective.backward()
    grads = [p.grad for p in policy.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


def _config(tmp_path, name, dual_lr):
    return config_from_dict({
        "experiment": {"name": name, "seed": 3, "results_dir": str(tmp_path),
                       "log_every": 50},
        "env": {"family": "gridworld",
                "params": {"size": 5, "slip": 0.1, "gamma": 0.95, "max_steps": 100},
                "tasks": [{"goal": [0, 4]}, {"goal": [4, 4]}, {"goal": [4, 0]}]},
        "policy": {"kind": "multihead", "hidden_sizes": [128, 128],
                   "task_conditioned": True},
        "estimator": {"kind": "exact"},
        "duals": {"kind": "projected_ascent", "lr": dual_lr, "max_value": 200.0,
                  "warm_start": True},
        "trainer": {"cycles_per_task": 3, "local_steps": 300, "global_steps": 300,
                    "task1_steps": 300, "lr_local": 0.05, "lr_global": 0.05,
                    "optimizer": "adam", "eps": 0.0016, "eval_episodes": 1,
                    "eval_probe_every": 50},
    })


def _run(config, name):
    set_seed(config.experiment.seed)
    family = make_family(config.env)
    policy = make_policy(config.policy, family)
    estimator = make_estimator(config.estimator, buffer_set=BufferSet())
    logger = RunLogger(config.experiment.results_dir, name, config.to_dict())
    matrix = AlternationTrainer(config, family, policy, estimator, logger).run()
    logger.close()
    return matrix, logger.run_dir


def test_constraint_retains_all_three_tasks(tmp_path):
    matrix, run_dir = _run(_config(tmp_path, "mh_constrained", 8.0), "mh_constrained")
    base, _ = _run(_config(tmp_path, "mh_baseline", 0.0), "mh_baseline")

    final = matrix[-1]
    base_final = base[-1]
    # Constrained retains all three tasks well above the trivial ~0.2 level.
    assert all(v > 0.6 for v in final), f"constrained did not retain all tasks: {final}"
    # The newest task is the one the baseline drops; the constraint saves it.
    assert final[-1] > base_final[-1] + 0.3, (
        f"no retention benefit on the newest task: constrained {final[-1]:.3f} "
        f"vs baseline {base_final[-1]:.3f}"
    )

    # Probe records exist and are monotonic in cumulative step.
    with open(run_dir / "logs.jsonl") as handle:
        probes = [json.loads(line) for line in handle
                  if line.strip() and json.loads(line).get("phase") == "probe"]
    assert len(probes) > 5
    steps = [p["cumulative_step"] for p in probes]
    assert steps == sorted(steps) and len(set(steps)) == len(steps)
