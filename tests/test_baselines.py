"""Baseline trainers: naive sequential fine-tuning and joint multi-task.

Fine-tuning must exhibit catastrophic forgetting of the OLDEST task (its whole
point as a lower bound); joint training must retain everything (upper bound).
Both must emit the artifacts the plotting code expects.
"""

import json

from crl.baselines import joint_multitask, sequential_finetune
from crl.buffers import BufferSet
from crl.config import config_from_dict
from crl.envs import make_family
from crl.estimators import make_estimator
from crl.logging_utils import RunLogger
from crl.policies import make_policy
from crl.seeding import set_seed


def _config(tmp_path, name):
    return config_from_dict({
        "experiment": {"name": name, "seed": 3, "results_dir": str(tmp_path),
                       "log_every": 100},
        "env": {"family": "gridworld",
                "params": {"size": 5, "slip": 0.1, "gamma": 0.95, "max_steps": 100},
                "tasks": [{"goal": [0, 4]}, {"goal": [4, 4]}, {"goal": [4, 0]}]},
        "policy": {"kind": "multihead", "hidden_sizes": [128, 128],
                   "task_conditioned": True},
        "estimator": {"kind": "exact"},
        "trainer": {"cycles_per_task": 2, "local_steps": 200, "global_steps": 200,
                    "task1_steps": 400, "lr_local": 0.05, "lr_global": 0.05,
                    "optimizer": "adam", "eval_episodes": 1, "eval_probe_every": 50},
    })


def _components(config, name):
    set_seed(config.experiment.seed)
    family = make_family(config.env)
    policy = make_policy(config.policy, family)
    estimator = make_estimator(config.estimator, buffer_set=BufferSet())
    logger = RunLogger(config.experiment.results_dir, name, config.to_dict())
    return family, policy, estimator, logger


def test_sequential_finetune_forgets_oldest(tmp_path):
    config = _config(tmp_path, "finetune")
    family, policy, estimator, logger = _components(config, "finetune")
    matrix = sequential_finetune(config, family, policy, estimator, logger)
    logger.close()

    final = matrix[-1]
    # Newest task learned; oldest task forgotten well below it.
    assert final[-1] > 0.6, f"newest task not learned: {final}"
    assert final[0] < final[-1] - 0.3, (
        f"fine-tuning did not forget the oldest task: {final}"
    )
    # Artifacts the plotting code needs.
    assert (logger.run_dir / "eval_matrix.json").exists()
    assert (logger.run_dir / "final_policy.pt").exists()
    with open(logger.run_dir / "logs.jsonl") as handle:
        phases = {json.loads(line)["phase"] for line in handle if line.strip()}
    assert {"finetune", "probe", "eval"} <= phases


def test_joint_multitask_retains_all(tmp_path):
    config = _config(tmp_path, "joint")
    family, policy, estimator, logger = _components(config, "joint")
    matrix = joint_multitask(config, family, policy, estimator, logger)
    logger.close()

    final = matrix[-1]
    assert len(final) == 3
    assert all(v > 0.6 for v in final), f"joint training did not retain all tasks: {final}"
