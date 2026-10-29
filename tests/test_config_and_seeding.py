"""Config strictness and run-level determinism."""

import copy
from pathlib import Path

import pytest

from crl.config import config_from_dict, load_config
from experiments.run import run_from_config

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


@pytest.mark.parametrize(
    "name", ["gridworld_exact.yaml", "gridworld_sampled.yaml", "minatar_multihead.yaml"]
)
def test_shipped_configs_load(name):
    config = load_config(CONFIG_DIR / name)
    assert config.experiment.seed is not None
    assert len(config.env.tasks) >= 1


def test_unknown_keys_rejected():
    with pytest.raises(KeyError):
        config_from_dict({"trainer": {"learning_rate": 0.1}})  # typo'd key
    with pytest.raises(KeyError):
        config_from_dict({"trainerr": {}})  # typo'd section


def _tiny_mc_config(tmp_path, name):
    return config_from_dict(
        {
            "experiment": {"name": name, "seed": 11,
                           "results_dir": str(tmp_path), "log_every": 5},
            "env": {
                "family": "gridworld",
                "params": {"size": 3, "slip": 0.1, "gamma": 0.9, "max_steps": 30},
                "tasks": [{"goal": [2, 2]}],
            },
            "policy": {"kind": "tabular"},
            "estimator": {"kind": "monte_carlo", "episodes_per_eval": 8,
                          "episodes_per_grad": 8},
            "trainer": {"task1_steps": 15, "eval_episodes": 8},
        }
    )


def test_same_seed_same_run(tmp_path):
    """Two identical Monte-Carlo runs must produce identical eval matrices."""
    matrix_a = run_from_config(_tiny_mc_config(tmp_path, "det_a"))
    matrix_b = run_from_config(copy.deepcopy(_tiny_mc_config(tmp_path, "det_b")))
    assert matrix_a == matrix_b
