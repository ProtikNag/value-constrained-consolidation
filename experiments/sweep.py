"""Grid-sweep runner: one base config, a YAML grid, sequential runs.

The grid file maps dotted config paths to value lists; the Cartesian product
is enumerated and each point becomes one run whose name encodes the swept
values. Example ``configs/sweeps/eps_grid.yaml``:

    trainer.eps: [0.01, 0.05, 0.2]
    duals.lr: [0.01, 0.05]

Usage:
    python -m experiments.sweep --base configs/gridworld_exact.yaml \
        --grid configs/sweeps/eps_grid.yaml [--seeds 0 1 2] [--dry-run]

Runs execute sequentially in-process. On a cluster, wrap this command in a
single batch script, or split the grid across array jobs with --index.
"""

from __future__ import annotations

import argparse
import copy
import itertools
from typing import Any

from crl.config import config_from_dict, load_config
from experiments.run import run_from_config

import yaml


def _set_dotted(raw: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Assign into a nested dict via 'section.key' path."""
    node = raw
    *parents, leaf = dotted_key.split(".")
    for part in parents:
        node = node.setdefault(part, {})
    node[leaf] = value


def _grid_points(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = sorted(grid)
    combos = itertools.product(*(grid[key] for key in keys))
    return [dict(zip(keys, combo)) for combo in combos]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Base YAML config.")
    parser.add_argument("--grid", required=True, help="YAML: dotted key -> list.")
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument(
        "--index", type=int, default=None,
        help="Run only the i-th grid point (for cluster array jobs).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base_config = load_config(args.base)
    base_raw = base_config.to_dict()
    with open(args.grid) as handle:
        grid = yaml.safe_load(handle)
    points = _grid_points(grid)
    seeds = args.seeds if args.seeds is not None else [base_config.experiment.seed]

    jobs = [(point, seed) for point in points for seed in seeds]
    if args.index is not None:
        jobs = [jobs[args.index]]

    for job_number, (point, seed) in enumerate(jobs):
        raw = copy.deepcopy(base_raw)
        for dotted_key, value in point.items():
            _set_dotted(raw, dotted_key, value)
        tag = "_".join(
            f"{key.split('.')[-1]}{value}" for key, value in sorted(point.items())
        )
        raw["experiment"]["name"] = f"{base_config.experiment.name}_{tag}"
        raw["experiment"]["seed"] = seed
        print(f"[sweep] job {job_number + 1}/{len(jobs)}: {point} seed={seed}")
        if args.dry_run:
            continue
        run_from_config(config_from_dict(raw))


if __name__ == "__main__":
    main()
