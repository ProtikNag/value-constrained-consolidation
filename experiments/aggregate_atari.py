"""Aggregate the Atari PPO continual-learning runs into figures + tables.

Discovers per-seed run directories named
``results/atari5_ppo_<method>_seed<seed>`` for ``method`` in {constrained,
finetune}, then builds the retention table, retention bars, per-method forgetting
matrices and a game-score table under ``reports/<name>/``. Probe-based learning
curves are produced only if the runs logged probes (``ppo.eval_every > 0``).

    python -m experiments.aggregate_atari --name atari5_ppo --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

import analysis.aggregate as A
from analysis import continual_metrics as CM

METHODS = ("constrained", "finetune", "clear")


def _eval_matrices(run_dirs: list[Path]) -> list[np.ndarray]:
    """Per-seed [T, G] score matrices from eval_matrix.json (complete runs only)."""
    mats = []
    for d in run_dirs:
        m = np.array(json.loads((d / "eval_matrix.json").read_text()), dtype=float)
        if m.shape[0] == m.shape[1]:  # fully trained (rows == games)
            mats.append(m)
    return mats


def _cl_metrics_table(runs, games, targets, tables_dir: Path) -> None:
    """Write raw + normalized CL metrics (AvgPerf / Forgetting / BWT) per method."""
    rows = []
    for method, dirs in runs.items():
        mats = _eval_matrices(dirs)
        if not mats:
            print(f"[aggregate] {method}: no COMPLETE runs yet for CL metrics")
            continue
        s = CM.summarize(mats, games, targets)
        for space in ("raw", "norm"):
            for metric, (mean, std) in s[space].items():
                rows.append({"method": method, "space": space, "metric": metric,
                             "mean": round(mean, 3), "std": round(std, 3),
                             "n_seeds": len(mats)})
    if not rows:
        return
    tables_dir.mkdir(parents=True, exist_ok=True)
    with open(tables_dir / "cl_metrics.csv", "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=["method", "space", "metric", "mean",
                                          "std", "n_seeds"])
        w.writeheader(); w.writerows(rows)
    print("[aggregate] cl_metrics.csv:")
    for r in rows:
        print(f"    {r['method']:12s} {r['space']:4s} {r['metric']:16s} "
              f"{r['mean']:8.3f} ± {r['std']:.3f}  (n={r['n_seeds']})")


def _discover(results_dir: Path, name: str, seeds: list[int]) -> dict[str, list[Path]]:
    runs: dict[str, list[Path]] = {}
    for method in METHODS:
        dirs = []
        for s in seeds:
            d = results_dir / f"{name}_{method}_seed{s}"
            if (d / "eval_matrix.json").exists():
                dirs.append(d)
            else:
                print(f"[aggregate] missing: {d}")
        if dirs:
            runs[method] = dirs
    return runs


def _game_names(run_dir: Path) -> list[str]:
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    return [t["game"] for t in cfg["env"]["tasks"]]


def _targets(run_dir: Path) -> list[float]:
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    return [float(t.get("threshold", float("inf"))) for t in cfg["env"]["tasks"]]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", default="atari5_ppo")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--results-dir", default="results")
    p.add_argument("--reports-dir", default="reports")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    runs = _discover(results_dir, args.name, args.seeds)
    if not runs:
        raise SystemExit(f"[aggregate] no runs found for '{args.name}' under {results_dir}")
    print(f"[aggregate] methods={ {m: len(d) for m, d in runs.items()} }")

    out = Path(args.reports_dir) / args.name
    figures = out / "figures"
    tables = out / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    ref_dir = next(iter(runs.values()))[0]
    game_names = _game_names(ref_dir)
    targets = _targets(ref_dir)

    # Continual-learning metrics (raw + normalized): AvgPerf / Forgetting / BWT.
    _cl_metrics_table(runs, game_names, targets, tables)

    # Core (from end-of-task eval matrices; no probes required).
    A.build_retention_table_ci(runs, figures, tables)
    print("[aggregate] retention_table done")
    A.plot_retention_bars_ci(runs, figures, metric_label="game score")
    print("[aggregate] retention_bars done")
    for method, dirs in runs.items():
        A.plot_forgetting_matrix_mean(
            dirs, figures / method,
            name="forgetting_matrix_mean", metric_label="game score",
        )
    print("[aggregate] forgetting matrices done")

    # Game-score table (final row per method as [S, T]).
    returns = {m: A._final_stack(dirs) for m, dirs in runs.items()}
    try:
        A.build_score_table_ci(returns, game_names, figures, tables)
        print("[aggregate] score_table done")
    except Exception as exc:  # optional; robust to signature drift
        print(f"[aggregate] score_table skipped: {exc}")

    # Probe-based learning curves (only if probes were logged).
    for fn, label in ((A.plot_retention_curves_ci, "retention_curves_ci"),
                      (A.plot_average_performance_curve, "average_performance_curve")):
        try:
            fn(runs, figures, metric_label="game score")
            print(f"[aggregate] {label} done")
        except Exception as exc:
            print(f"[aggregate] {label} skipped (no probes?): {exc}")

    print(f"[aggregate] figures -> {figures}")
    print(f"[aggregate] tables  -> {tables}")


if __name__ == "__main__":
    main()
