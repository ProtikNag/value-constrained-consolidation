"""Regenerate setting_b_v2_summary.json from the live contract runs.

Reads results/<run>/ (progress.jsonl + eval_matrix.json + run.json) through
analysis.contract_metrics, so PERF/BWT/forgetting/FWT are computed the same way
for every method. FWT needs the within-phase learning curves, which only live in
progress.jsonl -- a matrix alone cannot give it.

CLEAR and RECALL pool ALL their seeds under one label. Every run is read at its
own completed depth, so a partial run gives a real number, never nan. Multi-seed
methods report mean +/- std; the pooled number is taken at the deepest depth that
EVERY seed of that method has reached, because PERF is only comparable at equal
depth.

Run from repo root:  python -m reports.biggrid50v2.make_summary
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from analysis.contract_metrics import (
    _matrix_from_eval_json, normalize, within_phase_auc, _read_jsonl,
)
from analysis.continual_metrics import cl_metrics

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
OUT = Path(__file__).resolve().parent / "setting_b_v2_summary.json"

# method label -> (list of seed run dirs under results/, method type). CLEAR and
# RECALL pool every seed they have, with no budget split.
METHODS = {
    "DUEL":         (["duel_v2_fix_seed0",
                      "duel_v2_fix_fresh_seed1",
                      "duel_v2_fix_fresh_seed2"],           "ours"),
    "finetune":     ([f"biggrid50v2_finetune_seed{s}" for s in (0, 1, 2)], "naive"),
    "from_scratch": ([f"biggrid50v2_scratch_seed{s}" for s in (0, 1, 2)],  "naive"),
    "CbpNet":       ([f"biggrid50v2_cbp_seed{s}" for s in (0, 1, 2)],       "plasticity"),
    "CReLU":        ([f"biggrid50v2_crelu_seed{s}" for s in (0, 1, 2)],     "plasticity"),
    "CKA_RL":       ([f"biggrid50v2_cka_rl_seed{s}" for s in (0, 1, 2)],    "growth"),
    "CLEAR":        (["biggrid50v2_clear_seed0",
                      "biggrid50v2_clear_seed1",
                      "biggrid50v2_clear_seed2"],            "replay"),
    "RECALL":       (["biggrid50v2_recall_seed0",
                      "biggrid50v2_recall_seed1",
                      "biggrid50v2_recall_seed2"],           "replay"),
}
# from-scratch is the FWT reference, paired by seed index.
SCRATCH = {0: "biggrid50v2_scratch_seed0", 1: "biggrid50v2_scratch_seed1",
           2: "biggrid50v2_scratch_seed2"}
RETAINERS = ["DUEL", "CLEAR", "RECALL"]
DEPTHS = [10, 20, 34, 50]
N_FULL = 50

REFERENCE = {
    "joint_ceiling": 0.967, "oracle": 0.982, "random_floor": -0.003,
    "frames_matched": 204800000,
    "duel_pre_fix": {"PERF": 0.041, "BWT": -0.684},
}
FWT_CAVEAT = (
    "FWT eq-13 ratio is unstable on v2: from-scratch within-phase AUC A_i^b is "
    "near 1 for ~34% of tasks, so (1-A_i^b) explodes the per-task ratio and can "
    "flip its sign. Lead with raw_gap (A_i - A_i^b)."
)
NOTE = (
    "All eight methods are complete at 3 seeds x 50 tasks. Each method pools its "
    "seeds under one label (mean +/- std). PERF is only compared across methods "
    "at equal depth (equal_depth_retainers)."
)


def seed_index(run: str) -> int:
    return int(run.rsplit("seed", 1)[1])


def load_norm(run: str) -> np.ndarray:
    d = RESULTS / run
    ref = json.loads((d / "run.json").read_text()).get("reference", {})
    rows = json.loads((d / "eval_matrix.json").read_text())
    M = _matrix_from_eval_json(rows, len(rows))
    return normalize(M, list(ref.get("random", [])), list(ref.get("ceiling", [])))


def metrics_at(Mn: np.ndarray, d: int) -> dict:
    sub = Mn[:d, :d]
    m = cl_metrics(sub)
    return {"PERF": float(np.nanmean(sub[d - 1, :d])),
            "BWT": m["bwt"], "forgetting": m["forgetting"]}


def fwt_raw(run: str) -> float | None:
    """A_i - A_i^b vs the seed-matched from-scratch run, or None.

    FWT needs each task's within-phase acquisition curve. The replay trainer
    (CLEAR/RECALL) logs only the end-of-task score, so it has a curve for just
    the first task; averaging over that is meaningless, so we require the curve
    to cover at least half the tasks and otherwise return None."""
    base = SCRATCH.get(seed_index(run))
    if base is None or not (RESULTS / base / "progress.jsonl").exists():
        return None
    auc = within_phase_auc(_read_jsonl(RESULTS / run / "progress.jsonl"))
    auc_b = within_phase_auc(_read_jsonl(RESULTS / base / "progress.jsonl"))
    gap = [auc[i] - auc_b[i] for i in auc if i in auc_b]
    return float(np.mean(gap)) if len(gap) >= 25 else None


def mean_std(xs: list[float]) -> dict:
    """Mean and SAMPLE standard deviation over seeds.

    ddof=1 is deliberate. numpy's default is the population standard deviation,
    which divides by n rather than n-1 and so under-reports the spread of a
    sample of seeds. At n=3 that is the difference between 0.031 and 0.039 on
    DUEL's PERF, and the figures and the paper both use the sample form.
    """
    a = np.array(xs, float)
    sd = float(a.std(ddof=1)) if len(xs) > 1 else 0.0
    return {"mean": float(a.mean()), "std": sd, "n": len(xs)}


def method_entry(runs: list[str], typ: str) -> dict:
    norms = {r: load_norm(r) for r in runs}
    depths = {r: norms[r].shape[0] for r in runs}
    per_seed = []
    for r in sorted(runs, key=seed_index):
        d = depths[r]
        at = metrics_at(norms[r], d)
        per_seed.append({"seed": seed_index(r), "depth": d, **at,
                         "FWT_raw_gap": fwt_raw(r)})
    common = min(depths.values())
    deepest = max(runs, key=lambda r: depths[r])
    entry = {
        "type": typ,
        "n_seeds": len(runs),
        "status": "complete" if common >= N_FULL else "partial",
        "depths": [depths[r] for r in sorted(runs, key=seed_index)],
        # Equal-depth pooled number at the shallowest seed's depth -- the only
        # cross-seed-comparable scalar. PERF across methods is compared in
        # equal_depth_retainers, NOT by reading this across rows at mixed depths.
        "equal_depth_tasks": common,
        "PERF": mean_std([metrics_at(norms[r], common)["PERF"] for r in runs]),
        "BWT": mean_std([metrics_at(norms[r], common)["BWT"] for r in runs]),
        "forgetting": mean_std(
            [metrics_at(norms[r], common)["forgetting"] for r in runs]),
        # compat for make_figures: scalar FWT at the deepest seed's depth.
        "tasks": depths[deepest],
        "FWT_raw_gap": fwt_raw(deepest),
        "seeds": per_seed,
    }
    return entry


def equal_depth_table(norms: dict[str, dict[str, np.ndarray]]) -> dict:
    """For each depth d, each retainer's mean+/-std over the seeds that reached d."""
    out = {}
    for d in DEPTHS:
        row = {}
        for m in RETAINERS:
            vals = [metrics_at(Mn, d) for Mn in norms[m].values()
                    if Mn.shape[0] >= d]
            if vals:
                row[m] = {"n_seeds": len(vals),
                          "PERF": mean_std([v["PERF"] for v in vals]),
                          "BWT": mean_std([v["BWT"] for v in vals]),
                          "forgetting": mean_std([v["forgetting"] for v in vals])}
        if row:
            out[str(d)] = row
    return out


def main() -> None:
    methods = {m: method_entry(runs, typ) for m, (runs, typ) in METHODS.items()}
    norms = {m: {r: load_norm(r) for r in METHODS[m][0]} for m in RETAINERS}
    equal_depth = equal_depth_table(norms)

    OUT.write_text(json.dumps({
        "reference": REFERENCE,
        "note": NOTE,
        "fwt_caveat": FWT_CAVEAT,
        "methods": methods,
        "equal_depth_retainers": equal_depth,
    }, indent=2))
    print(f"wrote {OUT}\n")
    for m, v in methods.items():
        p, b = v["PERF"], v["BWT"]
        print(f"  {m:13s} {v['status']:8s} seeds={v['n_seeds']} depths={v['depths']}  "
              f"PERF={p['mean']:.3f}±{p['std']:.3f}  BWT={b['mean']:+.3f}±{b['std']:.3f}")
    print("\n  equal-depth retainers (mean±std over seeds reaching depth):")
    for d, row in equal_depth.items():
        cells = "  ".join(f"{m}={row[m]['PERF']['mean']:.3f}(n{row[m]['n_seeds']})"
                          for m in row)
        print(f"    depth {d:>2}:  {cells}")


if __name__ == "__main__":
    main()
