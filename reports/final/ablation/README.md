# Ablation figures

Nine figures over two experiments, both on a 10-task slice of GridWorld-50, both
single seed, both at a reduced budget. Built for review: keep the ones that earn
their space and delete the rest.

Regenerate with `python reports/final/ablation/make_figures.py`. Source data is
`reports/gridworld_ablations/`. Every metric is recomputed here from each run's
`eval_matrix.json` rather than read from `summary.json`, and the two agree to
about 0.001.

**The caveat is printed on every figure.** Single seed, ten tasks, reduced
budget, directional rather than full-scale. The two experiments used different
local budgets, so no figure puts a value from one on the same axis as a value
from the other.

---

## Experiment 1, the constraint on against off

Two runs identical except the multiplier cap. 5.0 leaves μ free, 0.0 pins it to
zero and removes the current-task constraint, at which point the consolidation
objective is the past-task sum alone and carries no term for the task in front
of the policy.

| | PERF | BWT | Forgetting | acquired | solved |
|---|---:|---:|---:|---:|---:|
| Constraint on | **0.610** | +0.128 | 0.083 | 5/10 | 6/10 |
| Constraint off, μ = 0 | 0.337 | +0.272 | 0.081 | 1/10 | 3/10 |

"Acquired" counts tasks reaching the 0.6 success bar while they are current.
"Solved" counts tasks above that bar at the end.

- **`constraint_matrices`** — the two retention matrices on one scale, with the
  diagonal outlined. The clearest statement of the result: with the constraint
  off the diagonal is near zero for every task after the first.
- **`constraint_retention_curve`** — the mean score over the tasks seen so far
  as the sequence advances, one line per setting and no band, since each line is
  one run. Each line is labelled with its PERF after ten tasks. The
  unconstrained run starts higher, because task 1 is the one task it does learn,
  then falls below at task 2 and never recovers.

## Experiment 2, the consolidation rollout depth

Five runs identical except `global_n_steps`, the PPO rollout depth inside the
consolidation phase. The value estimator stays Monte-Carlo and `window_H` is
untouched, so this is about rollout depth and not about the windowed value
estimate. All five spend the same 1800 consolidation iterations.

| depth | PERF | BWT | Forgetting | wall | steps/iter |
|---:|---:|---:|---:|---:|---:|
| 1 | −0.000 | −0.036 | 0.041 | 6.9 min | 32 |
| 10 | 0.056 | +0.028 | 0.120 | 15.4 min | 320 |
| 20 | 0.168 | −0.034 | 0.164 | 24.1 min | 640 |
| 40 | 0.279 | +0.061 | 0.172 | 39.1 min | 1280 |
| 80 | **0.483** | +0.198 | 0.025 | 70.0 min | 2560 |

- **`depth_perf_and_cost`** — PERF and consolidation wall clock on one panel
  with two y axes. The right-hand range is deliberately wider than the data so
  the curves do not lie on top of one another; their near-parallel shape is the
  result, and the vertical gap between them carries no meaning.
- **`depth_matrices`** — the five retention matrices on one scale. Locates what
  depth buys: the diagonal fills in, so it buys acquisition, not retention.

Dropped at review: `constraint_acquire_vs_retain`, `constraint_headline`,
`constraint_task_profile`, `depth_efficiency`, `depth_metrics`, `depth_perf`.
PERF against depth survives inside `depth_perf_and_cost`. The script
deletes their files on every run so a stale PNG cannot be mistaken for current
output.

---

## Conventions

The academic template from `report/acviz.py`. White ground, Tufte spine, faint
horizontal grid, the shared palette with blue for the configuration we advocate
and amber for the one it is measured against, mono numerals, PNG at 300 dpi and
SVG from the same figure object. The matrix scale is the Setting B scale
unchanged, so a colour means the same thing in both figure sets. Set
`FIGS_FOR_PAPER=1` to drop the footnotes, which the LaTeX caption would
otherwise duplicate.
