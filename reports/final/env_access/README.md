# Env-access ablation

Four figures. The question is whether VCC's advantage is bought by re-entering
earlier environments, so CLEAR and RECALL are rerun with the same access and no
replay buffer. The gap does not close.

Regenerate with `python reports/final/env_access/make_figures.py`. Source data
is `reports/gridworld_env_access/`. Every metric is recomputed from each run's
`eval_matrix.json` and asserted against `summary.json` at build time.

**Caveat on every figure.** Single seed, ten tasks, reduced budget, one task
order. Directional rather than full-scale.

---

## Which comparison the figures report

The record holds two. Only the second is clean.

The first ran every arm at one reduced budget. Under it VCC left the specialists
for tasks 8 and 10 untrained, so its own diagonal is zero there, and CLEAR reads
higher than VCC at 0.629 against 0.610. That reading is confounded.

The second resumes those two tasks for all three arms. The replay baselines are
given the budget of VCC's deployed policy rather than VCC's full budget, which
concedes VCC's specialist phase to them and makes this the conservative
comparison. It is the one the first three figures report.

| | PERF | BWT | Forgetting | acquired | solved |
|---|---:|---:|---:|---:|---:|
| VCC | **0.778** | +0.095 | 0.112 | 7/10 | 8/10 |
| CLEAR, env access | 0.616 | −0.035 | 0.036 | 6/10 | 6/10 |
| RECALL, env access | 0.579 | +0.004 | 0.000 | 6/10 | 6/10 |

"Acquired" counts tasks reaching the 0.6 success bar while they are current.
"Solved" counts tasks above that bar at the end.

## The figures

- **`env_access_matrices`** — the three retention matrices on one shared scale,
  with the diagonal outlined. The replay baselines show pale vertical stripes,
  which are tasks they never learn and never recover.
- **`env_access_retention_curve`** — mean score over the tasks seen so far. VCC
  starts lowest at 0.59 and ends highest at 0.78, while both replay baselines
  start near the ceiling and end below it.
- **`env_access_acquire_vs_final`** — each task's score on arrival against its
  score at the end. The clearest statement of the mechanism: the replay
  baselines are almost all stubs, because they keep what they acquire and
  recover nothing, while VCC's bars move.
- **`env_access_budget_correction`** — PERF under the first reading beside the
  corrected one, so the confound and its repair are visible rather than
  asserted.

## What the numbers say

Giving replay methods live environment access does not close the gap. Both
baselines retain almost perfectly, with forgetting of 0.036 and 0.000, but they
retain less because they acquire less and never recover a task once it is
missed. VCC forgets more on that measure, 0.112, and still ends 0.16 ahead,
because its backward transfer is positive while theirs is about zero.

---

## Conventions

The academic template from `report/acviz.py`. White ground, Tufte spine, faint
horizontal grid, blue for ours and the Setting B colours for CLEAR and RECALL,
mono numerals, PNG at 300 dpi and SVG from the same figure object. The matrix
scale is the Setting B scale unchanged. Set `FIGS_FOR_PAPER=1` to drop the
footnotes, which a LaTeX caption would otherwise duplicate.
