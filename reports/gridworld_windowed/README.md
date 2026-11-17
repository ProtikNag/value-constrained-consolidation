# Windowed against full-episode shortfall estimation, GridWorld first 15 tasks

VCC on the first fifteen tasks of the fifty-task GridWorld sequence, three seeds
(2, 3, 4), metrics read at task 15. `summary.json` holds per-seed values,
aggregates as `[mean, sample sd (ddof=1)]`, per-seed anytime PERF, the
current-task scores (`diagonal_vs_k`, the score on each task at the end of its
own phase), the final rows and the raw 15x15 retention matrices.

| estimator | PERF | BWT | forgetting | >=0.9 | <=0.1 | current-task score |
|---|---|---|---|---|---|---|
| full episodes (`value_mode: mc`) | 0.796 ± 0.020 | +0.358 ± 0.092 | 0.014 ± 0.019 | 0.78 | 0.18 | 0.462 |
| window, H=10 (`value_mode: bootstrap`) | 0.804 ± 0.037 | +0.559 ± 0.074 | 0.090 ± 0.030 | 0.78 | 0.16 | 0.282 |

Per-seed PERF: window [0.800, 0.843, 0.769], full episodes [0.779, 0.817, 0.791].
Per-seed BWT: window [0.627, 0.481, 0.569], full episodes [0.289, 0.462, 0.322].

The full-episode row is `duel_v2_fix_fresh` read at task 15; the first fifteen
rows of a retention matrix do not depend on later tasks. The window run uses
`configs/duel_v2_window_h10.yaml`, which differs from `duel_v2_fix_fresh.yaml`
only in `value_mode: bootstrap` and `window_H: 10`. With `value_mode: bootstrap`
the trainer skips the full-episode Monte-Carlo estimate (`crl/ppo/trainer.py`),
so the windowed estimate alone drives the constraint. Reported metrics come from
the greedy evaluation matrix and do not depend on the estimator.

The two estimators reach the same PERF and solve the same fraction of tasks.
The window run has a lower current-task score and higher BWT and forgetting.
