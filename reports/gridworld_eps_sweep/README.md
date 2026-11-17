# Tolerance sweep, GridWorld first 15 tasks

VCC with `value_mode: mc` on the first fifteen tasks of the fifty-task GridWorld
sequence, four values of the tolerance `eps`, three seeds (2, 3, 4), metrics
read at task 15. `summary.json` holds per-seed values, aggregates as
`[mean, sample sd (ddof=1)]`, the current-task scores (`diagonal_vs_k`, the
score on each task at the end of its own phase), the final rows and the raw
15x15 retention matrices.

| eps | PERF | BWT | forgetting | >=0.9 | <=0.1 | current-task score |
|---|---|---|---|---|---|---|
| 0.03 | 0.752 ± 0.034 | +0.262 ± 0.037 | 0.048 ± 0.078 | 0.76 | 0.20 | 0.507 |
| 0.06 | 0.782 ± 0.061 | +0.309 ± 0.076 | 0.026 ± 0.039 | 0.78 | 0.20 | 0.493 |
| 0.09 | 0.796 ± 0.020 | +0.358 ± 0.092 | 0.014 ± 0.019 | 0.78 | 0.18 | 0.462 |
| 0.18 | 0.846 ± 0.005 | +0.487 ± 0.108 | 0.003 ± 0.002 | 0.87 | 0.13 | 0.392 |

`eps = 0.09` is the value of the fifty-task Setting B runs; its row is
`duel_v2_fix_fresh` read at task 15, since the first fifteen rows of a retention
matrix do not depend on later tasks. The other rows use
`configs/duel_v2_eps003_len15.yaml`, `duel_v2_eps006_len15.yaml` and
`duel_v2_eps018_len15.yaml`, which differ from `duel_v2_fix_fresh.yaml` only in
`eps` and in the task list being cut to the first fifteen tasks.

PERF and BWT rise with `eps` and forgetting falls, while the current-task score
falls from 0.507 to 0.392.
