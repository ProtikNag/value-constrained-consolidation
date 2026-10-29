# BigGrid v2 — figures

    python reports/final/gridworld_v2/make_figures.py

Writes `png/` at 300 dpi and `svg/` as true vector, both from the same figure
object. Every SVG is checked for rasterised `<image>` tags at build time; the
50×50 matrices draw explicit rectangles rather than a `go.Heatmap`, which would
embed a bitmap. The whole set builds in about 20 seconds.

## Read this before using any of them

**Three seeds per method, at unequal depths.** Every seed of a method is pooled
under one label, including the arms whose configuration differs. DUEL seed 0
warm-starts each specialist and seeds 1 and 2 re-initialise it. Those
differences are temporary and nothing here separates them.

| | PERF | Forgetting | BWT | ≥0.9 | ≤0.1 | complete | depths |
|---|---|---|---|---|---|---|---|
| Random floor | -0.003 | — | — | 0% | 100% | — | — |
| BFS oracle | 0.982 | — | — | 100% | 0% | — | — |
| Specialists, 4.096M frames each | 0.894 | — | — | 66% | 0% | — | — |
| Joint, 204.8M frames | 0.967 | — | — | 98% | 2% | — | — |
| **DUEL** | **0.779 ±0.039** | 0.053 ±0.021 | **+0.188 ±0.062** | **79%** | **19%** | 3/3 | 50/50/50 |
| CLEAR | 0.686 ±0.031 | 0.020 ±0.007 | -0.015 ±0.010 | 63% | 28% | 3/3 | 50/50/50 |
| RECALL | 0.589 ±0.028 | **0.002 ±0.001** | +0.000 ±0.001 | 53% | 37% | 3/3 | 50/50/50 |
| CKA-RL | 0.012 ±0.004 | 0.970 ±0.010 | -0.970 ±0.010 | 3% | 91% | 3/3 | 50/50/50 |
| CbpNet | -0.001 ±0.004 | 0.965 ±0.017 | -0.965 ±0.017 | 1% | 92% | 3/3 | 50/50/50 |
| CReLU | -0.004 ±0.004 | 0.647 ±0.140 | -0.645 ±0.141 | 0% | 99% | 3/3 | 50/50/50 |
| Fine-tuning | -0.004 ±0.006 | 0.759 ±0.024 | -0.751 ±0.018 | 0% | 99% | 3/3 | 50/50/50 |
| From-scratch | -0.010 ±0.004 | 0.943 ±0.006 | -0.935 ±0.010 | 2% | 95% | 3/3 | 50/50/50 |

**Standard deviations here are the SAMPLE s.d.** (`ddof=1`) over the three
seeds. `reports/biggrid50v2/make_summary.py` previously used numpy's default,
which is the population s.d. and divides by n rather than n-1; at n=3 that
reported DUEL's PERF spread as 0.031 instead of 0.039. The script is fixed but
its JSON can only be regenerated where `results/` lives, so
`setting_b_v2_summary.json` still carries the old convention. Nothing in the
paper reads PERF or BWT from that file; only forward transfer comes from it.

**Every headline value is a mean over three complete seeds**, read at 50 tasks,
with ±1 s.d. across those seeds. All eight methods are complete at 3 seeds × 50
tasks. DUEL leads CLEAR by **0.094**, against spreads of 0.031 and 0.025, and
remains the only method with positive backward transfer.

## Two decisions that the figures depend on

**Cross-method numbers are read at one depth, never pooled across depths.**
PERF climbs as the sequence shortens, because a retaining method has less to
hold at once, so a mean over seeds of different lengths is not a measurement of
the method. Pooling was tried and discarded: it put CLEAR at 0.720 against
DUEL's 0.712 and reversed the ranking, entirely because the two methods'
unfinished seeds stop at different depths. The table therefore uses only the
seeds that reached all 50 tasks and reads them there. A seed still running
contributes to the curves and the matrix, where its own depth is respected, and
to nothing else.

**Nothing is held forward.** Where seeds end at different depths, each plotted
value is a mean over the seeds that actually reached that depth, and the band is
the min-to-max range over those same seeds. The band narrowing to nothing is the
signal that only one seed is left. A held-forward placeholder would have been
worse than nothing: CReLU's two short seeds stop at task 5 around 0.18 while its
full seed ends at 0.00, so carrying the short value to task 50 would have
tripled a score that is really zero.

In `performance_by_band` a seed scores a band only if it reached at least five
of that band's ten tasks. Without that floor a seed that stopped at task 8 put
CKA-RL at 0.64 on the hardest band off a single lucky task.

## The figures

| file | what it shows |
|---|---|
| `retention_matrix` | DUEL's 50×50 matrix beside CLEAR's, seed-averaged, one shared scale |
| `retention_curves` | mean score over tasks seen, all methods, with seed-range bands |
| `score_distribution` | violins of every task of every seed, when learned and at the end |
| `learned_vs_retained` | the same two quantities as a quadrant plot, for the four methods that retain |
| `performance_by_band` | grouped bars against the difficulty axis, with the joint ceiling as a rule |
| `constraint_satisfaction` | the shortfall distribution before and after each global phase |
| `headline_metrics` | the table: PERF, Forgetting, BWT, FWT, ≥0.9, ≤0.1 |

**`score_distribution` carries the main result.** Every method acquires: the
median score when a task is learned is 0.93 to 0.98 for all eight. Only DUEL,
CLEAR and RECALL still hold it at the end; the other five collapse to about
−0.03. The violin is there because the scores are strongly bimodal, nearly every
task being either solved or on the floor, which a jittered strip hides. The
mean is a diamond rather than a line: it falls in the thin waist between the two
lobes, where a line is only as wide as the density under it and cannot be seen.

**`performance_by_band` shows where DUEL's lead comes from.** On tasks whose
goal sits 60 to 80 steps from the spawn point DUEL scores 0.52 against CLEAR's
0.30 and RECALL's 0.28. On the two nearest bands the three are within 0.10.

**`constraint_satisfaction` is the one constraint figure worth printing, and
its x axis needs reading carefully.** It is a distribution plot. Each curve
holds the same 49 shortfalls it always had, sorted from largest to smallest,
and x is the rank in THAT curve's own sorted list. The two curves are sorted
independently, so rank 10 on one is usually a different task from rank 10 on
the other, and the shaded gap is how far the whole distribution moved down
rather than any single task's improvement. Where each curve crosses the
threshold is labelled, which is where 40 outside at the start and 16 at the end
can be read off the axis. The per-task facts the ranking cannot carry are the
counts in the header: 40 of 49 begin outside the constraint, 33 end inside it,
0 end worse. Training order was tried and is worse for this purpose, because
both curves zig-zag and the shift becomes impossible to see. ε = 0.09 is on the
SQUARED gap, so the satisfied region is a shortfall at or below √0.09 = 0.30.

**`learned_vs_retained` separates rescue from loss**, and shows only the four
methods that retain anything. The other four collapse onto the floor and produce
four identical panels.

## Retired, and why

- `compute_cost` — the runs sat on different CPUs and nodes, so a wall-clock
  comparison measures the cluster, not the method. The frame match it also
  carried belongs in the text, not in a figure.
- the start-against-end scatter that `constraint_satisfaction` carried for one
  revision — it said only that most gaps closed and none widened, which the
  header counts state directly.
- `acquisition_vs_retention` — superseded by `learned_vs_retained`.
- `axis_confound` — the correlations were weak and the figure asserted a
  relationship the scatter did not show.
- `specialist_margin` — a 50-row dumbbell that no reader would use.
- `method_comparison` — replaced by `score_distribution`.
- `dual_dynamics` — only its middle panel said anything a reader needs. μ not
  pinning and the past not starving are checks that the machinery works, not
  results. Replaced by `constraint_satisfaction`.
- `acquisition_curves` — **the data does not exist.** The global phase writes
  eval rows only at the end of each phase, all with `iter: -1`. The only
  within-phase time series in the log is for the LOCAL specialists, not the
  global model.

## Defects in the source logs, worked around here

**Forward transfer has three defects in the run summary**, and the figure code
filters all three. It is an area under a within-phase learning curve, so it
cannot be recomputed from an evaluation matrix and has to come from
`reports/biggrid50v2/setting_b_v2_summary.json`.

1. A value of exactly `0.0` on a method that is not from-scratch is the file's
   not-computed default rather than a measurement. From-scratch is 0 by
   construction, and every method really measured came back with a non-round
   value. CLEAR and RECALL carry this default on seeds 1 and 2.
2. The top-level `FWT_raw_gap` is **seed 0's value, not a mean over seeds**.
   DUEL has two measured seeds, −0.066 and +0.049, and the file reports −0.066.
   The table averages the seeds that carry a value, giving −0.008.
3. **CLEAR and RECALL report byte-identical forward transfer**,
   −0.14344736773259725 on seed 0 for both. An area under a curve over 50
   phases does not coincide to 17 significant figures, so one run's value has
   been copied onto the other. The file does not say which is real, so both are
   dropped and read `—`. The build prints a warning when this fires, and the
   check is general rather than hard-coded to these two methods.

`grad_share_past` and `shortfall_past` are `null` for the entire run. Nothing
shipped now depends on them, since `dual_dynamics` was retired, but they should
be populated before the next run.

The global phase logs eval rows only at phase end (`iter: -1`), so no
within-phase learning curve exists for the global model.

`frames_total` **undercounts the global phase 4×**. `crl/ppo_continual.py:87`
sets `_frames_per_iter = n_envs * n_steps`, assuming one collect per iteration,
but the global phase collects `past_task_sample_n + 1 = 4`. **Do not quote
`frames_total`.**

Only `duel_v2_fix_seed0` carries a `progress.jsonl`, so `constraint_satisfaction`
is that one run. The newer directories hold `config.yaml`, `eval_matrix.json` and
`resource_usage.json` only.

## Superseded

`reports/final/gridworld/` holds the v1 figures. That benchmark is defective and
its numbers should not be used: the random floor sat at 0.742, eight tasks were
unsolvable, and slip correlated +0.552 with score. They are kept only because
the paper still cites them; `docs/BIGGRID_V2.md` records what changed and why.
