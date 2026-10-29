# Order sensitivity: V5 (min-max) vs CLEAR against the Joint ceiling

**Headline.** The task order decides which method retains more. On the four
tasks learned *before* the last one, measured against the fixed Joint ceiling:

| Order | V5 (min-max) | CLEAR |
|-------|-------------:|------:|
| Canonical | 43% | 88% |
| Reversed  | **83%** | **43%** |

The lines cross. Neither method is order-robust, and either order taken alone
would have supported the opposite conclusion. This report is focused on the
**reversed** order; the canonical order appears in Figure 1 and as the second
variant of Figure 5.

## Why the Joint ceiling is the denominator

**Joint** is one budget-matched model trained on all five games at once. It is
*order-independent*, so it gives the same reference in both orders and makes the
cross-order comparison clean.

**Local** (the single-task specialist) is order-dependent: a task-1 game has no
local phase, and later locals start from the evolving global. SpaceInvaders is
1132.2 canonical against 588.5 reversed for the same game. Retention against
Local therefore mixes forgetting with reference drift, so Local survives here
only as a raw-score reference in Figure 3.

## Why "previously learned tasks" is the headline statistic

The last task in a sequence has had nothing trained after it. Its final score
measures capacity, not retention. Including it lets a method that simply
overfits the final task post a high mean.

In the reversed order CLEAR does exactly that. Q\*bert is learned last and runs
to **15350.8**, which is **3.6x** the Joint ceiling of 4261.5. That single cell
pulls CLEAR's five-game mean to 107% while it is forgetting everything else:
Breakout 8%, Pong 54%, Boxing 55%, SpaceInvaders 56%. The five-game mean is
reported below for completeness but it is not the number to quote.

| Order | Metric | V5 | CLEAR |
|-------|--------|---:|------:|
| Reversed | prior-task mean (4 tasks) | **83%** | **43%** |
| Reversed | all-five mean | 86% | 107% |
| Canonical | prior-task mean (4 tasks) | **43%** | **88%** |
| Canonical | all-five mean | 51% | 88% |

## Provenance

Seed 0, greedy-100 evaluation, plain `impala_ac_multihead` net, same five games.
**Single seed, so no error bars are drawn and none are invented.**

CLEAR is apples-to-apples with V5: identical net, task set, thresholds, and PPO
hyperparameters; frame-matched budget; equal replay buffer
(`clear_snapshot_batches=4`, `clear_replay_task_per_step=8`). The only method
difference is CLEAR's replay plus policy/value cloning against V5's min-max dual
constraint.

Orders:

- **Canonical** Q\*bert → Pong → Breakout → Boxing → SpaceInvaders
- **Reversed** SpaceInvaders → Boxing → Breakout → Pong → Q\*bert

Sources, one `eval_matrix.json` per run:

| Run | Path |
|-----|------|
| V5 canonical | `results/atari5_v5_seed0` |
| V5 reversed | `results/atari5_v5_order2_seed0` |
| CLEAR canonical | `results/atari5_v5_clearA_equal_seed0` |
| CLEAR reversed | `joint-training run records, atari5_clear_order2_seed0` |

`results/` is gitignored and the reversed and CLEAR runs live only on the
cluster, so the numbers are cached in **`data.json`** and the figures rebuild
anywhere from that file alone. Re-transcribe `data.json` if a run is rerun.

## Exact data (raw greedy-100 scores)

Reversed order, final row of each forgetting matrix:

| Game | Local-R | V5-R | CLEAR-R | Joint |
|------|--------:|-----:|--------:|------:|
| SpaceInvaders | 588.5 | 711.7 | 505.9 | 905.8 |
| Boxing | 98.9 | 55.4 | 36.8 | 67.5 |
| Breakout | 363.9 | 199.8 | 23.9 | 285.4 |
| Pong | 21.0 | 21.0 | 11.1 | 20.7 |
| Q\*bert *(last)* | 4341.0 | 4270.2 | 15350.8 | 4261.5 |

Canonical order, for the Figure 1 contrast and the Figure 5 variant:

| Game | Local-C | V5-C | CLEAR-C | Joint |
|------|--------:|-----:|--------:|------:|
| Q\*bert | 4467.8 | 4075.0 | 4350.0 | 4261.5 |
| Pong | 20.0 | 19.8 | 21.0 | 20.7 |
| Breakout | 132.7 | 51.8 | 0.0 | 285.4 |
| Boxing | 94.0 | **-25.8** | 100.0 | 67.5 |
| SpaceInvaders *(last)* | 1132.2 | 765.8 | 800.0 | 905.8 |

Boxing V5-canonical is **negative**: V5 forgot Boxing worse than random (~0)
under that order. It is drawn below the zero line and never clipped.

## Figures

All five render with Plotly plus Kaleido through `report/acviz.py`, so they
follow the academic template: white ground, Tufte spine, horizontal grid only,
palette in series order (V5 blue, CLEAR amber, Joint green, Local neutral grey),
PNG at 300 dpi plus SVG.

1. **`fig1_order_contrast`** — the order contrast. Slope chart of prior-task mean
   retention, canonical against reversed. The two lines cross. The only figure
   here that shows both orders in one panel.
2. **`fig2_reversed_retention`** — what survives. Per-game retention against the
   ceiling in the reversed order, tasks top to bottom in learning order. The
   final task sits in its own panel on its own scale, because its score is not a
   retention measurement.
3. **`fig3_reversed_raw_scores`** — the absolute numbers. Small multiples, one
   panel per game with its own y-axis always including zero, four bars (Local,
   V5, CLEAR, Joint). Raw scales differ by roughly 700x, so a shared axis would
   flatten four of five panels.
4. **`fig4_reversed_trajectories`** — the timing. Each task's retention at every
   later training phase. The forgetting matrix read as a trajectory, which is
   what makes the *when* visible: CLEAR's Breakout goes 106% → 19% → 8% while
   V5's dips to 55% and recovers to 70%. Q\*bert is absent because it is learned
   last and has no later phase.
5. **`fig5_matrices_reversed`** / **`fig5_matrices_canonical`** — the full
   forgetting matrices for both methods, as two variants of one figure. Cells are
   fractions of the ceiling on one shared diverging scale (-50% to 250%, centred
   on the ceiling, red below and blue above) so switching variants is not
   misleading. CLEAR's reversed Q\*bert cell saturates at the top of the scale and
   keeps its printed number.

Dropped from the previous version of this report: the two retention-vs-Local
figures (the Local denominator is order-dependent and the vs-Joint view is the
cleaner comparison) and the raw-score matrix (its colour was normalised
per-column, so the shade carried no cross-panel meaning).

## Files

- Data: `data.json`
- Script: `make_figures.py` (also writes the dashboard copies and appends to
  `report/manifest.json`; pass `--no-dashboard` to skip that)
- PNG: `png/fig{1..5}*.png`
- SVG: `svg/fig{1..5}*.svg`

```bash
python reports/order_sensitivity/make_figures.py
python report/build_dashboard.py
python report/verify_dashboard.py
```
