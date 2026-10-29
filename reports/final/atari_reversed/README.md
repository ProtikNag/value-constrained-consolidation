# Line 1: five-game Atari sequence, reversed order

Reversed order **SpaceInvaders → Boxing → Breakout → Pong → Q\*bert**. Seed 0,
greedy-100 evaluation, plain `impala_ac_multihead` net, identical task set,
thresholds, PPO hyperparameters and frame-matched budget for both methods.
Single seed, so no error bars.

Order sensitivity is reported in the paper as an observed phenomenon, with the
two orders' retention matrices as the evidence. Its root cause and any mitigation
are out of scope here, so the canonical order is **not** rebuilt in this folder;
`reports/order_sensitivity/` keeps that material.

## Status, 2026-09-18

**Complete.** All four runs have finished all five tasks, so every number in
this folder is measured and nothing is extrapolated.

| Method | Tasks finished | Source |
|---|---:|---|
| DUEL (ours) | 5 / 5 | `data_live.json`, post-threshold-fix |
| CLEAR | 5 / 5 | `../../order_sensitivity/data.json` |
| CKA-RL | 5 / 5 | `data_live.json`, post-threshold-fix |
| CompoNet | 5 / 5 | `data_live.json`, post-threshold-fix |

Ours' final row against each game's joint ceiling.

| | SpaceInv | Boxing | Breakout | Pong | Q\*bert |
|---|---:|---:|---:|---:|---:|
| raw | 762.6 | 10.5 | 235.6 | 17.7 | 5195.2 |
| vs ceiling | 84% | **16%** | 83% | 86% | 122% |

Four of five games end above 83%. **Boxing is the casualty**, decaying
monotonically through the sequence at 68.7 → 59.8 → 42.5 → 10.5 and ending at
16% of its ceiling. No other method holds four games this high, and none of the
others learns the final task as well; ours' Q\*bert diagonal of 5195.2 clears
both the 4260 threshold and the 4750 local specialist.

Caveats that have to travel with any of these figures.

**CLEAR's run predates the threshold change**, which does not affect it. CLEAR
never reached its thresholds on the first four games, so it trained to its full
budget on them regardless. Only Q\*bert cleared its threshold, and Q\*bert is
the last task, so nothing earlier is affected. CLEAR does not need rerunning.

**The threshold fix landed hard.** Ours' SpaceInvaders went 588.5 → 1318.1 on
task 1, which is 146% of the joint ceiling. That is the single largest change
in this tier and it is why the old and new runs are never averaged together.

**Ours' Breakout shows the value-vs-score gap, then recovers from it.** The
local phase peaked at 396, the value-constraint shortfall reached ≈0, and the
greedy diagonal still fell to 77.6 during consolidation. The constraint is
satisfied on `V` while the score collapses. It then climbed back
non-monotonically across the two following phases, 77.6 → 144.5 → 235.6,
finishing at 83% of its ceiling. Both halves belong in the paper, the
value-versus-score mismatch as a limitation of constraining value, and the
recovery as evidence that consolidation reclaims a task the constraint had let
slip.

**Model selection is the final iterate, not the best checkpoint.** This run kept
the final global of each phase rather than its best checkpoint, because the
global trainer had no best-checkpoint facility when it ran. That has since been
added and verified, so future runs will differ. Worth disclosing, since Boxing's
decay is exactly the kind of thing a best-checkpoint rule would have caught.

**A forgotten task cannot be patched afterwards.** Post-hoc recovery of Boxing
from the final global was tried three ways and all three were discarded as
net-negative. Ten iterations of pure Boxing moved it +2.3 while costing Breakout
159 and Q\*bert 1648. A hundred iterations of pure Boxing bought +36 on Boxing
and lost 4424 on Q\*bert. A hundred iterations of a 40/60 Boxing-to-rest mix
gained 23.5 on Boxing for a net −0.81 in normalised terms. The final matrix
stands unchanged, and the ablation is worth reporting because it argues that
retention has to be maintained during the sequence rather than repaired at the
end.

### What the two finished baselines did

The two failures are opposite, and between them they frame the stability and
plasticity trade the method is for.

**CKA-RL forgot almost everything.** Its final row is Space Invaders 212.6,
Boxing −15.5, Breakout 6.7, Pong −21.0, Q\*bert 4420.5. Boxing and Pong finish
**below a random policy**. Only the last-learned game survives, which leaves
average performance at 0.01 against a ceiling of 1.00 and backward transfer at
−0.99, the worst of the four.

**CompoNet forgot nothing and stopped learning.** Its lower triangle is
constant by construction, since components freeze, so backward transfer is
exactly 0.00 on every task. The cost shows up on the diagonal: Breakout reached
only 99.2 against a threshold of 285, and Q\*bert scored **0.0**, never learned
at all. Average performance lands at 0.66, and the whole of the shortfall is the
one task it never learned.

Ours sits between the two. On the final row it holds four of five games above
83% where CKA-RL holds none and CompoNet holds three, and it is the only method
that both retains earlier tasks and learns the last one.

| | Backward transfer | Forgetting | Average, all | Average, before last |
|---|---:|---:|---:|---:|
| **DUEL (ours)** | **−0.26** | 0.40 | **0.78** | 0.67 |
| CLEAR | −0.62 | 0.62 | 1.07 | 0.43 |
| CKA-RL | −0.99 | 0.99 | 0.01 | −0.25 |
| CompoNet | +0.00 | **0.00** | 0.66 | **0.83** |

Read the two averages together. CompoNet leads before the last task precisely
because frozen components cannot be overwritten, and its 0.66 overall is what
that costs. CLEAR's 1.07 is an artefact of a Q\*bert score 3.6× its ceiling
while everything before it decayed, which is why it falls to 0.43 once the last
task is excluded. Ours is the only column strong in both.

## Normalisation

Everything in this folder is on **one scale, `score / Joint ceiling`**. The
retention matrices always used it and `transfer_table` now does too, so a cell
in a matrix and the aggregate under it are the same quantity.

The random floor is **not** 0 on this scale. It is 16% on Space Invaders, 4% on
Q\*bert, near 0% on Boxing and Breakout, and **−100% on Pong**, whose random
policy scores −20.7 against a ceiling of 20.7. So CKA-RL's Pong cell at −101%
means *at random*, not far below it, and a reader has to be told that once.

## Figures

| Stem | What it shows |
|---|---|
| `forgetting_matrices` | Retention matrices, 2×2, all four methods |
| `final_scores` | Per-game score after the final task, five series |
| `transfer_table` | Five aggregate metrics per method, no per-task rows |

`backward_transfer_matrix` and `compute_cost` were dropped. Compute was measured
across different GPUs per method, so the comparison was never fair.

## What the dynamics log shows, and why it is not a figure

`consolidation_dynamics.json` and `boxing_topup_ablation.json` are tracked and
deliberately unplotted. Four figures were built from them and all four cut.

- **retention_tradeoff, forward_transfer, boxing_topup.** Every number already
  appears in `transfer_table`. A second rendering of the same values is not a
  second piece of evidence.
- **constraint_activity.** The one-sided hinge contributing nothing above zero
  is true by construction, so plotting it establishes nothing a reader could
  have doubted. Its only empirical content is that the deployed policy does
  rise above its expert, 41%, 7% and 22% of the three logged phases. The claim
  that would justify, that this headroom makes positive backward transfer
  reachable, cannot be made on this tier, where backward transfer is −0.26 and
  Boxing ends at 16% of ceiling.

**One finding from the log does belong in the paper, as a sentence.** The dual
multiplier is active in every one of the 88 logged steps of all three
consolidated tasks here, saturating at its cap of 5.0. In the GridWorld runs it
sits at zero in roughly 98% of steps. The same mechanism is slack in one tier
and pinned to its ceiling in the other, which is direct evidence for the claim
that each task's multiplier is set by the optimisation rather than chosen in
advance. That belongs in the method section, not in a figure.

Mechanism and outcome sit on different tiers, and it is worth being aware of
that. GridWorld carries the outcome, +0.358 backward transfer, with a
multiplier that is near-zero throughout. Atari carries the mechanism, a
saturated multiplier, with backward transfer at −0.26.

## Nothing is stood in for

Every cell in this folder is measured. Earlier drafts padded a run's unreached
tasks at ours' value, which was defensible while the baselines were the ones
still going but became untenable once ours was the run in flight, since the only
complete five-task run of ours available to pad from was the pre-threshold-fix
one, whose task 1 scored 588.5 against 1318.1. The padding was removed then and
is not needed now.

The machinery that handles partial runs is still in `make_figures.py` and is
driven by the data, so a future partial run renders correctly without an edit.
An unreached task is left absent rather than stood in for, the caption naming
which runs are short derives from the row counts rather than being typed, and
every aggregate in `transfer_table` is computed over the tasks its own run has
finished.

**The local-specialist reference is only partly refreshed.** The cached vector
belongs to the pre-threshold-fix run. Task 1 has no prior global to consolidate
against, so the specialist's score and that run's own task-1 diagonal are one
measurement rather than two, and in the cached run both read 588.5 to the
decimal. `final_scores` therefore takes task 1 from the live diagonal, 1318.1,
computed at render time rather than typed. The identity holds for no other task
(Boxing 98.9 against 96.85, Breakout 363.9 against 293.58, Q\*bert 4341.0
against 4270.25), so **Boxing, Breakout, Pong and Q\*bert still carry pre-fix
specialist values** and can only be corrected by re-running the specialists.

**The score axis runs below zero where the data does.** CKA-RL finishes Boxing
at −15.5 and Pong at −21.0, both under the random floor. A zero-pinned axis drew
those two bars with no height at all, which read as *not reached* against a
caption saying exactly that. The floor now follows the data.

## Forward transfer

`fwt.json` carries the AUC form, `FWT_i = (AUC_i − AUC_i^b) / (1 − AUC_i^b)`,
computed from each task's local learning curve against a from-scratch
single-task run. **Use the `fresh_baseline` block.** Its baselines run the exact
continual configuration, `configs/atari_base_<Game>.yaml` matching
`configs/atari5_reversed.yaml`, rather than the earlier `experts/` runs the
top-level blocks used.

Per game, on the expert-peak scale.

| | SpaceInv | Boxing | Breakout | Pong | Q\*bert | matched subset |
|---|---:|---:|---:|---:|---:|---:|
| **DUEL (ours)** | — | — | **+0.17** | — | **+0.08** | **+0.13** |
| CKA-RL | −0.28 | −2.10 | −2.15 | — | −0.07 | −1.11 |
| CompoNet | −0.34 | −3.96 | −1.99 | — | −0.26 | −1.13 |
| CLEAR | — | — | — | — | — | no curves logged |

Three reasons for the blanks. **Ours has no curve for Space Invaders or
Boxing**, its run having resumed from task 3, which is a logging gap rather than
a result. **Pong is ill-conditioned for every method**, since the from-scratch
baseline reaches the normalised ceiling before the comparison window opens,
leaving `1 − AUC_b < 0.05`. **CLEAR logged no learning curves at all.**

`transfer_table` shows the matched subset, Breakout and Q\*bert, the only games
all three runs have a usable curve for. Per-method means over different game
sets are not comparable to each other.

### CLEAR and CKA-RL are different runs

Worth stating plainly, because the two have already been confused once. The
`cka_rl` entry in `data_live.json` was labelled `"CLEAR / CKA-RL"`, and that
label was read as meaning the CKA-RL run *is* the CLEAR baseline. It is not.

| | source | final row |
|---|---|---|
| CKA-RL | `results/atari5_rev_cka_rl_seed0` | 212.6, −15.5, 6.7, −21.0, 4420.5 |
| CLEAR | `joint-training run records, atari5_clear_order2_seed0` | 505.9, 36.8, 23.9, 11.1, 15350.8 |

The label is now `"CKA-RL"` and the data file carries a `_method_naming` note.
CKA-RL's forward transfer is computed and in the table at −1.11.

### CLEAR's forward transfer is wanted

The blank cell is a real gap, not a presentational choice, and the number is
wanted. `make_figures.py` already lists CLEAR in `FWT_KEY` and its lookup
tolerates the missing entry, so the cell fills itself the moment a `"clear"`
entry appears under `fresh_baseline.<ceiling>.per_method` in `fwt.json`. No
figure code needs touching.

**Check before rerunning.** The reversed CLEAR run is
`joint-training run records, atari5_clear_order2_seed0`, which is where the
retention numbers in this folder already come from. It has not been inspected
for per-iteration logs. What the AUC form needs is rows carrying a
`greedy_score` during each task's own training phase, the same cadence the other
three reversed runs used, roughly every 200 iterations. If those rows exist the
number is a computation rather than a run.

Only if they do not is a rerun needed, at roughly 1.5 A100-days. Note that it
would not widen the matched subset, which is capped at Breakout and Q\*bert by
our own missing Space Invaders and Boxing curves and by Pong being
ill-conditioned for every method. CLEAR would be a fourth column over the same
two games.

### Which ceiling, and why not the flattering one

`fwt.json` offers two normalisations and they disagree on magnitude.

| matched subset | ours | CKA-RL | CompoNet |
|---|---:|---:|---:|
| expert peak (**used**) | +0.13 | −1.11 | −1.13 |
| threshold | +0.59 | −2.05 | −5.01 |

**The expert peak is used, and the reason is conditioning, not preference.**
The metric divides by `1 − AUC_b`, and `fwt.json` already nulls anything under
0.05.

| | AUC_b | 1 − AUC_b | ours FWT |
|---|---:|---:|---:|
| Threshold, Breakout | 0.526 | 0.474 | +0.18 |
| Expert, Breakout | 0.537 | 0.464 | +0.17 |
| **Threshold, Q\*bert** | 0.780 | **0.220** | **+1.00** |
| Expert, Q\*bert | 0.181 | 0.819 | +0.08 |

Q\*bert's threshold is 4,260 and ours scored 5,195, so under the threshold scale
the normalised score clips at 1 for the whole window and the metric returns
exactly +1.00. That is saturation, not perfect transfer, and 0.220 is heading
toward the cliff the file already guards against. Under the expert ceiling of
18,000 the same phase reads +0.08.

Breakout agrees to 0.01 across the two scales, so **the entire four-fold
difference in the matched-subset mean comes from that one saturated cell.** The
threshold scale being more favourable to us is therefore an artefact rather than
a signal, and the principled choice and the conservative one coincide.

**On the folder carrying two scales.** Retention and forward transfer measure
different things against different references, which is coherent. Retention asks
how much of the jointly-achievable score survives, so the Joint ceiling is right.
Forward transfer asks how much faster than learning the task alone, and its
formula already requires a from-scratch curve in both numerator and denominator,
so the from-scratch peak keeps that quantity on one reference. Mixing scales
*within* a quantity was the earlier defect; using the right reference for each
quantity is not.

### Why every forward-transfer number in this project carries `*`

The from-scratch baseline the AUC form needs is being recomputed, here and in
the GridWorld tier, so treat all of them as provisional until it lands.

Note this supersedes an earlier claim in this file that Atari forward transfer
was *not measurable at all*. That was true of the zero-shot form, since each
task has its own head that stays at its initialisation until the task arrives
and no task is evaluated before training. The AUC form does not need a zero-shot
evaluation, only the within-phase curve, which three of the four runs have.

## Rebuild

```bash
python reports/final/atari_reversed/make_figures.py
```

Reads `data_live.json` for the three live methods and the cached transcription
for CLEAR. Re-transcribe `data_live.json` as the run advances; the figures and
every count in their captions follow the data.
