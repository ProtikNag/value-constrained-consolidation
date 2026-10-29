# Logging contract

**Every method — ours, CompoNet, CKA-RL, CLEAR, any baseline — emits this same
shape.** One metrics module then computes PERF, FWT, BWT, forgetting, retention
and compute cost for all of them from the same files.

This exists because the alternative is three incompatible log piles and someone
hand-merging CSVs the night before the deadline. If a method cannot emit a field,
it writes `null`; it does not invent its own key.

---

## 1. Layout

```
results/<run_name>/
  run.json              # written once at start, never mutated
  progress.jsonl        # append-only, one JSON object per line, flushed every write
  status.json           # overwritten each heartbeat; the "is it alive" file
  eval_matrix.json      # lower-triangular raw scores (kept for back-compat)
  checkpoints/
    after_task{k}.pt        # global/consolidated model after task k
    local_after_task{k}.pt  # per-task specialist reference (ours only)
    optim_after_task{k}.pt  # optimiser + dual state, so resume is exact
```

`progress.jsonl` is append-only and flushed on every write. A killed job leaves a
valid file up to the last line. Never rewrite it.

## 2. `run.json`

```json
{
  "run_name": "atari5_rev_ours_seed0",
  "method": "minmax | clear | componet | cka_rl | baseline | joint",
  "seed": 0,
  "git_sha": "ab2e019",
  "started": "2026-09-14T17:02:11",
  "env_family": "atari | biggrid | gridworld",
  "tasks": ["SpaceInvaders", "Boxing", "Breakout", "Pong", "Qbert"],
  "task_order": "reversed",
  "reference": {
    "random":  [148.0, 0.1, 1.7, -20.7, 163.9],
    "ceiling": [905.8, 67.5, 285.4, 20.7, 4261.5]
  },
  "config": { "...": "the full resolved config, verbatim" }
}
```

`random` and `ceiling` are written **at run start** so every downstream metric is
reproducible from the run directory alone, with no lookup into another file that
may have changed since.

## 3. `progress.jsonl` record types

Every record carries `type`, `t_wall` (seconds since run start), and
`frames_total` (cumulative env frames, or env steps for GridWorld).

### `phase_start` / `phase_end`

```json
{"type": "phase_end", "task_idx": 2, "task": "Breakout",
 "phase": "task1 | local | global",
 "iters": 3000, "frames_phase": 15360000, "wall_s_phase": 4120.5,
 "t_wall": 51230.1, "frames_total": 41000000}
```

Compute cost comes from `wall_s_phase` and `frames_phase` summed per phase. Both
are required: wall-clock is machine-dependent, frames are not.

### `eval` — the one that matters most

```json
{"type": "eval", "task_idx": 2, "phase": "local", "iter": 600,
 "evaluated_on": 2, "evaluated_on_task": "Breakout",
 "raw": 143.2, "normalized": 0.497,
 "episodes": 10, "greedy": true, "seen": true,
 "t_wall": 49001.3, "frames_total": 38400000}
```

- `normalized` is `(raw − random[j]) / (ceiling[j] − random[j])`, using the
  values frozen in `run.json`.
- `seen` is false when the task has not been trained yet.
- `greedy` is true for reported evaluation, false for the on-policy stochastic
  value used inside the constraint.

**Three eval cadences are required, and each feeds a different metric:**

| Cadence | When | Feeds |
|---|---|---|
| **within-phase** | every `eval_every` iters of task `i`'s own phase, `evaluated_on == i` | **FWT** (the learning-curve AUC) |
| **end-of-phase, all seen** | at every `phase_end`, one record per seen task | BWT, forgetting, retention, the matrix |
| **end-of-run, all tasks** | after the last task, `eval_episodes: 100`, greedy | PERF, final row, Table-3-style success |

The within-phase cadence is the one that was missing before
(`eval_every: 0`), and it is **not optional**: without it there is no `p_i(t)`,
and forward transfer cannot be computed afterwards from a finished run.

### `dual` — ours only

```json
{"type": "dual", "task_idx": 3, "iter": 400,
 "mu": 2.71, "lambda": [0.0, 0.41, 1.02],
 "shortfall_current": 0.083, "shortfall_past": [0.0, 0.11, 0.26],
 "coeff_current": 0.45, "grad_share_past": 0.38,
 "t_wall": 60110.2}
```

`grad_share_past` is the normalised past-task share of the actor gradient. This
is the quantity that caught F16 (past-task starvation, past tasks at 0.003% of
the update). It stays logged.

### `note`

```json
{"type": "note", "text": "resumed from checkpoints/after_task2.pt", "t_wall": 0.0}
```

## 4. `status.json` — heartbeat

Overwritten at least every 60 s so a run can be checked mid-flight without
parsing the whole log:

```json
{"task_idx": 3, "task": "Boxing", "phase": "global",
 "iter": 1400, "iters_target": 2000,
 "frac_done": 0.62, "t_wall": 60110.2, "eta_s": 18400,
 "last_eval": {"Qbert": 0.91, "Pong": 1.01, "Breakout": 0.70},
 "alive": "2026-09-15T04:11:02"}
```

## 5. Checkpoints and resume

Write after **every task**, not every run. A failure on task 4 must resume from
task 3, never from task 0.

```bash
python -m experiments.run --config <cfg> --seed 0 \
       --resume-from results/<run>/checkpoints/after_task3.pt --resume-after 3
```

Resume appends a `note` record and continues `progress.jsonl`. It must restore
the optimiser and dual state, not just the weights, or the run silently changes
method at the resume point.

## 6. Metrics derived from this, and nothing else

| Metric | Derivation |
|---|---|
| **PERF** | mean `normalized` over all tasks in the end-of-run eval |
| **FWT** | `FT_i = (AUC_i − AUC_i^b) / (1 − AUC_i^b)`, `AUC_i` = mean `normalized` over task `i`'s within-phase evals, `AUC_i^b` from the from-scratch baseline run's within-phase evals on task `i`. Reported as the mean over tasks. |
| **BWT** | per task `j`, `normalized(final) − normalized(just-learned)`; the full lower triangle gives BWT at every phase |
| **Forgetting** | per task `j`, `max over phases − final` |
| **Retention** | `normalized(final)` against the ceiling, per task |
| **Compute** | Σ `wall_s_phase` and Σ `frames_phase` |

**Forward transfer needs a paired baseline run.** For each benchmark there must
be a `method: "baseline"` run that trains each task from scratch, with the same
within-phase eval cadence. Without it `AUC_i^b` does not exist and FWT is `null`
— not estimated, not approximated.

For Atari, `experiments/train_expert.py` already trains from-scratch single-task
experts from a shared init. **Check whether their learning curves were logged
before training new ones** — that is the baseline, and it may already be on disk.

## 7. Rules

- Never delete or rewrite `progress.jsonl`. Append only.
- Never compute a metric that the log does not support. Write `null`.
- Never mix runs with different `git_sha` into one aggregate without saying so.
- Every number that reaches a figure is traceable to a line in a `progress.jsonl`.
