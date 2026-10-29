# Value-Constrained Consolidation

Code and run records for *Value-Constrained Consolidation for Continual
Reinforcement Learning*.

A policy that learns tasks in sequence carries one parameter vector for all of
them, so a gradient step that helps the current task is free to damage the
earlier ones. Existing methods act on **how** the policy may change. They
penalise parameter movement, replay stored transitions, or freeze components.
None of them states what the policy must still be able to **do**.

VCC states it. When task `k` arrives:

1. **Specialist phase.** Copy the deployed policy and train the copy on task `k`
   alone with ordinary PPO. Nothing about the earlier tasks constrains it, so the
   value it reaches is a fair estimate of what the network can achieve on that
   task. Freeze it.

2. **Consolidation phase.** Train the single deployed policy over the tasks seen
   so far, subject to not falling more than a set amount below the value the
   specialist reached:

   ```
   max_w   Σ_{i<k} ω_i V_i(π(w))
   s.t.    F_k = [ V_k(specialist) − V_k(π(w)) ]₊²  ≤  ε
   ```

   A single scalar multiplier `μ` enforces the constraint by projected dual
   ascent. No replay buffer is kept. Earlier environments are re-entered for
   fresh gradients instead.

---

## What is where

Everything is on one branch. Each experiment is a set of configs plus the run
records it produced plus the script that draws its figures.

| Experiment | Configs | Records and figures |
|---|---|---|
| Setting A, the established benchmark | not runnable here, see the caveat below | `reports/final/cka_rl/` |
| Setting B, fifty GridWorld tasks | `duel_v2_fix.yaml`, `biggrid50v2_*.yaml` | `reports/final/gridworld_v2/`, `reports/biggrid50v2/` |
| Setting B, environment access | `biggrid50v2_*_env.yaml` | `reports/final/env_access_full/`, `reports/gridworld_env_access/` |
| Setting C, five Atari games | `atari5*.yaml`, `atari_base_*.yaml` | `reports/final/atari_reversed/`, `reports/order_sensitivity/` |
| Ablations | `abl_*.yaml`, `biggrid50_*.yaml`, `sweeps/` | `reports/final/ablation/`, `reports/final/boot_vs_mc_gridworld/`, `reports/gridworld_ablations/` |
| Value-parameter landscape | `biggrid50v2.yaml` | `reports/final/landscape/` |
| The fifty-task set itself | `biggrid50v2.yaml`, `biggrid50v2.manifest.json` | `data/biggrid50v2.npz` |

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # torch, gymnasium, numpy, pyyaml, matplotlib
pip install -r requirements-viz.txt      # plotly + kaleido, only needed to redraw figures
```

Python 3.10 or newer. A GPU is optional. The runner falls back to CPU when none
is visible, which is fine for GridWorld and slow for Atari.

Atari additionally needs the ROMs:

```bash
pip install "gymnasium[atari,accept-rom-license]"
```

---

## Running an experiment

Every run goes through one entry point.

```bash
python -m experiments.run --config configs/<name>.yaml --seed 0
```

Useful flags:

| Flag | Effect |
|---|---|
| `--seed N` | Override the config's seed. Results in the paper use seeds 0, 1 and 2. |
| `--results-dir DIR` | Write somewhere other than the config's `results_dir`. |
| `--auto-resume` | Pick up from the newest checkpoint in the run directory. |
| `--resume-from FILE --resume-after K` | Resume from a specific checkpoint, having completed task `K`. |

### Setting B, the fifty-task GridWorld sequence

`configs/biggrid50v2.yaml` is the *task set*, the fifty generated tasks and their
parameters. It is not a method config. The method configs reference it through
`env.params.dataset_npz`.

```bash
python -m experiments.run --config configs/duel_v2_fix.yaml --seed 0     # VCC
python -m experiments.run --config configs/biggrid50v2_clear.yaml --seed 0
python -m experiments.run --config configs/biggrid50v2_recall.yaml --seed 0
python -m experiments.run --config configs/biggrid50v2_cka_rl.yaml --seed 0
python -m experiments.run --config configs/biggrid50v2_cbp.yaml --seed 0
python -m experiments.run --config configs/biggrid50v2_crelu.yaml --seed 0
python -m experiments.run --config configs/biggrid50v2_finetune.yaml --seed 0
python -m experiments.run --config configs/biggrid50v2_scratch.yaml --seed 0
```

The two `_env` configs are the environment-access ablation, in which the replay
baselines re-enter past environments instead of sampling a buffer.

Every method receives the same 204.8M environment frames. One collect is 32
parallel environments for 128 steps, so 4,096 frames, and the budget buys 50,000
collects per method. What differs is how each one spends them.

### Setting C, five heterogeneous Atari games

```bash
python -m experiments.run --config configs/atari5.yaml --seed 0          # VCC
python -m experiments.run --config configs/atari5_clear.yaml --seed 0
```

The `atari_base_*.yaml` configs train the single-game specialists that set the
normalisation ceiling. The `atari5_reversed*.yaml` configs are the task-order
sensitivity check reported in the appendix.

### Ablations

The `abl_*.yaml` configs vary the constraint, and
`configs/sweeps/eps_grid.yaml` drives the budget sweep:

```bash
python -m experiments.sweep --base configs/biggrid50v2.yaml \
    --grid configs/sweeps/eps_grid.yaml --seeds 0 1 2
```

### Regenerating the task set

```bash
python -m experiments.generate_biggrid_tasks --out configs/biggrid50v2.yaml --seed 20260930
```

The generator verifies with breadth-first search that every task is solvable
before it writes. `configs/biggrid50v2.manifest.json` records the seed, the
command and a checksum of the output, so a regenerated set can be compared
against the one the paper used.

---

## Where results are stored

Each run writes one directory under `results/`, named after the config and seed:

```
results/<run_name>/
  run.json              written once at start, never mutated
  progress.jsonl        append-only, one JSON object per line
  status.json           overwritten each heartbeat
  eval_matrix.json      the lower-triangular retention matrix
  checkpoints/
    after_task{k}.pt        the consolidated policy after task k
    local_after_task{k}.pt  the frozen specialist for task k
    optim_after_task{k}.pt  optimiser and dual state, so resume is exact
```

`progress.jsonl` is the record that matters. It carries one typed row per event,
including an `eval` row per evaluation and a `dual` row per constraint refresh
holding the multiplier, the measured shortfall and the budget. Every number in
the paper derives from these files and nothing else.

[`LOGGING_CONTRACT.md`](LOGGING_CONTRACT.md) specifies every record type and
field.

`results/` is in `.gitignore`. The run records behind the published figures are
committed separately, under `reports/`, so the figures can be redrawn without
retraining anything.

---

## Figures

Each figure set has its own directory under `reports/final/` with a README
describing what it shows and a script that draws it:

```bash
python reports/final/gridworld_v2/make_figures.py      # Setting B
python reports/final/atari_reversed/make_figures.py    # Setting C
python reports/final/ablation/make_figures.py          # ablations
python reports/final/env_access_full/make_figures.py   # environment access
python reports/final/cka_rl/make_figures.py            # Setting A table
python reports/final/landscape/make_figures.py         # landscape surfaces
```

Each writes 300 dpi PNG into `png/` and vector SVG into `svg/` beside itself.
Those output directories are not committed, so the first run of a script creates
them. The scripts read only the committed records under `reports/`, so they work
offline and need no GPU.

The landscape figure is a three-stage pipeline, since the surface is expensive:

```bash
python reports/final/landscape/run_trajectory.py --task 5 --phase local  --stride 10
python reports/final/landscape/run_trajectory.py --task 5 --phase global --stride 5
python reports/final/landscape/project.py --task 5 --n 31     # a few minutes
python reports/final/landscape/make_figures.py                # seconds
```

---

## Changing the configuration

Configs are YAML and carry no magic numbers in code. The fields that matter for
the method:

| Field | Meaning |
|---|---|
| `trainer.eps` | The value budget `ε`. The allowed shortfall is `√ε`. |
| `duals.lr` | Step size `η` of the dual ascent on `μ`. |
| `duals.max_value` | Cap `μ_max`, which keeps the dual update stable. |
| `duals.init` | Starting multiplier, `0` everywhere. |
| `ppo.value_mode` | `mc` for a Monte-Carlo shortfall, `bootstrap` for the truncated window. |
| `ppo.window_H` | Window length `H`, used when `value_mode` is `bootstrap`. |
| `ppo.constraint_every` | Iterations between dual refreshes. |
| `ppo.constraint_episodes` | Episodes averaged per shortfall estimate. |
| `ppo.past_task_sampling` | `all`, `sample` for a uniform draw, or `worst`. |
| `ppo.past_task_sample_n` | Earlier tasks drawn per consolidation iteration. |
| `ppo.task1_iters`, `ppo.local_iters`, `ppo.global_iters` | Budget for the first task, the specialist phase and the consolidation phase. |
| `ppo.n_envs`, `ppo.n_steps` | One collect is `n_envs × n_steps` environment frames. |
| `experiment.seed`, `experiment.results_dir` | Seeding and output location. |

The settings differ. Setting B uses `ε = 0.09`, `η = 0.1`, `μ_max = 5`, draws
three earlier tasks per iteration and refreshes the dual every five. Setting C
uses `ε = 0.04`, `η = 0.3`, `μ_max = 5`, sums over every earlier task, which is
affordable because that sequence holds five of them, and refreshes every twenty.
Setting A uses `ε = 0.04`, `η = 1.0`, `μ_max = 100` and draws one earlier task.

Seeding is enforced centrally in `crl/seeding.py`, which sets Python, NumPy and
Torch seeds and puts cuDNN in deterministic mode. A run is reproducible from its
`run.json` alone.

---

## Tests

```bash
pytest
```

Thirteen test modules covering the dual controller, the exact and rollout value
estimators, the Bellman-residual estimator, the GridWorld and BigGrid
environments, config loading and seeding, the multi-head network, the PPO
trainer and the baselines.

---

## Two things worth knowing

**The method is called `DUEL` in parts of the code.** That was its working name
before it was renamed to VCC. The configs, the summary keys in the committed
records and some figure labels still use it. `DUEL` and `VCC` are the same
method throughout; nothing else changed with the name.

**Setting A is reported here but not runnable from this repository.** That
comparison was produced by adding our method as a row inside the released
codebase of the benchmark's authors, which uses their harness and their task
definitions for the Space Invaders and Freeway mode sequences. The environment
layer here has no mode parameter and cannot construct those sequences. What this
repository carries is the output side in full: per-mode return files for all eight
Freeway modes and all ten Space Invaders modes, the phase summaries, the
retention history, the transcribed benchmark table, and the script that draws
the comparison. The Meta-World figures for the other methods are the published
ones. The hyperparameters we used are in the table above, so the row can be
reproduced inside that harness.

---

## Source layout

```
crl/                    the library
  trainer.py              the alternating specialist/consolidation loop
  ppo/trainer.py          PPO, and the constrained consolidation step
  duals/controllers.py    the projected dual ascent on μ
  estimators/             exact, rollout and truncated-window value estimates
  envs/                   GridWorld, BigGrid, Atari, MinAtar
  policies/               MLP and CNN actor-critics, plus baseline architectures
  run_logging.py          the logging contract emitter
experiments/            entry points: run.py, sweep.py, the task generator
configs/                one YAML per run, plus the generated task set
data/                   the generated fifty-task file
report/acviz.py         the shared plotting template every figure script imports
reports/final/<name>/   a README, a figure script, and the records it reads
reports/<run>/          committed run records, one directory per run
tests/                  pytest suite
```
