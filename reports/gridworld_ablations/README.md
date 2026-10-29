# GridWorld ablations (10-task, single seed, reduced budget, DUEL)

Two self-contained experiments on a 10-task GridWorld slice. Everything needed to
regenerate figures is here: per-run `eval_matrix.json` + `run.json` (normalization
reference) + `resource_usage.json` (phase wall-clock), plus a precomputed
`summary.json` with every number below.

**Caveats (put on every figure):** single seed, length 10, reduced budget —
directional, not a full-scale result. Compare **within** each experiment only;
the ablation and the sweep used different local budgets, so do not cross-compare
their absolute PERF.

Normalized scores: 0 = random policy, 1 = task solved. Metrics via
`analysis.contract_metrics` (PERF = mean of the final eval-matrix row, BWT, forgetting).

---

## 1. Constraint on vs off (`constraint_ablation/`)

Two DUEL runs identical except `duals.max_value` (5.0 = constraint on; 0.0 = μ
pinned to 0 = current-task floor off). `con/` and `mu0/`.

| | PERF | BWT | Forgetting |
|---|---|---|---|
| **constraint on** | **0.610** | +0.128 | 0.083 |
| **μ = 0 (off)** | **0.338** | +0.272 | 0.081 |

The constraint nearly doubles PERF. μ=0's higher BWT is an artifact of weaker
acquisition (tasks start near zero, so they have more room to drift up), not a win.

**Figures to make:**
- Two **retention-matrix heatmaps** (con and μ=0), lower-triangular, shared colour
  scale. `summary.json.constraint_ablation.{con,mu0}.norm_matrix` is the 10×10
  (nulls above the diagonal). The **diagonal** (`.diagonal`) is the key read: con
  learns each task on arrival; μ=0 is ~0 everywhere except task 0.
- A small **PERF / BWT bar pair**, con vs μ=0.

## 2. Global rollout-depth sweep (`gns_sweep/`)

Five DUEL runs identical except `global_n_steps` (the global-consolidation PPO
rollout depth) ∈ {1,10,20,40,80}. `value_mode` stays `mc`, `window_H` unchanged.
Run sequentially on one node so the wall-clock is comparable.

| global_n_steps | PERF | BWT | global-phase wall | collect-steps/iter |
|---|---|---|---|---|
| 1 | +0.001 | −0.036 | 6.9 min | 32 |
| 10 | +0.058 | +0.028 | 15.4 min | 320 |
| 20 | +0.169 | −0.034 | 24.1 min | 640 |
| 40 | +0.280 | +0.061 | 39.1 min | 1280 |
| 80 | +0.484 | +0.197 | 70.0 min | 2560 |

PERF rises monotonically with depth and the global-phase wall scales ~linearly
(≈ 6 min fixed + ~0.8 min per step of depth). Not saturated at 80.

**Figures to make:**
- **PERF vs rollout depth** (line/scatter).
- **Global-phase wall vs rollout depth** (line) — the runtime story. Optionally a
  twin-axis PERF-and-runtime vs depth. `collect-steps/iter` (= depth × n_envs) is
  the hardware-independent cost if you want a machine-independent x or y.

Numbers verifiable from the raw files; `summary.json` is the convenience copy.
