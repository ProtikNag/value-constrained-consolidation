# CLEAR/RECALL with VCC's environment access (10-task GridWorld, seed 0)

Does VCC's edge over the replay baselines come from its **environment access**
(fresh re-collection of past environments) or from its **mechanism** (the
one-sided value-constrained floor)? We strip CLEAR/RECALL of their replay buffer
and give them VCC's live past-env access (`clear_env_access` mode: each
consolidation iteration rolls the current policy in one random past env and
clones/distills toward that task's frozen end-of-task policy at the fresh states),
then compare.

Everything to regenerate figures is here: per-run `eval_matrix.json` + `run.json`
(normalization ref) + `resource_usage.json` under `runs/`, plus `summary.json`
with every number (PERF/BWT/forgetting + the full 10x10 normalized retention
matrix, diagonal, and final row for each run). 0 = random, 1 = solved.

---

## Comparison 1 — same reduced budget (CONFOUNDED, shown for the record)

| run | PERF | BWT |
|---|---|---|
| VCC | 0.610 | +0.128 |
| CLEAR + env access | 0.629 | −0.018 |
| RECALL + env access | 0.576 | −0.000 |

This *looked* like env access closes the gap (CLEAR edges VCC). It is an
artifact: VCC undertrained its T8/T9 **local specialists** (diag = 0.0), so its
floor was vacuous on the last two tasks. Do not use this as the headline.

## Comparison 2 — deployed-policy budget matched (THE CLEAN RESULT)

VCC's *deployed* policy is the global one, which sees `global_iters`; its local
specialist is VCC's own concession. So the fair budget for the replay methods is
what VCC's global sees. **VCC2** = VCC with T8/T9 specialists trained properly
(resume after T7: local 600→2000, stall-guard 500→1000, global 300→800).
**CLEAR/RECALL v2** = T8/T9 redone at `num_iters = 800` = VCC's deployed-global
budget (conceding VCC's local specialist).

| run | PERF | BWT | Forgetting | T8 diag | T9 diag |
|---|---|---|---|---|---|
| **VCC2** | **0.779** | +0.094 | 0.112 | 0.98 | **0.99** |
| CLEAR + env access v2 | 0.616 | −0.035 | 0.036 | 0.95 | **0.00** |
| RECALL + env access v2 | 0.580 | +0.004 | 0.000 | 0.99 | **0.00** |

**Headline:** with env access AND matched deployed-policy budget, the replay
methods still trail VCC by ~0.16–0.20, and **both fail the last task** (T9 = 0.0)
while VCC learns it (0.99). The replay anti-forgetting signal is a *symmetric*
clone/distill penalty that fights learning a new task while cloning the nine past
ones; VCC's *one-sided* floor only blocks dropping below the specialist, leaving
the policy free to learn T9. So env access alone does not explain VCC's edge —
the mechanism does.

---

## Figures to make
- **Three-way PERF (and BWT) bars** for Comparison 2: VCC2 0.779 / CLEAR-v2 0.616
  / RECALL-v2 0.580. Optionally a second panel for Comparison 1 with a
  "confounded" label.
- **Three retention-matrix heatmaps** (VCC2, CLEAR-v2, RECALL-v2) from
  `summary.comparison_2_deployed_budget_matched.*.norm_matrix` (10x10, nulls above
  the diagonal). The **T9 column** is the story: VCC2 reaches 0.99, both replay
  methods sit at 0.00.
- Optional: the `diagonal` arrays side by side (task-when-learned), highlighting
  T8 learned by all, T9 only by VCC.

## Caveats (put on every figure)
Single seed, 10 tasks, reduced budget, one task order. Comparison 2 concedes
VCC's local specialist (matches only the deployed-global budget, the fair quantity).
A directional ablation probe, not a general CL claim. Code: `clear_env_access`
mode in `crl/ppo/clear.py` (code-verifier PASS). Configs:
`configs/{clear,recall}_env_len10.yaml` (reduced), `configs/{clear,recall}_env_v2.yaml`
+ `configs/duel_abl_len10_vcc2.yaml` (budget-matched).
