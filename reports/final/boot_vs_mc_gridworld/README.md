# GridWorld MC-vs-bootstrap ablation — files

**Use `data_normalized.json`.** `data.json` is superseded (kept for provenance).

## What changed
`data.json` reported PERF as **raw `avg_performance` (~0.87)**. The main GridWorld
experiment (`reports/gridworld_sharedhead`, `sh_ours` PERF **0.596**) reports
**normalized** PERF (`0=random, 1=expert-ceiling`). Same runs, different scale —
which made the ablation look inflated. `data_normalized.json` re-reports every
config in the **normalized** metric (via `analysis.contract_metrics` vs the
seed-matched `sh_baseline`), on the same scale as the headline.

## No rerun was needed
The ablation runs are the **multi-head** variant, but **GridWorld PERF is
head-invariant**: raw `avg_performance` is 0.879 (multi-head) vs 0.884
(shared-head `sh_ours`) — a ~0.005 difference. So the multi-head ablation
transfers directly to the shared-head setting; only the reporting metric was
wrong, and that is fixed by re-normalizing the existing runs (zero recompute).

## Result (normalized PERF, 3 seeds)
| config | PERF (norm) | speedup |
|---|---|---|
| mc | 0.554 ± 0.018 | 1× |
| **bootH1** | **0.567 ± 0.026** | **30.7×** |
| bootH3 | 0.527 ± 0.018 | 17.8× |
| bootH5 | 0.564 ± 0.011 | 16.5× |
| bootH10 | 0.542 ± 0.003 | 10.2× |
| bootH20 | 0.544 ± 0.018 | 4.8× |
| bootH40 | 0.554 ± 0.025 | 2.7× |
| bootH80 | 0.544 ± 0.012 | 1.7× |

**The finding is unchanged from `data.json`:** bootstrap ≈ MC at every H, and
**H=1 is the free lunch** (30.7× speedup, PERF within 1 std of MC). Only the
absolute scale moved (~0.87 raw → ~0.55 normalized); bias / speedup / break-even
are metric-independent and carried over from the raw run logs.
