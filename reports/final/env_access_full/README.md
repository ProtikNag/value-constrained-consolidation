# Env-access at the full budget

Four figures. The ten-task probe in `../env_access` asked whether VCC's lead is
bought by re-entering earlier environments and answered no. It could not answer
the other half, because every replay arm in it already had access and none held
a buffer, so nothing in that record measured what access was worth.

This run supplies the control. Over the full fifty-task sequence at the full
204.8M-frame budget, each replay method runs twice, once with its buffer and
once re-collecting in the earlier environments with no buffer, and nothing else
differs.

Regenerate with `python reports/final/env_access_full/make_figures.py`. Source
data is `reports/gridworld_env_access/full_budget/`. Every metric is recomputed
from the stored matrices and asserted against `summary.json` at build time.

**Caveat on every figure.** Fifty tasks, full budget, one task order. The four
replay arms are seed0; VCC has three seeds.

---

## What the numbers say

| | PERF | BWT | Forgetting | solved ≥0.9 | failed ≤0.1 |
|---|---:|---:|---:|---:|---:|
| VCC, seed0 | **0.753** | +0.117 | 0.076 | 76% | 20% |
| CLEAR, buffer | 0.718 | −0.006 | 0.016 | 68% | 24% |
| CLEAR, env access | 0.709 | −0.013 | 0.015 | 68% | 26% |
| RECALL, buffer | 0.619 | −0.001 | 0.004 | 50% | 32% |
| RECALL, env access | 0.540 | −0.000 | 0.000 | 52% | 42% |

VCC's three-seed mean is 0.779 (sd 0.032), BWT +0.188. Every value in the table
and in the figures is seed0, because the replay arms have no other seed and
mixing a three-seed mean into the same panel would compare two different
quantities.

**The control.** CLEAR goes 0.718 to 0.709, a change of −0.009, which is inside
single-seed noise. RECALL goes 0.619 to 0.540, −0.079. Neither improves.
Swapping a replay buffer for live access to the earlier environments does not
help a replay method, so access is not what separates them from VCC.

**Where the separation is.** Backward transfer. VCC is the only arm above zero,
at +0.117 against −0.013 to −0.000. The replay arms forget almost nothing, down
to 0.000 for RECALL with access, and that is not a strength: they hold what they
acquire and recover nothing. VCC forgets more by that measure and still ends
ahead, because a task it leaves behind can come back.

**One caution.** CLEAR's access arm runs above its buffer arm for most of the
sequence and only finishes below it. Read CLEAR as unchanged by access rather
than as harmed by it. RECALL's access arm sits below its buffer arm throughout.

## The figures

- **`env_full_buffer_vs_env`** — the headline. Left, where the five arms land,
  with VCC as the rule. Right, the change from each method's own buffer run to
  its access run, as a bar from zero. A single dumbbell was tried first and
  fails, because CLEAR's change is nine times smaller than RECALL's and the two
  ends collide at any scale that shows both.
- **`env_full_retention_curve`** — mean score over the tasks seen so far, all
  five arms across fifty tasks. Dotted holds a buffer, solid re-collects.
- **`env_full_acquire_vs_final`** — each task's score on arrival against its
  score at the end, for VCC and the two access arms. The clearest statement of
  the mechanism. VCC moves 11 tasks up and 4 down. CLEAR moves 0 up and 4 down.
  RECALL moves nothing either way.
- **`env_full_metrics`** — PERF, BWT, forgetting and solved share for all five
  arms. Each panel carries its own axis, because the scales differ by more than
  ten times.

## How this relates to the earlier env-access records

Three datasets sit under `reports/gridworld_env_access/`, at increasing scale.

| Record | Scale | Holds a buffer control? |
|---|---|---|
| `summary.json` | 10 tasks, reduced budget | no |
| `full_run/summary.json` | 25 tasks depth 25, 50 tasks depth 50, reduced | no |
| `full_budget/summary.json` | 50 tasks, full 204.8M budget | **yes** |

Appendix D.3 currently reports the ten-task record. This one is longer, at the
reported budget, and carries the control, so it is the stronger version of the
same argument.

---

## Conventions

The academic template from `report/acviz.py`. White ground, Tufte spine, faint
horizontal grid, blue for ours with the Setting B colours for CLEAR and RECALL,
mono numerals, PNG at 300 dpi and SVG from the same figure object. Set
`FIGS_FOR_PAPER=1` to drop the footnotes, which a LaTeX caption would otherwise
duplicate.
