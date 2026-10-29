"""Typed configuration loaded from YAML.

Every hyperparameter of a run lives in one YAML file under ``configs/``.
Unknown keys raise immediately so silent typos cannot corrupt a sweep.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentConfig:
    """Run identity and bookkeeping."""

    name: str = "run"
    seed: int = 42
    results_dir: str = "results"
    device: str = "cpu"
    log_every: int = 10


@dataclass
class EnvConfig:
    """Task-family selection.

    ``family`` picks an entry from :data:`crl.envs.FAMILY_REGISTRY`;
    ``params`` are family-wide settings; ``tasks`` is an ordered list of
    per-task parameter dicts (one dict per task, arrival order = list order).
    """

    family: str = "gridworld"
    params: dict[str, Any] = field(default_factory=dict)
    tasks: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class PolicyConfig:
    """Policy architecture."""

    kind: str = "tabular"  # tabular | mlp
    hidden_sizes: list[int] = field(default_factory=lambda: [64, 64])
    # Append a one-hot task id to observations (decide early; README risk 2).
    task_conditioned: bool = False
    # CKA-RL knowledge-pool size (kind == "cka_rl"): the pool merges down to this
    # many tau vectors by cosine-similarity averaging (their default is 5).
    pool_size: int = 5


@dataclass
class EstimatorConfig:
    """Value / gradient estimation backend.

    ``kind`` picks from :data:`crl.estimators.ESTIMATOR_REGISTRY`:
      exact        -- tabular dynamic-programming evaluation, zero variance
      monte_carlo  -- on-policy episode rollouts (REINFORCE-style)
    """

    kind: str = "exact"
    episodes_per_eval: int = 16  # episodes for a scalar value estimate
    episodes_per_grad: int = 16  # episodes for one gradient estimate
    episodes_per_ref: int = 64  # frozen-reference values (once per phase)
    baseline: str = "batch_mean"  # batch_mean | none
    # Keep the gamma^t weight on the REINFORCE terms so the estimator matches
    # the discounted policy gradient exactly (tested against `exact`).
    time_discount_weighting: bool = True


@dataclass
class DualConfig:
    """Dual-variable controller (lambda and mu use identical settings)."""

    kind: str = "projected_ascent"  # projected_ascent | pid
    lr: float = 0.05
    init: float = 0.0
    max_value: float = 100.0
    # Keep the dual value across phases/tasks instead of resetting to init.
    warm_start: bool = True
    # Decay-when-stuck (projected_ascent only): if mu sits at >= stuck_frac*max_value
    # for stuck_patience consecutive refreshes WITHOUT the constraint improving, the
    # task is treated as unlearnable and mu is multiplied by stuck_decay so it stops
    # permanently poisoning the objective. Off by default (no change to existing runs).
    decay_when_stuck: bool = False
    stuck_patience: int = 10
    stuck_decay: float = 0.5
    stuck_frac: float = 0.95
    # PID gains (used only when kind == "pid"; Stooke et al. 2020).
    kp: float = 0.05
    ki: float = 0.01
    kd: float = 0.0


@dataclass
class TrainerConfig:
    """Alternation schedule and primal-dual updates (eqs 22-24 and 30-32).

    The past-task constraint is a per-task, one-sided, *squared* hinge
    (eqs 7, 11): a penalty applies only where the trained policy falls below
    its frozen reference on a task (forgetting), and is zero when it is at or
    above the reference. The local phase therefore carries one multiplier
    lambda_i per past task; the global phase carries a single mu on the
    current task.
    """

    cycles_per_task: int = 2  # local<->global alternations per task
    local_steps: int = 50  # primal steps per local phase
    global_steps: int = 50  # primal steps per global phase
    task1_steps: int = 200  # plain ascent on task 1 (no past tasks yet)
    lr_local: float = 0.05  # alpha in eq 22
    lr_global: float = 0.05  # beta in eq 30
    optimizer: str = "sgd"  # sgd | adam  (sgd matches the derivation)
    # Constraint tolerance epsilon. IMPORTANT: units are squared value
    # (the constraint is a squared shortfall), so a tolerated value gap of g
    # corresponds to eps = g**2. A scalar is broadcast to every past task
    # (eps_i = eps in eq 8); a list sets eps_i per past task (length k-1 at
    # the final task, indexed by task arrival order).
    eps: float = 0.0025
    omega: list[float] | None = None  # per-task weights; None = uniform 1/k
    # all: sum over every past task per step (matches the derivation exactly).
    # sample: one past task per step, contribution rescaled by the past-task
    # count -- an unbiased O(1)-cost variant for large task sequences.
    past_task_sampling: str = "all"
    entropy_coef: float = 0.0
    eval_episodes: int = 32  # end-of-task evaluation matrix
    eval_all_tasks: bool = True  # include future tasks in the matrix
    # Probe the global policy on every task this often (in cumulative primal
    # steps) to build learning-curve data; 0 disables probing.
    eval_probe_every: int = 25
    # Report the *undiscounted* return (task performance / game score) in the
    # eval matrix and probes, instead of the discounted value. The constraint
    # and objective still use discounted value; only reporting changes. Set for
    # environments where the paper metric is the score (e.g. MinAtar).
    report_return: bool = False
    # Optimizer backend / trainer variant. "alternation" = the REINFORCE
    # primal-dual trainer (crl.trainer.AlternationTrainer). "ppo" = the PPO
    # backend (crl.ppo_continual.PPOAlternationTrainer): PPO replaces REINFORCE
    # as the optimizer, the CL framework is unchanged, and PPO settings come from
    # the ``ppo`` config section.
    kind: str = "alternation"
    # Variant: drop the past-task constraint from the LOCAL phase, so the local
    # policy is a pure-plasticity learner of the current task (maximizes V_k
    # only, no lambda shortfall terms). The GLOBAL phase is unchanged -- it is
    # still constrained w.r.t. the frozen local via mu. This tests whether
    # letting the local fully master the new task (then consolidating into the
    # global) fixes current-task underlearning without losing past retention.
    local_unconstrained: bool = False


@dataclass
class PPOConfig:
    """PPO optimizer settings for the ``trainer.kind == "ppo"`` backend.

    PPO replaces REINFORCE as the policy-gradient *optimizer* only; the
    continual-learning framework (local/global alternation, squared-shortfall
    constraint, lambda/mu duals, replay-free fresh rollouts) is unchanged and is
    configured through :class:`TrainerConfig` / :class:`DualConfig`. The critic,
    GAE, value loss and entropy bonus here are standard PPO -- only the ACTOR
    receives the constraint (in the global phase).
    """

    # Continual method: "constrained" = the full local/global min-max
    # consolidation (Part A -- experts NOT stored; objective doc eqs 1-48);
    # "finetune" = naive sequential standard PPO on one shared net (the
    # catastrophic-forgetting baseline; no local phase, no constraint);
    # "clear" = CLEAR (Rolnick 2019 / CORA baseline): replay + behavioral-cloning
    # + value-cloning on stored past-task data (recent replay-based SOTA baseline);
    # "joint" = EXPERIMENT 1 (feasibility upper bound): first train a fresh
    # single-task model per game (the ceiling), then train ONE shared-trunk model
    # on ALL games with mixed batches (no continual constraint). If the joint
    # model reaches each ceiling, a feasible theta EXISTS (rules out infeasibility
    # / capacity); if not, capacity/feasibility is the binding constraint.
    method: str = "constrained"
    # Past-task gradient collection in the global phase. "all" = roll out every
    # past task each iteration (exact sum, O(k) cost). "sample" = roll out ONE
    # uniformly-random past task per iteration, rescaled by the past-task count --
    # an unbiased minibatch estimate of the same sum at O(1) cost (removes the
    # O(k) blow-up on late tasks). Noise averages out over the iterations.
    # "all"    -- every past task each iteration, O(k)
    # "sample" -- past_task_sample_n drawn UNIFORMLY at random, rescaled by
    #             (k-1)/n so the estimate of the full past-task sum is unbiased.
    #             This is what the paper does.
    # "worst"  -- the n past tasks with the LOWEST current score. Biased for the
    #             sum by construction, and deliberately so: it spends the budget
    #             where the policy is failing rather than spreading it evenly.
    #             Not what the paper does; the weights omega_i stay 1/k, so the
    #             rescaling that makes "sample" unbiased does not apply.
    past_task_sampling: str = "all"
    constraint_greedy: bool = False  # estimate the constraint value V with greedy rollouts
    # Evaluate greedy value/score by ENUMERATING every no-op start (1..noop_max),
    # exactly one deterministic rollout each -> exact expected greedy return in
    # noop_max rollouts (no sampling). Overrides eval_episodes/constraint_episodes
    # for greedy eval. Correct only for deterministic envs (repeat_action_prob=0).
    eval_noop_enumerate: bool = False
    # Per-episode step cap for EVAL rollouts only (agent-steps; 0 = uncapped).
    # Safe (eval has no GAE); with gamma<1 the discounted value is unchanged, and
    # it bounds the cost of enumerating long-episode experts. Raw score becomes
    # "score over the first N steps" (applied equally to global + expert).
    eval_max_ep_steps: int = 0
    joint_iters: int = 2000        # mixed-batch PPO iters for the joint model
    joint_single_iters: int = 600  # per-task standard-PPO iters for each ceiling
    # EXPERIMENT 2 (value-constraint sufficiency): add a current-task behavioral
    # cloning term coef * KL(pi_local || pi_global) on fresh current-task states
    # to the global actor loss. 0 = off (pure value constraint). Tests whether a
    # scalar value constraint is too weak (needs behavior matching, not just V).
    global_bc_coef: float = 0.0
    # --- CLEAR baseline hyperparameters (CORA Table 3, Atari values) ---
    clear_policy_clone_cost: float = 0.01  # weight on KL(behavior||current) on replay
    clear_value_clone_cost: float = 0.005  # weight on value-cloning MSE on replay
    clear_snapshot_batches: int = 2  # rollout batches stored per task (the replay set)
    clear_replay_task_per_step: int = 1  # past tasks sampled for cloning per update
    # Environment-access variant (ablation): replace the stored replay buffer with
    # VCC-style FRESH re-collection. Each consolidation iteration rolls the CURRENT
    # policy in one random past environment (count = past_task_sample_n) and clones
    # toward that task's FROZEN end-of-task policy (global_after_task{i}.pt) queried
    # at the fresh states -- no stored transitions. Gives CLEAR/RECALL the same live
    # past-environment access VCC has, to isolate whether the access or the mechanism
    # is what matters. Naturally resumable (the frozen refs are on-disk checkpoints).
    clear_env_access: bool = False
    # --- RECALL (Replay-enhanced Continual RL, TMLR 2023) port of its MECHANISM
    # onto the CLEAR shared-head PPO backbone (the paper is SAC + multi-head). Two
    # additions over CLEAR on the replay stream: (1) policy DISTILLATION
    # KL(pi_phi || pi_old) [eq 8] at weight recall_distill_cost (paper lambda=10),
    # replacing CLEAR's behavioral-cloning KL(pi_old||pi_phi); (2) per-task PopArt
    # [eqs 3,6] running normalisation of the value-regression targets so cross-task
    # reward-scale bias cannot make high-return tasks dominate. The multi-head
    # output-preservation ("POP") is per-task-head specific and omitted in the
    # shared-head port. On this uniform [0,1]-return benchmark PopArt is near-identity.
    recall_distill_cost: float = 10.0  # lambda on KL(pi_phi||pi_old), RECALL eq 8/9
    recall_popart: bool = True         # per-task running target normalisation (eqs 3,6)
    recall_popart_beta: float = 3e-4   # PopArt running-stat rate beta (eq 6)
    # Past-reference budget match (CLEAR/RECALL). By default the replay stream draws
    # `mb` (the novel minibatch size) samples per minibatch-step, so over a task phase
    # it uses ppo_epochs*num_minibatches*mb*num_iters replay gradient samples -- for the
    # BigGrid v2 setup that is 802.8M across 49 phases, vs DUEL's 152.3M of fresh unique
    # past-task frames. Set clear_past_frames_per_phase > 0 to instead size the per-step
    # replay draw so the TOTAL replay gradient samples per task phase equal this target
    # (e.g. 3,108,864 = DUEL's 3 past x 32 x 128 x 253 per phase), matching DUEL's
    # past-reference update budget. The replay BUFFER (unique stored transitions) is
    # unchanged; only the reuse count drops. 0 = unchanged (old behaviour).
    clear_past_frames_per_phase: int = 0
    # Use AsyncVectorEnv (subprocess-parallel ALE stepping) for the TRAINING
    # collectors -- a large speedup for rollout collection on multi-CPU nodes
    # (env-stepping is ~60% of an iteration). Eval stays synchronous.
    async_envs: bool = False
    n_envs: int = 8  # parallel vectorized envs feeding the collector
    n_steps: int = 128  # rollout length per env per PPO iteration
    ppo_epochs: int = 4  # optimization epochs over each collected batch
    # KL early-stop for the LOCAL / task1 phase only: if a PPO epoch's mean approx_kl
    # exceeds this, stop optimizing that batch (prevents the runaway actor blow-up
    # seen on Boxing where the actor diverged to a uniform/NaN policy). The GLOBAL
    # consolidation phase always runs FULL epochs (passes kl_stop=False) so the
    # min-max retention update is never throttled. 0 = off.
    target_kl: float = 0.0
    num_minibatches: int = 4  # minibatches per epoch (batch = n_envs*n_steps)
    clip_ratio: float = 0.1  # PPO clip epsilon (0.1 is standard for Atari)
    gae_lambda: float = 0.95  # GAE(lambda)
    vf_coef: float = 0.5  # value-loss weight (standard PPO critic)
    ent_coef: float = 0.01  # entropy bonus (standard PPO)
    max_grad_norm: float = 0.5  # global grad-norm clip
    lr: float = 2.5e-4  # Adam learning rate (actor+critic share the trunk)
    normalize_advantage: bool = True  # per-minibatch advantage normalization
    # Phase budgets are now MAX CAPS (each model early-stops at its per-game
    # threshold; see below). Counted in PPO iterations (each = collect+ppo_epochs).
    # For a fair per-MODEL comparison, local, global and finetune all use the
    # SAME cap and the SAME per-game thresholds.
    task1_iters: int = 2000  # cap for plain PPO on task 1
    local_iters: int = 2000  # cap for the local phase (standard PPO on current)
    # Per-game local-phase cap, keyed by game name (e.g. {"Boxing": 4000}); games
    # not listed use ``local_iters``. Lets slow/hard games get more budget while
    # easy games keep a small cap -- sized from resource_usage.json (#1).
    local_iters_per_task: dict[str, int] = field(default_factory=dict)
    # Keep the BEST local model (by greedy score) from a phase instead of the final
    # one, so a late training collapse (e.g. Boxing diverging to a uniform policy)
    # doesn't discard the good peak. Applies to the local + task1 phases (single-game
    # objective); the kept snapshot becomes pi_L / the frozen reference.
    select_best_local: bool = False
    # Local specialist init + stall guard (constrained method only). The specialist is
    # only used to read the reference value refL, so a FRESH random init is equivalent
    # to warm-starting from the global -- and avoids the loss-of-plasticity trap where a
    # collapsed global init cannot learn a hard new task. local_init_fresh=True builds a
    # fresh policy each local phase. The stall guard aborts a local attempt early (an
    # explicit greedy eval at local_stall_at; if best greedy < local_stall_thr it is
    # deemed not-learning) and RETRIES with a new fresh init, escalating exploration
    # (entropy x3 then x5) and doubling the iters, up to local_max_attempts; the best
    # specialist across attempts is kept. The MAIN run is NEVER cancelled -- if every
    # attempt stalls the best (even if poor) is used and consolidation proceeds.
    # Defaults reproduce the old behaviour (warm init, no retry, guard off).
    local_init_fresh: bool = False
    local_max_attempts: int = 1
    local_stall_at: int = 0       # iter to run the stall greedy-eval (0 = guard off)
    local_stall_thr: float = 0.1  # best greedy below this at local_stall_at => stalled
    # V5 global early-stop: the consolidation stops ONLY when EVERY seen task (past +
    # current) reaches >= global_retention_frac of its LOCAL reference greedy score,
    # for `patience` consecutive checks; otherwise it runs to global_iters (no early
    # stop). Default off = the current-task-only criterion. It evaluates ALL seen
    # tasks each check, so pair it with a sparse stop_eval_every and small
    # stop_eval_episodes.
    global_stop_all_tasks: bool = False
    global_retention_frac: float = 0.7
    global_iters: int = 2000  # cap for the constrained global consolidation
    # Early stopping: stop a phase once the current game's GREEDY score is
    # >= its threshold for `patience` consecutive checks (but at least min_iters),
    # else run to the cap. Thresholds are per task (env.tasks[i].threshold).
    min_iters: int = 200  # floor per phase before early-stop can trigger
    patience: int = 3  # consecutive threshold-meeting checks required to stop
    stop_eval_every: int = 50  # iters between early-stop score checks
    stop_eval_episodes: int = 15  # greedy episodes per early-stop check
    # Full-episode Monte-Carlo estimates of V_k for the shortfall / references
    # (paper-faithful: the constraint value is the on-policy STOCHASTIC return).
    constraint_episodes: int = 16
    # Re-estimate V_k^G and update mu every N global iterations (slower dual
    # timescale; between updates the mu*2*shortfall coefficient is held fixed).
    constraint_every: int = 5
    # Constraint-value estimator: "mc" = full-episode Monte-Carlo (paper-faithful,
    # current default). "bootstrap" = windowed H-step return with the frozen LOCAL
    # critic at BOTH endpoints: V_k^L = V_L(S_0), V_k^G = sum_{t<H} gamma^t r_t +
    # gamma^H V_L(S_H) (global-policy rollout for the reward). `window_H` = window.
    # BOTH estimates are always logged each constraint check (boot-vs-MC bias).
    value_mode: str = "mc"
    window_H: int = 20
    # "residual" = BELLMAN RESIDUAL consolidation. Every task's advantage is
    # A_t = G_t - V_L_i(s_t): the FULL-EPISODE return minus that task's own frozen
    # specialist critic used purely as a BASELINE. A state-dependent baseline is
    # unbiased whatever its error, so the specialist cannot skew the direction --
    # unlike a bootstrap, which can. The constraint scalar falls out of the SAME
    # rollout as -mean(A_0), which by the telescoping identity equals
    # V_k^{pi_L} - V_k^{pi_phi} exactly. One rollout, one critic, no window, and
    # the global critic plays no part at all.
    # With gae_lambda=1 and whole episodes the specialist is a pure BASELINE and
    # cannot bias the gradient. With a SHORT window it becomes a bootstrap, which
    # can -- that is the regime `global_n_steps` < episode cap deliberately tests.
    #
    # `residual_scalar` decouples the constraint measurement from the gradient:
    #   "rollout" -- free, mean(G_0) from the same batch. Only honest when the
    #                episode finishes inside the window, else it reads
    #                (1-gamma^H) of the true gap, which is 26% at H=30.
    #   "mc"      -- a separate Monte-Carlo evaluation, as DUEL does. Costs frames
    #                but stays unattenuated at any window length.    # How many past tasks contribute a gradient each iteration. Their coefficients
    # are rescaled by (k-1)/n so the estimate of the full past-task sum stays
    # unbiased at any n.
    # Which critic supplies the BASELINE in residual mode.
    #   "expert" -- each task's frozen specialist. Costs a stored critic per task.
    #   "global" -- the global policy's own critic, trained normally. Pair with
    #               mlp_sharedactor_multicritic so it is not one network holding
    #               fifty value functions.
    # At gae_lambda=1 over whole episodes this choice cannot bias the gradient
    # either way -- a state-dependent baseline is unbiased however wrong it is --
    # so it is purely a variance question, and an empirical one.
    residual_baseline: str = "expert"
    residual_scalar: str = "rollout"
    past_task_sample_n: int = 1
    # Starvation fixes (default off, no change to existing runs):
    # past_full_rescale -- on the "worst" past path, scale the sampled past
    # coefficients by (k-1)/n so the n drawn tasks STAND IN for all k-1 past tasks
    # (the "sample" path already does this; "worst" deliberately did not).
    # past_fixed_fraction f in (0,1) -- guarantee the past collectively keeps at
    # least a fraction f of each global update by CAPPING the current-task
    # coefficient c_k = 2*mu*shortfall at past_total*(1-f)/f, so the current task
    # can never monopolise the step and starve retention (0 = off).
    past_full_rescale: bool = False
    past_fixed_fraction: float = 0.0
    # The global (consolidation) phase overrides these two. The LOCAL phase keeps
    # n_steps/gae_lambda, because ordinary truncated PPO with its own critic is
    # what produces a well-trained specialist and none of that needs changing.
    # Residual consolidation needs whole episodes (so the telescoping identity
    # holds and the specialist stays a pure baseline), hence >= the episode cap,
    # and lambda = 1 so the critic is never used as a bootstrap.
    global_n_steps: int = 0        # 0 -> fall back to n_steps
    global_gae_lambda: float = 0.0  # 0 -> fall back to gae_lambda
    # Envs for the GLOBAL phase only. Protocol rule 5: the global model's frames
    # must be matched across compared methods, while local specialist training is
    # unconstrained. Separating the two counts lets the specialists be trained as
    # well as we like without inflating the budget the comparison turns on.
    global_n_envs: int = 0         # 0 -> fall back to n_envs
    # Reported evaluation (eval matrix, probes): greedy (argmax) actions, many
    # episodes, fixed seed -> low variance, reproducible, respectable scores.
    eval_episodes: int = 100  # TOTAL episodes for the eval matrix / probes
    eval_greedy: bool = True  # PROJECT RULE: reported eval is ALWAYS greedy, never stochastic
    # Of the `eval_episodes` total, this many are GREEDY (argmax); the remaining
    # `eval_episodes - eval_greedy_episodes` are STOCHASTIC (sampled actions). The
    # reported score is the pooled mean over all episodes -- a blend of best-case
    # (greedy) and on-policy (stochastic) behaviour, not best-of. Clamped to
    # eval_episodes, so the large default => all-greedy (backward compatible).
    eval_greedy_episodes: int = 100_000
    eval_seed: int = 100_000  # fixed base seed for evaluation rollouts
    eval_every: int = 0  # probe the global policy every N cumulative iters (0=off)
    # End-of-task forgetting-matrix cadence. The scalar metrics (PERF, BWT,
    # forgetting, retention) need only each task's diagonal (its score right after
    # its own phase) + the final all-seen row, i.e. O(n) evals. Evaluating ALL seen
    # tasks after EVERY task is O(n^2) and only buys the full forgetting-trajectory
    # heatmap. matrix_eval_every>0 evaluates the full seen-row only every N tasks
    # (plus always the final task); other tasks eval only the current task and carry
    # prior columns forward. 0 = full row every task (fine for short Atari sequences).
    matrix_eval_every: int = 0
    # --- Global-phase DIAGNOSTICS (heavy; off by default) --------------------
    # When on, the global (consolidation) phase logs, every `diag_every` iters, a
    # rich "global_diag" row: V_k^L / V_k^G and their gap (oscillation), mu, the
    # constraint F_G vs eps (active/inactive), the current-task greedy score
    # trajectory, EACH past task's value + greedy score (is one dominating?), and
    # the actor-gradient decomposition ||g_new||/||g_old|| + cos(g_old, g_new)
    # (which term drives the update). g_new = current-task constraint gradient
    # (coeff_k * grad), g_old = past-task objective gradient (sum_i omega_i grad).
    diagnostics: bool = False
    diag_every: int = 25  # log a full diagnostics row every N global iters
    diag_score_episodes: int = 6  # greedy episodes per task for the perf trajectory
    # --- EXPERIMENT: head-only consolidation probe ---------------------------
    # When true, each global phase starts from the frozen LOCAL model (so the
    # shared trunk already carries the current-task features), then FREEZES the
    # trunk and consolidates ONLY the per-task heads under the same constrained
    # objective. Decisive test: if the current task is retained ~perfectly, the
    # damage in the normal method comes from shared-representation (trunk)
    # updates; if it still degrades, the constrained objective itself compromises.
    global_probe_head_only: bool = False


@dataclass
class Config:
    """Top-level run configuration."""

    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    estimator: EstimatorConfig = field(default_factory=EstimatorConfig)
    duals: DualConfig = field(default_factory=DualConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict view (for logging alongside results)."""
        return dataclasses.asdict(self)


def _build(cls: type, data: dict[str, Any], path: str) -> Any:
    """Instantiate a dataclass from a dict, rejecting unknown keys."""
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise KeyError(f"Unknown config keys at '{path}': {sorted(unknown)}")
    return cls(**data)


def config_from_dict(raw: dict[str, Any]) -> Config:
    """Build a :class:`Config` from a nested dict with strict key checking."""
    sections = {
        "experiment": ExperimentConfig,
        "env": EnvConfig,
        "policy": PolicyConfig,
        "estimator": EstimatorConfig,
        "duals": DualConfig,
        "trainer": TrainerConfig,
        "ppo": PPOConfig,
    }
    unknown = set(raw) - set(sections)
    if unknown:
        raise KeyError(f"Unknown top-level config sections: {sorted(unknown)}")
    kwargs = {
        name: _build(cls, raw.get(name, {}) or {}, name)
        for name, cls in sections.items()
    }
    return Config(**kwargs)


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML config file."""
    with open(path) as handle:
        raw = yaml.safe_load(handle) or {}
    return config_from_dict(raw)
