"""PPO continual-learning orchestrator (min-max local/global alternation).

This is the PPO analogue of :class:`crl.trainer.AlternationTrainer`. The
continual-learning framework is identical to the document -- only the optimizer
is PPO instead of REINFORCE:

* task 1        -- standard PPO trains the global model (no past tasks).
* task k >= 2   -- for each cycle: a *local* phase (standard PPO on the current
                   task from theta^0 = phi) produces the frozen reference V_k^L;
                   a *global* phase (PPO + the actor-only mu constraint) pushes
                   the global model's past-task value up while keeping
                   V_k^G >= V_k^L via the one-sided squared shortfall.

Uniform weights omega_i = 1/k, a single persistent mu controller (reset per
global phase, warm-started by config), replay-free fresh rollouts. The eval
matrix and probes report the raw game score (task performance), while the
constraint uses the discounted return (V), exactly as in the REINFORCE trainer.

``method: finetune`` runs naive sequential standard PPO on one shared net (no
local phase, no constraint) -- the catastrophic-forgetting baseline.
"""

from __future__ import annotations

import torch

from crl.config import Config
from crl.duals import make_dual
from crl.envs.base import TaskFamily
from crl.logging_utils import RunLogger
from crl.policies import make_policy
from crl.policies.base import Policy, clone_policy
from crl.ppo.evaluate import evaluate_value_and_score
from crl.ppo.trainer import GlobalTrainer, LocalTrainer
from crl.run_logging import ContractLogger


class PPOAlternationTrainer:
    """Runs the full task sequence with PPO; records the eval matrix + probes."""

    def __init__(
        self,
        config: Config,
        family: TaskFamily,
        global_policy: Policy,
        logger: RunLogger,
    ) -> None:
        self._config = config
        self.cfg = config.trainer
        self.ppo = config.ppo
        self.dual_cfg = config.duals
        self.family = family
        self.global_policy = global_policy
        self.logger = logger
        # Grow-methods (CompoNet) build their modules lazily in add_task, so the
        # policy can have NO parameters at construction -> fall back to the config
        # device. Eager policies are unchanged (first param's device).
        _first = next(global_policy.parameters(), None)
        if _first is not None:
            self.device = _first.device
        else:
            _dev = config.experiment.device
            self.device = torch.device(
                "cuda" if _dev in ("auto", "cuda") and torch.cuda.is_available() else "cpu")
        self.seed = config.experiment.seed
        self.method = config.ppo.method
        # Frozen per-task specialists, loaded lazily for value_mode="residual"
        # and kept, so a 50-task run does not reload 49 checkpoints every task.
        self._expert_cache: dict[int, Policy] = {}
        self._clear_ref_cache: dict[int, Policy] = {}  # env-access clone targets
        log_every = config.experiment.log_every

        self.local_trainer = LocalTrainer(self.ppo, self.device, logger, log_every)
        self.global_trainer = GlobalTrainer(self.ppo, self.device, logger, log_every)
        self.mu_ctrl = make_dual(self.dual_cfg)

        # CLEAR baseline: single replay store + trainer (created lazily on use).
        self._clear_trainer = None
        self._clear_replay = None
        self._log_every = log_every

        self.eval_matrix: list[list[float]] = []
        self.cumulative_step = 0
        self._resource: dict = {}      # per-game per-phase iters/wall/early-stop (#1)

        # Logging-contract emitter (LOGGING_CONTRACT.md), created in run().
        self.clog: ContractLogger | None = None
        frame_skip = int(config.env.params.get("frame_skip", 1))  # gridworld = 1
        self._frames_per_iter = self.ppo.n_envs * self.ppo.n_steps * frame_skip
        self._hb_task = 0            # heartbeat context, updated per probe
        self._hb_phase = "init"

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #

    def _eps(self) -> float:
        e = self.cfg.eps
        return float(e) if isinstance(e, (int, float)) else float(e[0])

    def _eval_report(self, policy: Policy, task) -> tuple[float, float]:
        """Reported (raw) game score for ``policy`` on ``task``, pooled over a
        mix of GREEDY and STOCHASTIC episodes (fixed seeds -> reproducible).

        Of ``eval_episodes`` total, ``eval_greedy_episodes`` use argmax actions
        and the rest sample; the reported score is the pooled mean over ALL
        episodes (a blend of best-case and on-policy behaviour, not best-of).
        Returns ``(pooled_mean_score, pooled_std)``."""
        if self.ppo.eval_noop_enumerate:  # exact greedy via no-op enumeration
            _, score, std, _ = evaluate_value_and_score(
                policy, task, 0, self.ppo.n_envs, self.device,
                seed=self.ppo.eval_seed, noop_enumerate=True,
                max_ep_steps=self.ppo.eval_max_ep_steps)
            return score, std
        total = self.ppo.eval_episodes
        n_greedy = min(self.ppo.eval_greedy_episodes, total) if self.ppo.eval_greedy else 0
        n_stoch = total - n_greedy

        groups = []  # (mean, std, n)
        if n_greedy > 0:
            _, m, s, n = evaluate_value_and_score(
                policy, task, n_greedy, self.ppo.n_envs, self.device,
                seed=self.ppo.eval_seed, greedy=True,
            )
            groups.append((m, s, n))
        if n_stoch > 0:
            _, m, s, n = evaluate_value_and_score(
                policy, task, n_stoch, self.ppo.n_envs, self.device,
                seed=self.ppo.eval_seed + 1, greedy=False,
            )
            groups.append((m, s, n))

        n_tot = sum(n for _, _, n in groups)
        if n_tot == 0:
            return 0.0, 0.0
        pooled_mean = sum(m * n for m, _, n in groups) / n_tot
        # Exact pooled population variance across the (possibly two) groups.
        pooled_var = sum(n * (s * s + (m - pooled_mean) ** 2)
                         for m, s, n in groups) / n_tot
        return pooled_mean, pooled_var ** 0.5

    def _eval_value(self, policy: Policy, task) -> float:
        """On-policy STOCHASTIC discounted value V^pi (the constraint reference)."""
        value, _, _, _ = evaluate_value_and_score(
            policy, task, self.ppo.constraint_episodes, self.ppo.n_envs,
            self.device, greedy=False,
        )
        return value

    def _evaluate_row(self, k: int) -> tuple[list[float], list[float]]:
        """Raw-score row (+ per-game std) of the forgetting matrix after task k."""
        last = len(self.family) if self.cfg.eval_all_tasks else k
        row, stds = [], []
        for i in range(last):
            score, std = self._eval_report(self.global_policy, self.family.tasks[i])
            row.append(score)
            stds.append(std)
        return row, stds

    def _init_contract(self) -> None:
        """Build the ContractLogger + normalization reference. ``random`` is the
        FRESH (untrained) policy's greedy score per task, measured once at run start;
        ``ceiling`` is per-task from config (default 1.0 = proximity max for gridworld
        / a normalized ceiling elsewhere). On resume, run.json already exists so its
        reference is preserved and we skip the (now-invalid) fresh-policy measurement."""
        tasks = [t.spec.name for t in self.family.tasks]
        if (self.logger.run_dir / "run.json").exists():
            random: list[float] = []            # preserved from the original run.json
            ceiling: list[float] = []
        else:
            # Random reference per task. Grow-methods (CompoNet) have no modules for
            # future tasks yet at run start, so their fresh-policy eval fails -> fall
            # back to a uniform-random-action rollout (the canonical random floor).
            random = []
            for t in self.family.tasks:
                try:
                    random.append(self._eval_report(self.global_policy, t)[0])
                except (IndexError, KeyError, StopIteration, RuntimeError, AttributeError):
                    random.append(self._uniform_ref(t))
            ceiling = [float(t.spec.params.get("ceiling", 1.0))
                       for t in self.family.tasks]
        self.clog = ContractLogger(
            self.logger.run_dir, method=self.method, seed=self.seed,
            env_family=self._config.env.family, tasks=tasks,
            task_order=str(self._config.env.params.get("task_order", "as_configured")),
            reference={"random": random, "ceiling": ceiling},
            config=self._config.to_dict(), frames_per_iter=self._frames_per_iter)

    def _uniform_ref(self, task) -> float:
        """Random floor = mean score of a uniform-random-action policy on ``task``.
        Used when the method's own policy cannot be evaluated pre-training (CompoNet's
        modules are added lazily)."""
        n_actions = self.family.num_actions
        dev = self.device

        class _Uniform:
            def dist(self, obs, task_id):
                return torch.distributions.Categorical(
                    logits=torch.zeros(obs.shape[0], n_actions, device=obs.device))

        _, score, _, _ = evaluate_value_and_score(
            _Uniform(), task, self.ppo.eval_episodes, self.ppo.n_envs, dev,
            seed=self.ppo.eval_seed, greedy=False)  # sample => true random walk
        return score

    def _emit_row(self, k: int, row: list[float], phase: str, episodes: int) -> None:
        """Emit one contract eval record per SEEN task from an end-of-task row
        (the end-of-phase / end-of-run all-seen cadence, on the global policy)."""
        if self.clog is None:
            return
        for j, sc in enumerate(row):
            self.clog.eval(task_idx=k - 1, phase=phase, it=-1, evaluated_on=j,
                           evaluated_on_task=self.family.tasks[j].spec.name,
                           raw=float(sc), episodes=episodes, greedy=True, seen=True)

    def _probe(self, phase_type: str, current_task: int) -> None:
        """Record the global policy's score on every task vs cumulative iters."""
        self.cumulative_step += 1
        # Keep the contract frame counter + heartbeat live for mid-flight checks.
        if self.clog is not None:
            self.clog.frames_total = self.cumulative_step * self._frames_per_iter
            self._hb_task, self._hb_phase = current_task, phase_type
            if self.cumulative_step % 50 == 0:
                self.clog.heartbeat(task_idx=current_task - 1,
                                    task=self.family.tasks[current_task - 1].spec.name,
                                    phase=phase_type, iter=self.cumulative_step,
                                    cumulative_step=self.cumulative_step)
        every = self.ppo.eval_every
        if not every or self.cumulative_step % every != 0:
            return
        # Live retention probe: greedy score on the SEEN tasks only (past + current;
        # future tasks aren't learned yet, so evaluating them wastes eval time).
        # current_task is 1-based, so range(current_task) = tasks 0..k-1 = all seen.
        seen = range(current_task)
        values = [self._eval_report(self.global_policy, self.family.tasks[i])[0]
                  for i in seen]
        self.logger.log(
            {
                "phase": "probe",
                "cumulative_step": self.cumulative_step,
                "current_task": current_task,
                "phase_type": phase_type,
                "values": values,             # greedy scores, seen tasks (past + current)
            }
        )

    # ------------------------------------------------------------------ #
    # main loop
    # ------------------------------------------------------------------ #

    def _phase_end(self, k: int, phase: str, summ: dict | None) -> None:
        """Emit a contract phase_end from a trainer summary (iters/wall -> frames)."""
        if self.clog is None or summ is None:
            return
        iters = int(summ.get("iters_run", 0))
        self.clog.phase_end(task_idx=k - 1, task=self.family.tasks[k - 1].spec.name,
                            phase=phase, iters=iters,
                            frames_phase=iters * self._frames_per_iter,
                            wall_s_phase=float(summ.get("wall_s", 0.0)))

    def _train_first_task(self) -> None:
        """Standard PPO on task 1 (the global model; no past tasks, no constraint)."""
        task0 = self.family.tasks[0]
        # Task 1 uses its PER-GAME budget too (not just task1_iters). In the reversed
        # Atari order SpaceInvaders is task 1: with task1_iters=1500 it under-trained
        # (local ~588 vs achievable ~906), capping the global. Consult
        # local_iters_per_task[game] first so SI gets its 3000 (early-stops at its
        # threshold otherwise). Falls back to task1_iters for games not listed.
        game0 = getattr(task0, "game", task0.spec.name)
        n_iters = self.ppo.local_iters_per_task.get(game0, self.ppo.task1_iters)
        if self.clog is not None:
            self.clog.phase_start(0, task0.spec.name, "task1")
        summ = self.local_trainer.train(
            self.global_policy,
            task0,
            num_iters=n_iters,
            seed=self.seed + 1000,
            current_task=1,
            phase_type="task1",
            probe=self._probe,
            clog=self.clog,
        )
        self._phase_end(1, "task1", summ)
        self._record_resource(1, "task1", summ)
        # Task 1 has no local phase -> the task1 model IS its specialist; record its
        # greedy score as the local reference (used by the V5 all-tasks retention stop).
        self._resource.setdefault(self.family.tasks[0].spec.name, {"task": 1})[
            "local_greedy"] = self._eval_report(self.global_policy, self.family.tasks[0])[0]

    def _finetune_task(self, k: int) -> None:
        """Naive baseline: keep fine-tuning the one shared net on task k."""
        if self.clog is not None:
            self.clog.phase_start(k - 1, self.family.tasks[k - 1].spec.name, "finetune")
        summ = self.local_trainer.train(
            self.global_policy,
            self.family.tasks[k - 1],
            num_iters=self.ppo.local_iters + self.ppo.global_iters,
            seed=self.seed + 1000 * k,
            current_task=k,
            phase_type="finetune",
            probe=self._probe,
            clog=self.clog,
        )
        self._phase_end(k, "finetune", summ)

    def _grow_task(self, k: int) -> None:
        """CompoNet / CKA-RL: expand the architecture for task k (policy.add_task(k)
        freezes/combines the frozen past), then standard PPO on the CURRENT task only
        (the fresh optimizer sees only the newly-trainable params). No local/global
        min-max split -- these are their own continual methods. store_eval_head(k)
        (CKA-RL) then freezes task k's effective head for faithful past-task eval."""
        task = self.family.tasks[k - 1]
        if hasattr(self.global_policy, "add_task"):
            self.global_policy.add_task(k)
            # add_task creates new modules/params (CompoNet module, CKA-RL alpha) that
            # land on CPU (its own device detection is empty at task 1); move the whole
            # policy to the run device so nothing is left on CPU on a GPU run.
            self.global_policy.to(self.device)
        if self.clog is not None:
            self.clog.phase_start(k - 1, task.spec.name, self.method)
        # Use the SAME per-game budget as ours' local phase (local_iters_per_task[game])
        # so the grow-method baselines (CompoNet/CKA-RL) train each task for the same
        # iters as ours -- else SI-as-task-1 got task1_iters (1500) vs ours' 3000,
        # undertraining the baselines and unfairly flattering ours. Falls back to the
        # generic budget for envs with no per-game entry (e.g. gridworld).
        game = getattr(task, "game", task.spec.name)
        n_iters = self.ppo.local_iters_per_task.get(
            game, self.ppo.task1_iters if k == 1
            else self.ppo.local_iters + self.ppo.global_iters)
        summ = self.local_trainer.train(
            self.global_policy, task, num_iters=n_iters,
            seed=self.seed + 1000 * k, current_task=k, phase_type=self.method,
            probe=self._probe, clog=self.clog,
        )
        self._phase_end(k, self.method, summ)
        self._record_resource(k, self.method, summ)
        if hasattr(self.global_policy, "store_eval_head"):
            self.global_policy.store_eval_head(k)

    def _baseline_task(self, k: int) -> None:
        """From-scratch FWT baseline: train task k on a FRESH policy (no transfer,
        no constraint), so its within-phase learning curve is the AUC_i^b reference
        for forward transfer. The trained-from-scratch model is discarded -- only the
        current task's own head/trunk matter, and the global policy is left holding
        task k so the end-of-task eval row still reports it. Each task is independent,
        which is the definition of the CKA-RL / Continual-World baseline."""
        from crl.policies import make_policy
        fresh = make_policy(self._config.policy, self.family).to(self.device)
        if self.clog is not None:
            self.clog.phase_start(k - 1, self.family.tasks[k - 1].spec.name, "task1")
        summ = self.local_trainer.train(
            fresh, self.family.tasks[k - 1],
            num_iters=self.ppo.local_iters + self.ppo.global_iters,
            seed=self.seed + 1000 * k, current_task=k,
            phase_type="task1",           # from-scratch => same cadence label as task 1
            probe=self._probe, clog=self.clog,
        )
        self._phase_end(k, "task1", summ)
        # Copy the freshly-trained task-k head + trunk into the reported global policy
        # so the end-of-task row measures the from-scratch model on task k.
        self.global_policy.load_state_dict(fresh.state_dict())

    def _clear_task(self, k: int) -> None:
        """CLEAR baseline: PPO on task k + replay/behavioral/value cloning on the
        past, then snapshot task k's behavior into the replay store."""
        task_k = self.family.tasks[k - 1]
        env = self.ppo.clear_env_access
        self._clear_trainer.train(
            self.global_policy, task_k, self._clear_replay,
            num_iters=self.ppo.local_iters + self.ppo.global_iters,
            seed=self.seed + 1000 * k, current_task=k, probe=self._probe,
            past_tasks=self.family.tasks[:k - 1] if env else None,
            frozen_ref=self._clear_ref_for_task if env else None,
        )
        self._clear_trainer.snapshot(self.global_policy, task_k, self._clear_replay)

    def _clear_ref_for_task(self, tid: int) -> Policy:
        """Frozen end-of-task global policy for past task ``tid`` (0-based) -- the
        clone/distill target in env-access mode. ``global_after_task{tid+1}.pt`` is
        written by :meth:`_save_progress` after each task, so a resumed run finds the
        references of the tasks it skipped already on disk."""
        cached = self._clear_ref_cache.get(tid)
        if cached is not None:
            return cached
        from crl.policies import make_policy
        path = self.logger.run_dir / f"global_after_task{tid + 1}.pt"
        if not path.exists():
            raise FileNotFoundError(
                f"clear_env_access needs the frozen reference {path} for past task "
                f"{tid}; when resuming, point --resume-from at the same run directory.")
        ref = make_policy(self._config.policy, self.family).to(self.device)
        ref.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=False))
        for p in ref.parameters():
            p.requires_grad_(False)
        ref.eval()
        self._clear_ref_cache[tid] = ref
        return ref

    def _expert_for_task(self, i: int) -> Policy:
        """Frozen specialist for task ``i`` (1-based), loaded once and cached.

        Task 1 has no local phase, so the task-1 global model IS its specialist
        (``global_after_task1.pt``); tasks 2.. use ``local_after_task{i}.pt``.
        Both files are written by this trainer, so a resumed run finds the
        specialists of the tasks it skipped already on disk.
        """
        cached = self._expert_cache.get(i)
        if cached is not None:
            return cached
        from crl.policies import make_policy
        name = "global_after_task1.pt" if i == 1 else f"local_after_task{i}.pt"
        path = self.logger.run_dir / name
        if not path.exists():
            raise FileNotFoundError(
                f"value_mode='residual' needs the task-{i} specialist at "
                f"{path}, which is missing. When resuming, point --resume-from at "
                f"the same run directory so earlier specialists are available.")
        expert = make_policy(self._config.policy, self.family).to(self.device)
        expert.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=False))
        for param in expert.parameters():
            param.requires_grad_(False)
        expert.eval()
        self._expert_cache[i] = expert
        return expert

    def _constrained_task(self, k: int) -> None:
        task_k = self.family.tasks[k - 1]
        past_tasks = [self.family.tasks[i] for i in range(k - 1)]
        for cycle in range(self.cfg.cycles_per_task):
            # ---- local phase: theta^0 = phi, standard PPO on task k ---------
            game = getattr(task_k, "game", task_k.spec.name)
            n_local = self.ppo.local_iters_per_task.get(game, self.ppo.local_iters)
            if self.clog is not None:
                self.clog.phase_start(k - 1, task_k.spec.name, "local")
            # Local specialist training with the stall guard + escalating fresh retries.
            # The specialist is only read for refL, so a fresh init is equivalent to the
            # warm (global-cloned) init and avoids the loss-of-plasticity trap. Retries
            # NEVER cancel the run: the best specialist across attempts is kept and
            # consolidation proceeds regardless.
            ent_mult = [1.0, 3.0, 5.0]
            attempts = max(1, self.ppo.local_max_attempts)
            best_local, best_bg, loc_summ = None, float("-inf"), None
            for a in range(attempts):
                use_fresh = self.ppo.local_init_fresh or a > 0  # retries are always fresh
                local_policy = (make_policy(self._config.policy, self.family).to(self.device)
                                if use_fresh
                                else clone_policy(self.global_policy, trainable=True))
                last = (a == attempts - 1)
                summ = self.local_trainer.train(
                    local_policy, task_k,
                    num_iters=n_local * (2 if a > 0 else 1),
                    seed=self.seed + 1000 * k + 13 * cycle + 97 * a,
                    current_task=k, phase_type="local", probe=self._probe,
                    clog=self.clog,
                    ent_coef_override=(self.ppo.ent_coef * ent_mult[min(a, 2)] if a > 0 else None),
                    stall_at=(0 if last else self.ppo.local_stall_at),  # last attempt runs to the end
                    stall_thr=self.ppo.local_stall_thr,
                )
                bg = summ.get("best_greedy")
                bg = float(bg) if bg is not None else float("-inf")
                if bg > best_bg or best_local is None:
                    best_bg, best_local, loc_summ = bg, clone_policy(local_policy, trainable=True), summ
                if not summ.get("stalled"):
                    break  # learned (or ran to the end) -> keep this attempt
                print(f"[local-retry] task {k} attempt {a+1}/{attempts} stalled "
                      f"(best={bg:.3f}); re-initialising fresh with more exploration")
            local_policy = best_local
            if attempts > 1 and best_bg < self.ppo.local_stall_thr:
                print(f"[local-stall] task {k}: all {attempts} attempts stalled, best "
                      f"greedy={best_bg:.3f} -- keeping best, main run continues")
            self._phase_end(k, "local", loc_summ)
            self._record_resource(k, "local", loc_summ)
            frozen_local = clone_policy(local_policy, trainable=False)
            # Retain the local model + its greedy-100 score: for Part A (experts NOT
            # stored) the LOCAL model is the per-task reference/"specialist" (it IS the
            # constraint target V_k^L), so results are normalised against it, not an
            # external single-task expert. (Task 1 has no local phase; its specialist
            # is global_after_task1.)
            torch.save(frozen_local.state_dict(),
                       self.logger.run_dir / f"local_after_task{k}.pt")
            ref_current = self._eval_value(frozen_local, task_k)
            self._resource.setdefault(task_k.spec.name, {"task": k})["local_greedy"] = \
                self._eval_report(frozen_local, task_k)[0]

            # ---- global phase: PPO + actor-only mu constraint ---------------
            self.mu_ctrl.reset()
            omega = [1.0 / k] * (k - 1)  # uniform weights omega_i = 1/k
            if self.ppo.global_probe_head_only:
                # EXPERIMENT: consolidate on the local's (current-task) trunk,
                # frozen, moving only the per-task heads.
                self.global_policy.load_state_dict(local_policy.state_dict())
                for name, p in self.global_policy.named_parameters():
                    p.requires_grad_(not name.startswith("trunk."))
            refs = None
            if self.ppo.global_stop_all_tasks:
                # local reference greedy for every seen task (past + current), aligned
                # to past_tasks + [task_k]; the global stops only when ALL are >= frac.
                refs = [self._resource.get(t.spec.name, {}).get("local_greedy")
                        for t in past_tasks + [task_k]]
            # Per-task specialists for the past tasks, when each task's gradient
            # is to be bootstrapped by its own expert rather than by the global
            # critic. Loaded once and cached across tasks.
            past_experts = None
            if (self.ppo.value_mode == "residual"
                    and self.ppo.residual_baseline == "expert" and past_tasks):
                past_experts = [self._expert_for_task(i + 1) for i in range(k - 1)]
            if self.clog is not None:
                self.clog.phase_start(k - 1, task_k.spec.name, "global")
            glob_summ = self.global_trainer.train(
                self.global_policy, task_k, past_tasks,
                ref_current=ref_current, mu_ctrl=self.mu_ctrl, omega=omega,
                eps=self._eps(),
                num_iters=self.ppo.global_iters,
                seed=self.seed + 1000 * k + 13 * cycle,
                current_task=k, probe=self._probe,
                local_policy=frozen_local,  # for KL-gap logging + optional BC term
                retention_refs=refs, retention_frac=self.ppo.global_retention_frac,
                past_experts=past_experts,
                clog=self.clog,
            )
            self._phase_end(k, "global", glob_summ)
            self._record_resource(k, "global", glob_summ)
            if self.ppo.global_probe_head_only:  # restore full trainability
                for p in self.global_policy.parameters():
                    p.requires_grad_(True)
            self.logger.log(
                {"phase": "gaps", "task": k, "cycle": cycle,
                 "V_k_ref_local": ref_current}
            )

    def _train_joint(self) -> None:
        """EXPERIMENT 1 (feasibility upper bound). First train a FRESH single-task
        model per game (standard PPO) = the per-task ceiling; then train ONE
        shared-trunk model on ALL games with mixed batches (equal weight, no
        constraint). Compare joint per-task performance to the ceilings: if joint
        reaches them, a feasible shared theta EXISTS."""
        from crl.policies import make_policy
        from crl.ppo.collector import RolloutCollector

        cfg = self.ppo
        tasks = self.family.tasks
        n = len(tasks)

        # --- per-task ceilings (fresh model each; only its head + trunk move) ---
        ceilings = []
        for i, t in enumerate(tasks):
            m = make_policy(self._config.policy, self.family).to(self.device)
            self.local_trainer.train(
                m, t, num_iters=cfg.joint_single_iters, seed=self.seed + 500 + i,
                current_task=i + 1, phase_type="joint_single")
            score, _ = self._eval_report(m, t)
            ceilings.append(score)
            self.logger.log({"phase": "joint_ceiling", "task": i + 1,
                             "game": t.spec.name, "score": score})
            print(f"[joint ceiling] {t.spec.name}: {score:.1f}")

        # --- joint model: one shared trunk, all games mixed, no constraint ---
        joint = make_policy(self._config.policy, self.family).to(self.device)
        collectors = [RolloutCollector(t, cfg.n_envs, cfg.n_steps, self.device,
                                       self.seed + 700 + i) for i, t in enumerate(tasks)]
        opt = self.local_trainer._new_optimizer(joint)
        try:
            for it in range(cfg.joint_iters):
                streams = [c.collect(joint, cfg.gae_lambda) for c in collectors]
                self.local_trainer.optimize_batches(joint, opt, streams, [1.0] * n)
                if cfg.eval_every and (it + 1) % cfg.eval_every == 0:
                    scores = [self._eval_report(joint, t)[0] for t in tasks]
                    self.logger.log({"phase": "joint_probe", "step": it + 1,
                                     "scores": scores})
                    print(f"[joint] it={it+1} scores={['%.0f'%s for s in scores]}")
        finally:
            for c in collectors:
                c.close()

        joint_scores = [self._eval_report(joint, t)[0] for t in tasks]
        thresholds = [float(getattr(t, "threshold", 0.0)) for t in tasks]
        result = {"games": [t.spec.name for t in tasks], "ceilings": ceilings,
                  "joint": joint_scores, "thresholds": thresholds}
        self.eval_matrix = [joint_scores]
        self.logger.log({"phase": "joint_final", **result})
        self.logger.save_json("eval_matrix.json", self.eval_matrix)
        self.logger.save_json("joint_result.json", result)
        torch.save(joint.state_dict(), self.logger.run_dir / "final_policy.pt")
        print(f"[joint] ceilings={['%.0f'%c for c in ceilings]} "
              f"joint={['%.0f'%s for s in joint_scores]}")

    def _record_resource(self, task: int, phase: str, summ: dict | None) -> None:
        """Track per-game, per-phase budget: iters actually run, wall-time, whether
        it early-stopped (vs hit the cap), and the final greedy score. Saved to
        resource_usage.json so future runs can size per-game budgets instead of
        guessing (#1)."""
        game = self.family.tasks[task - 1].spec.name
        rec = self._resource.setdefault(game, {"task": task})
        rec[phase] = summ

    def resume(self, ckpt_path: str, after_task: int) -> None:
        """Continue from a saved global checkpoint instead of training from scratch.

        Loads the shared trunk + already-trained per-task heads with strict=False,
        so ADDING a task (a policy with more heads than the checkpoint) copies the
        trunk + old heads and leaves the new task's head freshly initialised (eq 6
        chaining: task k+1's frozen reference IS the loaded global). Reloads the
        partial eval matrix and makes run() skip tasks 1..after_task."""
        import json as _json
        import os as _os
        sd = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        # strict=False is what lets a grow-method add a task, but it also means a
        # checkpoint from a DIFFERENT architecture loads as silently-fresh weights.
        # Refuse that outright. (Where the two architectures share names but not
        # shapes -- mlp_ac vs mlp_ac_sep both have `actor.weight` -- torch raises
        # on the size mismatch instead, which is equally loud.)
        if not (set(self.global_policy.state_dict()) & set(sd)):
            raise SystemExit(
                f"[resume] {ckpt_path} shares no parameter names with "
                f"{type(self.global_policy).__name__}; strict=False would silently "
                "reinitialise the whole policy. Check that policy.kind matches the "
                "run that wrote this checkpoint.")
        missing, unexpected = self.global_policy.load_state_dict(sd, strict=False)
        new_heads = [m for m in missing if not m.startswith("trunk.")]
        print(f"[resume] {ckpt_path}: copied trunk + prior heads; {len(new_heads)} "
              f"fresh new-head params, {len(unexpected)} unexpected; "
              f"skipping tasks 1..{after_task}")
        src = _os.path.join(_os.path.dirname(str(ckpt_path)), "eval_matrix.json")
        if _os.path.exists(src):
            # Keep only the rows up to after_task (later rows belong to tasks we are
            # NOT carrying over -- e.g. dropping Boxing); the resumed run appends fresh.
            self.eval_matrix = _json.load(open(src))[:after_task]
            print(f"[resume] reloaded eval_matrix -> {len(self.eval_matrix)} rows "
                  f"(truncated to task {after_task})")
        # Carry over the local reference scores of the skipped tasks (needed by the
        # V5 all-tasks retention stop, since their local phases are not re-run).
        rsrc = _os.path.join(_os.path.dirname(str(ckpt_path)), "resource_usage.json")
        if _os.path.exists(rsrc):
            self._resource = _json.load(open(rsrc))
            print(f"[resume] reloaded resource_usage (local refs for "
                  f"{len(self._resource)} games)")
        # Restore the optimiser/dual state (contract §5): cumulative_step (eval-cadence
        # + frame counter) and the persistent mu. mu/local optimizer are reinitialised
        # each phase, so cumulative_step is the only cross-phase state to carry.
        optp = _os.path.join(_os.path.dirname(str(ckpt_path)),
                             "checkpoints", f"optim_after_task{after_task}.pt")
        if _os.path.exists(optp):
            st = torch.load(optp, map_location=self.device, weights_only=False)
            self.cumulative_step = int(st.get("cumulative_step", 0))
            if getattr(self.dual_cfg, "warm_start", False) and "mu_value" in st:
                self.mu_ctrl.value = float(st["mu_value"])
            print(f"[resume] restored optim state (cumulative_step="
                  f"{self.cumulative_step})")
        self._start_task = after_task + 1

    def _save_progress(self, k: int) -> None:
        """Persist the per-task global checkpoint + running eval matrix + resource
        log so a preemption/crash after task k retains progress (and gives per-task
        checkpoints ``global_after_task{k}.pt`` for the windowed expert-agreement
        eval and for :meth:`resume`). Cheap relative to a phase; called once per task."""
        self.logger.save_json("eval_matrix.json", self.eval_matrix)
        self.logger.save_json("resource_usage.json", self._resource)
        torch.save(self.global_policy.state_dict(),
                   self.logger.run_dir / f"global_after_task{k}.pt")
        # Contract checkpoints (checkpoints/): global + local specialist + optim/dual
        # state (cumulative_step + mu) so resume restores the method exactly, not just
        # the weights. mu/local optimizer are phase-local (reinit each phase), so at a
        # task boundary cumulative_step is the only cross-phase state to carry.
        if self.clog is not None:
            local_sd = None
            lp = self.logger.run_dir / f"local_after_task{k}.pt"
            if lp.exists():
                local_sd = torch.load(lp, map_location="cpu", weights_only=False)
            optim_state = {"cumulative_step": self.cumulative_step,
                           "mu_value": float(getattr(self.mu_ctrl, "value", 0.0))}
            self.clog.save_checkpoint(k, self.global_policy.state_dict(),
                                      local_sd, optim_state)

    def _record_eval_row(self, k: int, n: int) -> None:
        """End-of-task eval + logging, honoring the matrix_eval_every cadence.

        Full seen-row (O(k)) only every ``matrix_eval_every`` tasks and at the final
        task; otherwise eval ONLY the just-finished task (its diagonal, O(1)) and
        carry the prior columns forward so eval_matrix stays dense. The scalar metrics
        (PERF/BWT/forgetting/retention) come from the diagonal + the exact final row,
        so this is metric-preserving; only the intermediate forgetting-trajectory is
        coarsened."""
        every = self.ppo.matrix_eval_every
        full = every <= 0 or k % every == 0 or k == n
        if full:
            row, stds = self._evaluate_row(k)
            self._emit_row(k, row, "global", self.ppo.eval_episodes)
        else:
            cur, _ = self._eval_report(self.global_policy, self.family.tasks[k - 1])
            prev = list(self.eval_matrix[-1]) if self.eval_matrix else []
            row = (prev + [0.0] * (k - len(prev)))[:k]
            row[k - 1] = cur          # fresh diagonal; other columns carried forward
            if self.clog is not None:  # emit only the fresh EXACT measurement
                self.clog.eval(task_idx=k - 1, phase="global", it=-1,
                               evaluated_on=k - 1,
                               evaluated_on_task=self.family.tasks[k - 1].spec.name,
                               raw=float(cur), episodes=self.ppo.eval_episodes,
                               greedy=True, seen=True)
        self.eval_matrix.append(row)
        self.logger.log({"phase": "eval", "task": k, "values": row})
        self._save_progress(k)

    def run(self) -> list[list[float]]:
        if self.method == "joint":
            self._train_joint()
            return self.eval_matrix
        if self.method in ("clear", "recall"):
            from crl.ppo.clear import ClearTrainer, RecallTrainer, ReplayStore
            TrainerCls = RecallTrainer if self.method == "recall" else ClearTrainer
            self._clear_trainer = TrainerCls(self.ppo, self.device, self.logger,
                                             self._log_every)
            self._clear_replay = ReplayStore()

        self._init_contract()
        start = getattr(self, "_start_task", 1)
        if start > 1 and self.clog is not None:
            self.clog.note(f"resumed after task {start - 1}")
        if start <= 1:
            if self.method in ("componet", "cka_rl"):
                self._grow_task(1)
            else:
                self._train_first_task()
                if self.method in ("clear", "recall"):  # store task-1 behavior as a cloning target
                    self._clear_trainer.snapshot(self.global_policy,
                                                 self.family.tasks[0], self._clear_replay)
            self._record_eval_row(1, len(self.family))

        for k in range(max(2, start), len(self.family) + 1):
            if self.method == "finetune":
                self._finetune_task(k)
            elif self.method in ("clear", "recall"):
                self._clear_task(k)
            elif self.method == "constrained":
                self._constrained_task(k)
            elif self.method == "baseline":
                self._baseline_task(k)
            elif self.method in ("componet", "cka_rl"):
                self._grow_task(k)
            else:
                raise KeyError(
                    f"Unknown ppo.method '{self.method}'; available: constrained, "
                    "finetune, clear, recall, baseline, componet, cka_rl, joint"
                )
            self._record_eval_row(k, len(self.family))

        self.logger.save_json("eval_matrix.json", self.eval_matrix)
        torch.save(
            self.global_policy.state_dict(), self.logger.run_dir / "final_policy.pt"
        )
        if self.clog is not None:
            self.clog.note("run complete")
            self.clog.close()
        return self.eval_matrix
