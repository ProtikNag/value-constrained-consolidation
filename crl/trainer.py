"""Alternating local/global primal-dual trainer.

Implements the method of the paper exactly:

* eq 2      -- each local phase starts from the global: theta^(0) = phi
* eq 7-9    -- local problem: maximize the current-task lead over the global,
               subject to a per-task one-sided *squared* shortfall on each
               past task, F_{L,i} = max(0, V_i^G - V_i^L)^2 <= eps_i, one
               multiplier lambda_i per past task
* eq 11-13  -- global problem: maximize the past lead over the local, subject
               to a single one-sided squared shortfall on the current task,
               F_G = max(0, V_k^L - V_k^G)^2 <= eps, one multiplier mu
* eq 22-24  -- local primal / dual updates
* eq 30-32  -- global primal / dual updates

The one-sided hinge means a policy is penalized only where it is *below* its
frozen reference (that is what forgetting looks like); being above the
reference is not penalized. Differentiating the squared shortfall turns the
constraint's contribution to the primal update into a scalar coefficient
2 * shortfall times the ordinary value gradient (eqs 18, 26), so the past
tasks enter the local step through coefficients lambda_i * 2 * shortfall_i
and the current task enters the global step through mu * 2 * shortfall_k.

Weights are uniform, omega_i = 1/k (derivation, Setup). Past-task terms are
estimated from fresh rollouts in the old environments (the method is
replay-free: it stores no transitions but assumes the old environments
remain available). Task 1 is degenerate (no past tasks): the global policy
is trained by plain policy-gradient ascent on V_1.

Cost per local step with a sampled backend: one gradient batch on the current
task plus one per past task (``past_task_sampling: all``) or one sampled past
task (``sample``). See the paper for the rollout-budget discussion.
"""

from __future__ import annotations

import torch

from crl.config import Config
from crl.duals import make_dual
from crl.duals.controllers import DualController
from crl.envs.base import Task, TaskFamily
from crl.estimators.base import ValueEstimator
from crl.logging_utils import RunLogger
from crl.policies.base import Policy, clone_policy


def _make_optimizer(kind: str, params, lr: float) -> torch.optim.Optimizer:
    if kind == "sgd":
        return torch.optim.SGD(params, lr=lr)
    if kind == "adam":
        return torch.optim.Adam(params, lr=lr)
    raise KeyError(f"Unknown optimizer '{kind}'; available: sgd, adam")


class AlternationTrainer:
    """Runs the full task sequence and records diagnostics."""

    def __init__(
        self,
        config: Config,
        family: TaskFamily,
        global_policy: Policy,
        estimator: ValueEstimator,
        logger: RunLogger,
    ) -> None:
        self.cfg = config.trainer
        self.dual_cfg = config.duals
        self.family = family
        self.global_policy = global_policy
        self.estimator = estimator
        self.logger = logger
        self.log_every = config.experiment.log_every

        num_tasks = len(family)
        # Uniform weights follow the derivation (Setup): omega_i = 1/k where k is
        # the CURRENT task count (number of tasks seen so far), not 1/num_tasks.
        # The weight per task therefore shrinks as tasks accumulate. When an
        # explicit omega list is given it is used verbatim (indexed by task).
        self._uniform_omega = self.cfg.omega is None
        if self._uniform_omega:
            self.omega = None
        else:
            if len(self.cfg.omega) != num_tasks:
                raise ValueError(
                    f"trainer.omega has {len(self.cfg.omega)} entries "
                    f"but the family has {num_tasks} tasks."
                )
            self.omega = [float(w) for w in self.cfg.omega]

        # eps may be a scalar (broadcast to every past task) or a per-task list.
        if isinstance(self.cfg.eps, (int, float)):
            self._eps_scalar: float | None = float(self.cfg.eps)
            self._eps_list: list[float] | None = None
        else:
            self._eps_scalar = None
            self._eps_list = [float(e) for e in self.cfg.eps]

        # One persistent lambda controller per past-task index (warm-started
        # across tasks by index); a single persistent mu controller.
        self.lambda_ctrls: dict[int, DualController] = {}
        self.mu_ctrl = make_dual(self.dual_cfg)

        self.eval_matrix: list[list[float]] = []  # row per finished task
        self.cumulative_step = 0  # primal steps across all phases (probe x-axis)

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #

    def _omega(self, task_index: int, k: int) -> float:
        """Weight omega_i for task ``task_index`` (0-based) when the current task
        count is ``k``. Uniform default = 1/k for every appeared task (Setup);
        an explicit omega list is indexed directly."""
        if self._uniform_omega:
            return 1.0 / k
        return self.omega[task_index]

    def _eps_local(self, task_index: int) -> float:
        """Tolerance eps_i for past task ``task_index`` (0-based)."""
        return self._eps_scalar if self._eps_scalar is not None else self._eps_list[task_index]

    def _eps_global(self) -> float:
        """Single tolerance eps for the current-task constraint (eq 12)."""
        return self._eps_scalar if self._eps_scalar is not None else self._eps_list[0]

    def _lambda_for(self, task_index: int) -> DualController:
        if task_index not in self.lambda_ctrls:
            self.lambda_ctrls[task_index] = make_dual(self.dual_cfg)
        return self.lambda_ctrls[task_index]

    def _past_indices(self, k: int) -> list[int]:
        """Past-task indices for current task k, honoring the sampling mode."""
        indices = list(range(k - 1))
        if self.cfg.past_task_sampling == "sample" and len(indices) > 1:
            picked = int(torch.randint(len(indices), (1,)).item())
            return [indices[picked]]
        if self.cfg.past_task_sampling not in ("all", "sample"):
            raise KeyError(
                f"Unknown past_task_sampling '{self.cfg.past_task_sampling}'; "
                "available: all, sample"
            )
        return indices

    def _sample_scale(self, k: int, active: list[int]) -> float:
        """Rescale one sampled past task to an unbiased estimate of the sum."""
        if self.cfg.past_task_sampling == "sample" and len(active) == 1 and k > 2:
            return float(k - 1)
        return 1.0

    def _report_eval(self, policy: Policy, task: Task, num_episodes=None) -> float:
        """Reporting metric: undiscounted return (task performance) when
        ``report_return`` is set, else the discounted value. This affects only
        the eval matrix and probes -- the constraint/objective always use the
        discounted value."""
        if self.cfg.report_return:
            return self.estimator.evaluate_return(policy, task, num_episodes)
        return self.estimator.evaluate(policy, task, num_episodes)

    def _evaluate_row(self, k: int) -> list[float]:
        """Row of the forgetting matrix after finishing task k (global policy)."""
        last = len(self.family) if self.cfg.eval_all_tasks else k
        return [
            self._report_eval(
                self.global_policy, self.family.tasks[i], self.cfg.eval_episodes
            )
            for i in range(last)
        ]

    def _maybe_probe(self, current_task: int, phase_type: str) -> None:
        """Record the global policy's value on every task vs cumulative step.

        This is the learning-curve data (Fig 3). The global is frozen during
        local phases, so its curve is flat there by construction; task-boundary
        markers come from the recorded ``current_task``.
        """
        self.cumulative_step += 1
        every = self.cfg.eval_probe_every
        if not every or self.cumulative_step % every != 0:
            return
        values = [
            self._report_eval(self.global_policy, self.family.tasks[i])
            for i in range(len(self.family))
        ]
        self.logger.log(
            {
                "phase": "probe",
                "cumulative_step": self.cumulative_step,
                "current_task": current_task,
                "phase_type": phase_type,
                "values": values,
            }
        )

    # ------------------------------------------------------------------ #
    # phases
    # ------------------------------------------------------------------ #

    def _train_first_task(self) -> None:
        """Plain policy-gradient ascent on task 1 (no constraint exists yet)."""
        task = self.family.tasks[0]
        optimizer = _make_optimizer(
            self.cfg.optimizer, self.global_policy.parameters(), self.cfg.lr_global
        )
        for step in range(self.cfg.task1_steps):
            objective, entropy_term, stats = self.estimator.surrogate_objective(
                self.global_policy, task
            )
            # Task 1 is the current task with k=1, so omega_1 = 1/1 = 1.
            loss = -(self._omega(0, 1) * objective + self.cfg.entropy_coef * entropy_term)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self._maybe_probe(current_task=1, phase_type="task1")
            if step % self.log_every == 0:
                self.logger.log({"phase": "task1", "task": 1, "step": step, **stats})
                print(f"[task1] step={step:4d} V1={stats['value']:.4f}")

    def _local_phase(self, k: int, cycle: int) -> Policy:
        """Eqs 7-10, 22-24: train theta with the global frozen; returns pi_L bar."""
        task_k = self.family.tasks[k - 1]
        frozen_global = clone_policy(self.global_policy, trainable=False)
        local_policy = clone_policy(self.global_policy, trainable=True)  # eq 2

        # Frozen past references V_i^{pi_G}, one large batch per phase.
        ref_episodes = getattr(self.estimator, "episodes_per_ref", None)
        refs_past = {
            i: self.estimator.evaluate(frozen_global, self.family.tasks[i], ref_episodes)
            for i in range(k - 1)
        }

        for i in range(k - 1):
            self._lambda_for(i).reset()
        optimizer = _make_optimizer(
            self.cfg.optimizer, local_policy.parameters(), self.cfg.lr_local
        )

        for step in range(self.cfg.local_steps):
            active = self._past_indices(k)
            scale = self._sample_scale(k, active)

            objective_k, entropy_term_k, stats_k = self.estimator.surrogate_objective(
                local_policy, task_k
            )
            # eq 22: omega_k grad V_k + sum_i lambda_i * 2 * shortfall_i * grad V_i.
            loss = -(self._omega(k - 1, k) * objective_k)
            loss = loss - self.cfg.entropy_coef * entropy_term_k

            lambdas: dict[str, float] = {}
            shortfalls_sq: dict[str, float] = {}
            total_constraint = 0.0
            max_lambda = 0.0
            # Variant: pure-plasticity local. Skip the lambda shortfall terms so
            # the local policy maximizes V_k alone (the global phase still
            # consolidates past tasks under its own mu constraint). We still
            # measure the past shortfalls for logging, but they do not enter the
            # loss and lambda stays at 0.
            for i in active:
                obj_i, _, stats_i = self.estimator.surrogate_objective(
                    local_policy, self.family.tasks[i]
                )
                shortfall = max(0.0, refs_past[i] - stats_i["value"])  # V_i^G - V_i^L
                constraint_i = shortfall * shortfall  # eq 24: squared hinge
                if self.cfg.local_unconstrained:
                    lam = 0.0
                else:
                    lam = self._lambda_for(i).update(constraint_i, self._eps_local(i))
                    # Detached coeff 2 * shortfall multiplies the value gradient.
                    coeff = lam * 2.0 * shortfall * scale
                    loss = loss - coeff * obj_i
                lambdas[f"lambda_{i}"] = lam
                shortfalls_sq[f"F_L_{i}"] = constraint_i
                total_constraint += constraint_i
                max_lambda = max(max_lambda, lam)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self._maybe_probe(current_task=k, phase_type="local")

            if step % self.log_every == 0:
                self.logger.log(
                    {
                        "phase": "local", "task": k, "cycle": cycle, "step": step,
                        "F_L": total_constraint, "lambda": max_lambda,
                        "V_k_local": stats_k["value"], "entropy_k": stats_k["entropy"],
                        **lambdas, **shortfalls_sq,
                    }
                )
                print(
                    f"[local  k={k} c={cycle}] step={step:4d} "
                    f"Vk={stats_k['value']:.4f} sumF_L={total_constraint:.5f} "
                    f"maxlam={max_lambda:.3f}"
                )
        return clone_policy(local_policy, trainable=False)

    def _global_phase(self, k: int, cycle: int, frozen_local: Policy) -> None:
        """Eqs 11-14, 30-32: train phi with the local frozen."""
        task_k = self.family.tasks[k - 1]
        ref_episodes = getattr(self.estimator, "episodes_per_ref", None)
        ref_current = self.estimator.evaluate(frozen_local, task_k, ref_episodes)

        self.mu_ctrl.reset()
        optimizer = _make_optimizer(
            self.cfg.optimizer, self.global_policy.parameters(), self.cfg.lr_global
        )
        for step in range(self.cfg.global_steps):
            active = self._past_indices(k)
            scale = self._sample_scale(k, active)

            # Past objective sum_{i<k} omega_i V_i (eq 30, past term).
            past_objective = torch.zeros(())
            past_values: dict[str, float] = {}
            for i in active:
                obj_i, _, stats_i = self.estimator.surrogate_objective(
                    self.global_policy, self.family.tasks[i]
                )
                past_objective = past_objective + self._omega(i, k) * scale * obj_i
                past_values[f"V_past_{i}"] = stats_i["value"]

            objective_k, entropy_term_k, stats_k = self.estimator.surrogate_objective(
                self.global_policy, task_k
            )
            shortfall = max(0.0, ref_current - stats_k["value"])  # V_k^L - V_k^G
            constraint = shortfall * shortfall  # eq 32: squared hinge
            mu = self.mu_ctrl.update(constraint, self._eps_global())
            # eq 30: sum omega_i grad V_i + mu * 2 * shortfall * grad V_k.
            coeff_k = mu * 2.0 * shortfall
            loss = -(
                past_objective
                + coeff_k * objective_k
                + self.cfg.entropy_coef * entropy_term_k
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            self._maybe_probe(current_task=k, phase_type="global")

            if step % self.log_every == 0:
                self.logger.log(
                    {
                        "phase": "global", "task": k, "cycle": cycle, "step": step,
                        "F_G": constraint, "mu": mu, "V_k_global": stats_k["value"],
                        "V_k_ref_local": ref_current, **past_values,
                    }
                )
                print(
                    f"[global k={k} c={cycle}] step={step:4d} "
                    f"Vk={stats_k['value']:.4f} F_G={constraint:.5f} mu={mu:.3f}"
                )

    def _log_gaps(self, k: int, cycle: int, frozen_local: Policy) -> None:
        """Gap sequences (always logged; the alternation-cycling diagnostic)."""
        task_k = self.family.tasks[k - 1]
        gap_current = self.estimator.evaluate(
            frozen_local, task_k
        ) - self.estimator.evaluate(self.global_policy, task_k)
        past_gaps = {
            f"gap_past_{i}": self.estimator.evaluate(
                self.global_policy, self.family.tasks[i]
            )
            - self.estimator.evaluate(frozen_local, self.family.tasks[i])
            for i in range(k - 1)
        }
        self.logger.log(
            {
                "phase": "gaps", "task": k, "cycle": cycle,
                "gap_current": gap_current, **past_gaps,
            }
        )

    # ------------------------------------------------------------------ #
    # main loop
    # ------------------------------------------------------------------ #

    def run(self) -> list[list[float]]:
        """Train the whole sequence; returns the evaluation matrix."""
        self._train_first_task()
        self.eval_matrix.append(self._evaluate_row(1))
        self.logger.log({"phase": "eval", "task": 1, "values": self.eval_matrix[-1]})

        for k in range(2, len(self.family) + 1):
            for cycle in range(self.cfg.cycles_per_task):
                frozen_local = self._local_phase(k, cycle)
                self._global_phase(k, cycle, frozen_local)
                self._log_gaps(k, cycle, frozen_local)
            self.eval_matrix.append(self._evaluate_row(k))
            self.logger.log({"phase": "eval", "task": k, "values": self.eval_matrix[-1]})

        self.logger.save_json("eval_matrix.json", self.eval_matrix)
        torch.save(
            self.global_policy.state_dict(), self.logger.run_dir / "final_policy.pt"
        )
        return self.eval_matrix
