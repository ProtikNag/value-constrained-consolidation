"""Standard continual-learning baselines for comparison.

These are the reference points the paper's result must beat. They use the same
network and estimator as the method; only the training procedure differs, so a
comparison isolates what the min-max constraint contributes.

* sequential_finetune -- one network trained on each task in arrival order,
  optimizing only the current task. The canonical catastrophic-forgetting
  baseline: it forgets OLD tasks (the newest is fine). This is the "single
  network learning all tasks sequentially" baseline.
* joint_multitask -- one network trained on all tasks at once (the sum of
  per-task objectives). Not continual; it is the retention UPPER BOUND the
  method aspires to.

Both emit the same run artifacts as the main trainer (``logs.jsonl`` with
probe/eval records, ``eval_matrix.json``, ``final_policy.pt``) so the existing
plotting and evaluation code works on them unchanged.
"""

from __future__ import annotations

import torch

from crl.config import Config
from crl.envs.base import TaskFamily
from crl.estimators.base import ValueEstimator
from crl.logging_utils import RunLogger
from crl.policies.base import Policy


def _optimizer(kind: str, params, lr: float) -> torch.optim.Optimizer:
    return (torch.optim.Adam if kind == "adam" else torch.optim.SGD)(params, lr=lr)


class _BaselineRunner:
    """Shared plumbing: probing, evaluation matrix, and artifact saving."""

    def __init__(self, config: Config, family: TaskFamily, policy: Policy,
                 estimator: ValueEstimator, logger: RunLogger) -> None:
        self.cfg = config.trainer
        self.family = family
        self.policy = policy
        self.estimator = estimator
        self.logger = logger
        self.log_every = config.experiment.log_every
        self.omega = [1.0 / len(family)] * len(family)
        self.cumulative_step = 0

    def _report_eval(self, task, num_episodes=None) -> float:
        """Undiscounted return (task performance) when report_return is set."""
        if self.cfg.report_return:
            return self.estimator.evaluate_return(self.policy, task, num_episodes)
        return self.estimator.evaluate(self.policy, task, num_episodes)

    def _probe(self, current_task: int, phase_type: str) -> None:
        self.cumulative_step += 1
        every = self.cfg.eval_probe_every
        if not every or self.cumulative_step % every != 0:
            return
        values = [self._report_eval(self.family.tasks[i])
                  for i in range(len(self.family))]
        self.logger.log({"phase": "probe", "cumulative_step": self.cumulative_step,
                         "current_task": current_task, "phase_type": phase_type,
                         "values": values})

    def _eval_row(self) -> list[float]:
        return [self._report_eval(self.family.tasks[i], self.cfg.eval_episodes)
                for i in range(len(self.family))]

    def _finish(self, eval_matrix: list[list[float]]) -> list[list[float]]:
        self.logger.save_json("eval_matrix.json", eval_matrix)
        torch.save(self.policy.state_dict(), self.logger.run_dir / "final_policy.pt")
        return eval_matrix


def sequential_finetune(config: Config, family: TaskFamily, policy: Policy,
                        estimator: ValueEstimator, logger: RunLogger) -> list[list[float]]:
    """Train ``policy`` on each task in order, optimizing only the current task."""
    runner = _BaselineRunner(config, family, policy, estimator, logger)
    cfg = runner.cfg
    later_steps = cfg.cycles_per_task * (cfg.local_steps + cfg.global_steps)
    eval_matrix: list[list[float]] = []

    for k in range(1, len(family) + 1):
        task = family.tasks[k - 1]
        steps = cfg.task1_steps if k == 1 else later_steps
        # Fresh optimizer per task, mirroring a naive continual pipeline.
        optimizer = _optimizer(cfg.optimizer, policy.parameters(), cfg.lr_global)
        for step in range(steps):
            objective, entropy_term, stats = estimator.surrogate_objective(policy, task)
            loss = -(objective + cfg.entropy_coef * entropy_term)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            runner._probe(current_task=k, phase_type="finetune")
            if step % runner.log_every == 0:
                logger.log({"phase": "finetune", "task": k, "step": step, **stats})
        eval_matrix.append(runner._eval_row())
        logger.log({"phase": "eval", "task": k, "values": eval_matrix[-1]})

    return runner._finish(eval_matrix)


def joint_multitask(config: Config, family: TaskFamily, policy: Policy,
                    estimator: ValueEstimator, logger: RunLogger) -> list[list[float]]:
    """Train ``policy`` on all tasks simultaneously (retention upper bound)."""
    runner = _BaselineRunner(config, family, policy, estimator, logger)
    cfg = runner.cfg
    total_steps = cfg.task1_steps + (len(family) - 1) * cfg.cycles_per_task * (
        cfg.local_steps + cfg.global_steps)
    optimizer = _optimizer(cfg.optimizer, policy.parameters(), cfg.lr_global)

    for step in range(total_steps):
        objective = torch.zeros(())
        entropy = torch.zeros(())
        for i, task in enumerate(family.tasks):
            obj_i, ent_i, _ = estimator.surrogate_objective(policy, task)
            objective = objective + runner.omega[i] * obj_i
            entropy = entropy + ent_i
        loss = -(objective + cfg.entropy_coef * entropy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # "current_task" is meaningless jointly; label 0 so plots can skip it.
        runner._probe(current_task=0, phase_type="joint")

    eval_matrix = [runner._eval_row()]
    logger.log({"phase": "eval", "task": len(family), "values": eval_matrix[-1]})
    return runner._finish(eval_matrix)
