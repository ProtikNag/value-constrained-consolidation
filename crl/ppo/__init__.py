"""Native PPO backend for the constrained min-max continual-RL method.

PPO here is ONLY the policy-gradient optimizer that replaces REINFORCE; the
continual-learning framework (local/global alternation, one-sided squared-
shortfall constraint, lambda/mu dual ascent, replay-free fresh rollouts) is
unchanged. A single reusable :class:`~crl.ppo.trainer.PPOTrainer` core is
specialized into ``LocalTrainer`` (standard PPO) and ``GlobalTrainer``
(PPO + the actor-only constraint term).
"""

from crl.ppo.collector import RolloutBatch, RolloutCollector, compute_gae
from crl.ppo.evaluate import evaluate_value_and_score
from crl.ppo.trainer import GlobalTrainer, LocalTrainer, PPOTrainer

__all__ = [
    "RolloutBatch",
    "RolloutCollector",
    "compute_gae",
    "evaluate_value_and_score",
    "PPOTrainer",
    "LocalTrainer",
    "GlobalTrainer",
]
