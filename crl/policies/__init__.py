"""Policy registry and construction from config."""

from __future__ import annotations

from crl.config import PolicyConfig
from crl.envs.base import TaskFamily
from crl.policies.base import Policy, clone_policy
from crl.policies.cnn import MinAtarCNNPolicy, MinAtarMultiHeadCNNPolicy
from crl.policies.cnn_ac import (
    AtariActorCriticPolicy,
    AtariMultiHeadActorCriticPolicy,
)
from crl.policies.impala import (
    ImpalaActorCriticPolicy,
    ImpalaMultiHeadActorCriticPolicy,
)
from crl.policies.cka_rl import CkaRlPolicy
from crl.policies.cka_rl_cnn import CkaRlCNNPolicy
from crl.policies.mlp_ac_cbp import MLPActorCriticCbpPolicy
from crl.policies.mlp_ac_crelu import MLPActorCriticCReLUPolicy
from crl.policies.componet import CompoNetPolicy
from crl.policies.componet_cnn import CompoNetCNNPolicy
from crl.policies.mlp import (
    MLPActorCriticPolicy,
    MLPSeparateActorCriticPolicy,
    MLPSharedActorMultiCriticPolicy,
    MLPMultiHeadActorCriticPolicy,
    MLPPolicy,
    MultiHeadMLPPolicy,
)
from crl.policies.tabular import TabularPolicy


def make_policy(cfg: PolicyConfig, family: TaskFamily) -> Policy:
    """Instantiate the policy named in the config for the given family."""
    if cfg.kind == "tabular":
        return TabularPolicy(family.obs_dim, family.num_actions)
    if cfg.kind == "mlp":
        return MLPPolicy(
            obs_dim=family.obs_dim,
            num_actions=family.num_actions,
            hidden_sizes=list(cfg.hidden_sizes),
            task_conditioned=cfg.task_conditioned,
            num_tasks=len(family),
        )
    if cfg.kind == "multihead":
        return MultiHeadMLPPolicy(
            obs_dim=family.obs_dim,
            num_actions=family.num_actions,
            hidden_sizes=list(cfg.hidden_sizes),
            num_tasks=len(family),
            task_conditioned=cfg.task_conditioned,
        )
    if cfg.kind == "mlp_ac":
        # Single shared actor+critic head (no per-task heads); pair with
        # task_conditioned or a goal-in-obs family so the critic stays well-posed.
        return MLPActorCriticPolicy(
            obs_dim=family.obs_dim,
            num_actions=family.num_actions,
            hidden_sizes=list(cfg.hidden_sizes),
            num_tasks=len(family),
            task_conditioned=cfg.task_conditioned,
        )
    if cfg.kind == "mlp_ac_sep":
        # Actor and critic as two INDEPENDENT networks (no shared trunk), single
        # head each. Same actor capacity as "mlp_ac"; the extra parameters are all
        # in the critic, which never acts. Pair with task_conditioned.
        return MLPSeparateActorCriticPolicy(
            obs_dim=family.obs_dim,
            num_actions=family.num_actions,
            hidden_sizes=list(cfg.hidden_sizes),
            num_tasks=len(family),
            task_conditioned=cfg.task_conditioned,
        )
    if cfg.kind == "mlp_sharedactor_multicritic":
        # ONE shared actor head across all tasks (the headline architecture),
        # plus an independent critic per task so no single critic is overloaded
        # by fifty value functions. Critics never act and are dropped at eval.
        return MLPSharedActorMultiCriticPolicy(
            obs_dim=family.obs_dim,
            num_actions=family.num_actions,
            hidden_sizes=list(cfg.hidden_sizes),
            num_tasks=len(family),
            task_conditioned=cfg.task_conditioned,
        )
    if cfg.kind == "componet":
        # CompoNet (Malagón et al. ICML'24): per-task self-composing modules with
        # attention over frozen predecessors. Driven by ppo.method == "componet".
        return CompoNetPolicy(
            obs_dim=family.obs_dim,
            num_actions=family.num_actions,
            hidden_sizes=list(cfg.hidden_sizes),
            num_tasks=len(family),
            task_conditioned=cfg.task_conditioned,
        )
    if cfg.kind == "cka_rl":
        # CKA-RL (Hu et al. NeurIPS'25): FuseLinear actor head combining a pool of
        # frozen past-task deltas via a learned alpha. Driven by ppo.method=="cka_rl".
        return CkaRlPolicy(
            obs_dim=family.obs_dim,
            num_actions=family.num_actions,
            hidden_sizes=list(cfg.hidden_sizes),
            num_tasks=len(family),
            pool_size=cfg.pool_size,
            task_conditioned=cfg.task_conditioned,
        )
    if cfg.kind == "mlp_ac_cbp":
        # CbpNet / Continual Backprop plasticity baseline (Dohare 2024): shared-head
        # MLP-AC whose trunk resets low-utility mature units each step (via the
        # cbp_step hook in LocalTrainer). Run with method finetune. Fixed capacity.
        return MLPActorCriticCbpPolicy(
            obs_dim=family.obs_dim, num_actions=family.num_actions,
            hidden_sizes=list(cfg.hidden_sizes), num_tasks=len(family),
            task_conditioned=cfg.task_conditioned)
    if cfg.kind == "mlp_ac_crelu":
        # CReLUs plasticity baseline (Abbas 2023): shared-head MLP-AC with CReLU
        # activations; run with method finetune (naive sequential). Fixed capacity.
        return MLPActorCriticCReLUPolicy(
            obs_dim=family.obs_dim, num_actions=family.num_actions,
            hidden_sizes=list(cfg.hidden_sizes), num_tasks=len(family),
            task_conditioned=cfg.task_conditioned)
    if cfg.kind == "mlp_ac_multihead":
        # MLP actor-critic (per-task actor+critic heads) for the PPO backend on
        # flat-vector families (gridworld) -- the non-image analogue of
        # impala_ac_multihead / cnn_ac_multihead.
        return MLPMultiHeadActorCriticPolicy(
            obs_dim=family.obs_dim,
            num_actions=family.num_actions,
            hidden_sizes=list(cfg.hidden_sizes),
            num_tasks=len(family),
            task_conditioned=cfg.task_conditioned,
        )
    if cfg.kind in ("componet_cnn", "cka_rl_cnn"):
        # CompoNet / CKA-RL on the ATARI image backend (Nature-CNN trunk). Driven by
        # ppo.method componet/cka_rl via _grow_task, same as their MLP counterparts.
        obs_shape = getattr(family, "obs_shape", None)
        if obs_shape is None:
            raise ValueError(f"policy '{cfg.kind}' needs a family with obs_shape (e.g. atari).")
        hidden = list(cfg.hidden_sizes)[0] if cfg.hidden_sizes else 512
        if cfg.kind == "componet_cnn":
            return CompoNetCNNPolicy(
                obs_shape=obs_shape, num_actions=family.num_actions,
                hidden_size=hidden, num_tasks=len(family),
                task_conditioned=cfg.task_conditioned)
        return CkaRlCNNPolicy(
            obs_shape=obs_shape, num_actions=family.num_actions,
            hidden_size=hidden, num_tasks=len(family), pool_size=cfg.pool_size,
            task_conditioned=cfg.task_conditioned)
    if cfg.kind in ("cnn_ac", "cnn_ac_multihead"):
        obs_shape = getattr(family, "obs_shape", None)
        if obs_shape is None:
            raise ValueError(f"policy '{cfg.kind}' needs a family with obs_shape "
                             "(e.g. atari), not a flat-vector family.")
        hidden = list(cfg.hidden_sizes)[0] if cfg.hidden_sizes else 512
        ac_cls = (AtariMultiHeadActorCriticPolicy if cfg.kind == "cnn_ac_multihead"
                  else AtariActorCriticPolicy)
        return ac_cls(
            obs_shape=obs_shape,
            num_actions=family.num_actions,
            hidden_size=hidden,
            num_tasks=len(family),
            task_conditioned=cfg.task_conditioned,
        )
    if cfg.kind in ("impala_ac", "impala_ac_multihead"):
        obs_shape = getattr(family, "obs_shape", None)
        if obs_shape is None:
            raise ValueError(f"policy '{cfg.kind}' needs a family with obs_shape "
                             "(e.g. atari), not a flat-vector family.")
        hidden = list(cfg.hidden_sizes)[0] if cfg.hidden_sizes else 512
        cls = (ImpalaMultiHeadActorCriticPolicy if cfg.kind == "impala_ac_multihead"
               else ImpalaActorCriticPolicy)
        return cls(
            obs_shape=obs_shape,
            num_actions=family.num_actions,
            hidden_size=hidden,
            num_tasks=len(family),
            task_conditioned=cfg.task_conditioned,
        )
    if cfg.kind in ("cnn", "cnn_multihead"):
        obs_shape = getattr(family, "obs_shape", None)
        if obs_shape is None:
            raise ValueError(f"policy '{cfg.kind}' needs a family with obs_shape "
                             "(e.g. minatar), not a flat-vector family.")
        hidden = list(cfg.hidden_sizes)[0] if cfg.hidden_sizes else 128
        cls = MinAtarMultiHeadCNNPolicy if cfg.kind == "cnn_multihead" else MinAtarCNNPolicy
        return cls(
            obs_shape=obs_shape,
            num_actions=family.num_actions,
            hidden_size=hidden,
            num_tasks=len(family),
            task_conditioned=cfg.task_conditioned,
        )
    raise KeyError(
        f"Unknown policy kind '{cfg.kind}'; available: tabular, mlp, multihead, "
        "mlp_ac, mlp_ac_multihead, mlp_ac_crelu, mlp_ac_cbp, componet, cka_rl, "
        "componet_cnn, cka_rl_cnn, cnn, "
        "cnn_multihead, cnn_ac, cnn_ac_multihead, impala_ac, impala_ac_multihead"
    )


__all__ = [
    "Policy", "clone_policy", "make_policy",
    "MLPPolicy", "MultiHeadMLPPolicy", "TabularPolicy",
    "AtariActorCriticPolicy", "AtariMultiHeadActorCriticPolicy",
]
