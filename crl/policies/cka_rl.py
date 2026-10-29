"""CKA-RL (Hu et al., NeurIPS 2025, "Continual Knowledge Adaptation for RL")
as a discrete-action MLP actor-critic policy for this codebase's policy interface.

Paper method and reference implementation: Hu et al., 2026.

  A FuseLinear layer represents its effective weight as

      W_eff = W_base + sum_i  softmax(alpha * alpha_scale)_i * tau_i          (weight)
      b_eff = b_base + sum_i  softmax(alpha * alpha_scale)_i * beta_i         (bias)

  where {tau_i, beta_i} are "knowledge vectors" -- parameter deltas of past
  tasks relative to the current base -- kept FROZEN, and `alpha` (a short
  learnable vector, one entry per pooled tau) + scalar `alpha_scale` are the
  only NEW learnable combination parameters trained on the new task. A pool of
  past vectors is kept; when it exceeds `pool_size` the two most cosine-similar
  vectors are merged by averaging (their merge_vectors()).

This is the adaptation of that mechanism:
  * SAC's (fc_mean, fc_logstd) fuse heads collapse to ONE FuseLinear ACTOR head
    that outputs Categorical action logits (discrete actions).
  * No file-based checkpointing (their torch.load(model.pt)): the pool lives in
    memory as buffers/tensors; add_task(k) snapshots the just-trained head.

Ref->code map (fuse_module.FuseLinear):
  forward (ref L87-103)  -> FuseLinear.forward
  merge_weight (ref L63) -> FuseLinear.fold_into_base (delta_theta_mode="TAT")
  set_base_and_vectors (ref L108) -> FuseLinear.set_base_and_vectors
  get_vectors (ref L129)          -> FuseLinear.get_vectors
  merge_vectors (cka_rl L214/202) -> FuseLinear._merge_pool
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from crl.policies.base import Policy
from crl.policies.mlp import _mlp_trunk


class FuseLinear(nn.Module):
    """Linear layer whose weight = base + softmax(alpha*alpha_scale) @ tau_pool.

    base weight/bias are trainable Parameters (the "theta" trained on the
    current task). The tau pool (`weights` / `biaes`, matching the reference
    field names) are FROZEN buffers. `alpha` (len = pool size) and scalar
    `alpha_scale` are the only new learnable combination parameters. With an
    empty pool this is a plain nn.Linear (alpha is None).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self._bias = bias
        # Base theta (trainable) -- mirrors FuseLinear.weight / .bias (ref L40,L50).
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        # tau pool as frozen buffers (ref self.weights/self.biaes, requires_grad=False).
        # Empty (num_weights=0) until the first add_task snapshot.
        self.register_buffer("weights", torch.empty(0, out_features, in_features))
        if bias:
            self.register_buffer("biaes", torch.empty(0, out_features))
        else:
            self.biaes = None
        # Frozen anchor theta_base = theta_1 (the paper's fixed base): every tau is
        # stored as a delta vs this, and each new task's head is seeded from it.
        # None until fixed after task 1 (ref: base_dir checkpoint, kept in-memory).
        self.register_buffer("theta_base_w", None)
        self.register_buffer("theta_base_b", None)
        # Combination params: None while pool is empty (ref alpha=None path, L97).
        self.alpha: nn.Parameter | None = None
        self.alpha_scale: nn.Parameter | None = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / fan_in ** 0.5 if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    @property
    def num_weights(self) -> int:
        return self.weights.shape[0]

    def _combined(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        """(W_eff, b_eff) = base + softmax(alpha*alpha_scale) @ tau  (ref L87-103)."""
        if self.alpha is None or self.num_weights == 0:
            return self.weight, self.bias
        a = F.softmax(self.alpha * self.alpha_scale, dim=0)          # ref L91
        weight = self.weight + (a.view(-1, 1, 1) * self.weights).sum(dim=0)  # ref L92
        bias = self.bias
        if self._bias:
            bias = self.bias + (a.view(-1, 1) * self.biaes).sum(dim=0)       # ref L94
        return weight, bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight, bias = self._combined()
        return F.linear(x, weight, bias)

    @torch.no_grad()
    def set_alpha(self, alpha: nn.Parameter | None, alpha_scale: nn.Parameter | None) -> None:
        self.alpha = alpha
        self.alpha_scale = alpha_scale

    @torch.no_grad()
    def set_base_and_vectors(self, base: dict | None, vectors: dict | None) -> None:
        """Copy base theta and the frozen tau pool in (ref FuseLinear L108)."""
        if base is not None:
            self.weight.data.copy_(base["weight"])
            if self._bias and base.get("bias") is not None:
                self.bias.data.copy_(base["bias"])
        if vectors is not None and vectors["weight"].shape[0] > 0:
            self.weights = vectors["weight"].detach().clone()
            if self._bias:
                self.biaes = vectors["bias"].detach().clone()

    @torch.no_grad()
    def get_vectors(self, base: dict | None = None) -> tuple[dict, int]:
        """Return {current-task delta prepended to existing pool}, mirroring ref
        get_vectors (L129): tau_new = theta_current - theta_base, stacked on top
        of the existing pool. base=None -> delta vs zero (i.e. raw theta)."""
        if base is None:
            base_w = torch.zeros_like(self.weight)
            base_b = torch.zeros_like(self.bias) if self._bias else None
        else:
            base_w = base["weight"]
            base_b = base.get("bias") if self._bias else None
        new_w = (self.weight.detach() - base_w).unsqueeze(0)
        weights = torch.cat([new_w, self.weights], dim=0)
        out = {"weight": weights}
        if self._bias:
            new_b = (self.bias.detach() - base_b).unsqueeze(0)
            out["bias"] = torch.cat([new_b, self.biaes], dim=0)
        return out, weights.shape[0]

    @torch.no_grad()
    def fix_base(self) -> None:
        """Freeze the current head as theta_base = theta_1 (called once, after
        task 1). All future tau vectors are deltas vs this fixed anchor and each
        new task's head is seeded from it (paper: fixed base_dir checkpoint)."""
        self.theta_base_w = self.weight.detach().clone()
        if self._bias:
            self.theta_base_b = self.bias.detach().clone()

    @torch.no_grad()
    def seed_from_base(self) -> None:
        """Reset the trainable head to the fixed theta_base so the new task learns
        a residual on top of the same anchor the tau deltas are measured against."""
        if self.theta_base_w is None:
            return
        self.weight.data.copy_(self.theta_base_w)
        if self._bias:
            self.bias.data.copy_(self.theta_base_b)

    @torch.no_grad()
    def snapshot_delta(self) -> None:
        """Push (current head - theta_base) as a NEW frozen tau, prepended to the
        pool (ref get_vectors: tau_new = theta - theta_base vs the fixed base)."""
        base = None
        if self.theta_base_w is not None:
            base = {"weight": self.theta_base_w}
            if self._bias:
                base["bias"] = self.theta_base_b
        vectors, _ = self.get_vectors(base=base)
        self.weights = vectors["weight"].detach().clone()
        if self._bias:
            self.biaes = vectors["bias"].detach().clone()

    @torch.no_grad()
    def fold_into_base(self) -> None:
        """delta_theta_mode="TAT": fold alpha-weighted taus into the base (ref
        merge_weight L63). Used so a snapshot captures theta + alpha*tau."""
        if self.alpha is None or self.num_weights == 0:
            return
        a = F.softmax(self.alpha * self.alpha_scale, dim=0)
        self.weight.data = self.weight.data + (a.view(-1, 1, 1) * self.weights).sum(dim=0)
        if self._bias:
            self.bias.data = self.bias.data + (a.view(-1, 1) * self.biaes).sum(dim=0)

    @staticmethod
    def _merge_one(pool: torch.Tensor, target: int) -> torch.Tensor:
        """Repeatedly average the two most cosine-similar vectors until the pool
        is `target` long (ref merge_vectors, cka_rl.py L214-233)."""
        while pool.shape[0] > target:
            n = pool.shape[0]
            flat = pool.reshape(n, -1)
            # Upper-triangular cosine-similarity search (ref fills sims with -1).
            sims = torch.full((n, n), -1.0, device=pool.device)
            for i in range(n):
                for j in range(i + 1, n):
                    sims[i, j] = F.cosine_similarity(flat[i], flat[j], dim=0)
            idx = int(torch.argmax(sims))
            i1, i2 = divmod(idx, n)            # ref divmod(max_sim_idx, n)
            merged = ((pool[i1] + pool[i2]) / 2).unsqueeze(0)   # average (ref L225)
            keep = torch.cat([pool[:i1], pool[i1 + 1:i2], pool[i2 + 1:]], dim=0)
            pool = torch.cat([keep, merged], dim=0)             # ref reappends merged
        return pool

    @torch.no_grad()
    def merge_pool(self, pool_size: int) -> None:
        """Shrink the tau pool to `pool_size` by cosine-similarity averaging."""
        if self.num_weights <= pool_size:
            return
        self.weights = self._merge_one(self.weights, pool_size)
        if self._bias:
            self.biaes = self._merge_one(self.biaes, pool_size)


class CkaRlPolicy(Policy):
    """Discrete-action MLP actor-critic with a CKA-RL FuseLinear actor head.

    Architecture: a shared Tanh MLP trunk (like MLP policies' _mlp_trunk) ->
    a FuseLinear ACTOR head (logits over num_actions) + a plain critic head.
    The trunk and critic are ordinary trainable params; the actor head is the
    fused knowledge-adaptation head. Per the CKA-RL method, on each new task
    only the trunk/critic + a fresh short `alpha` (+ actor base) train, while
    the stored tau vectors stay frozen.

    Orchestrator contract: call ``add_task(k)`` (k 1-based) BEFORE training
    each task, then run the normal Local PPO loop (RolloutCollector +
    LocalTrainer.optimize_batches) on that task.
    """

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        hidden_sizes: list[int],
        num_tasks: int,
        pool_size: int = 5,
        task_conditioned: bool = False,
    ) -> None:
        super().__init__()
        self.num_tasks = num_tasks
        self.pool_size = pool_size
        self.task_conditioned = task_conditioned
        if task_conditioned and num_tasks <= 0:
            raise ValueError("task_conditioned=True requires num_tasks > 0.")

        input_dim = obs_dim + (num_tasks if task_conditioned else 0)
        self.trunk, last = _mlp_trunk(input_dim, hidden_sizes)
        self.actor = FuseLinear(last, num_actions)   # fused head (ref fc_mean)
        self.critic = nn.Linear(last, 1)
        # Small actor init keeps the initial policy near-uniform (as _init_head).
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)
        # Fix the anchor theta_base = the near-uniform init (the paper's fixed base
        # that every tau delta is measured against, kept in-memory). tau_j =
        # theta_after_task_j - theta_base, so task 1's residual is a real vector.
        self.actor.fix_base()

        self._tasks_added = 0
        # Per-task alpha vectors live here so they persist / are optimizable.
        self.alphas = nn.ParameterList()
        self.alpha_scales = nn.ParameterList()
        # Frozen per-task EFFECTIVE actor head (W_eff, b_eff), keyed by 0-based task.
        # CKA-RL evaluates each past task with ITS OWN learned combination over the
        # (shared, latest) trunk -- the in-memory analogue of the reference saving
        # each task's model. Populated by store_eval_head(k) after task k trains.
        self._eval_heads: dict[int, tuple[torch.Tensor, torch.Tensor | None]] = {}

    # -- CKA-RL task lifecycle -------------------------------------------------

    def add_task(self, k: int) -> None:
        """Register task ``k`` (1-based) BEFORE training it.

        For k == 1: no pool yet -> plain linear actor, everything trains.
        For k >= 2: snapshot the just-finished task's actor head as a NEW tau
        vector (theta - theta_base, vs the FIXED anchor), merge the pool down to
        pool_size, seed the head back to theta_base, allocate a FRESH learnable
        alpha (len = pool size) + alpha_scale, and set requires_grad so the trunk
        + critic + alpha (+ actor base) train while the tau pool stays frozen.
        """
        if k != self._tasks_added + 1:
            raise ValueError(f"add_task expects k={self._tasks_added + 1}, got {k}.")

        if k >= 2:
            # 1) Fold the head just trained (theta + alpha*tau) so the snapshot
            #    captures the full effective head (delta_theta_mode="TAT", ref).
            self.actor.fold_into_base()          # theta <- theta + alpha*tau (ref TAT)
            # 2) Push tau_new = theta - theta_base (fixed anchor) onto the pool
            #    (ref get_vectors: delta vs the fixed base_dir checkpoint).
            self.actor.snapshot_delta()
            # 3) Merge pool down to pool_size by cosine-sim averaging (ref merge_vectors).
            self.actor.merge_pool(self.pool_size)
            # 4) Seed the trainable head back to theta_base so the new task learns a
            #    residual on top of the SAME anchor the tau deltas are measured
            #    against (ref set_base_and_vectors copies theta_base into weight).
            self.actor.seed_from_base()
            # 5) Fresh learnable alpha (len = current pool) + scalar scale (ref setup_alpha
            #    Randn init: randn/num_vectors; alpha_scale=ones).
            n = self.actor.num_weights
            alpha = nn.Parameter(torch.randn(n) / n)
            alpha_scale = nn.Parameter(torch.ones(1))
            self.alphas.append(alpha)
            self.alpha_scales.append(alpha_scale)
            self.actor.set_alpha(alpha, alpha_scale)

        self._tasks_added = k
        self._set_trainable()

    def _set_trainable(self) -> None:
        """Trunk + critic + actor base + current alpha train; tau pool frozen
        (tau are buffers, already non-grad). Past-task alphas are frozen so only
        the CURRENT task's combination adapts (ref: one alpha per new task)."""
        for p in self.trunk.parameters():
            p.requires_grad_(True)
        for p in self.critic.parameters():
            p.requires_grad_(True)
        self.actor.weight.requires_grad_(True)
        if self.actor.bias is not None:
            self.actor.bias.requires_grad_(True)
        for i, (a, s) in enumerate(zip(self.alphas, self.alpha_scales)):
            train = i == len(self.alphas) - 1
            a.requires_grad_(train)
            s.requires_grad_(train)

    # -- Policy interface ------------------------------------------------------

    def _augment(self, obs: torch.Tensor, task_id: int) -> torch.Tensor:
        if not self.task_conditioned:
            return obs
        one_hot = torch.zeros(obs.shape[0], self.num_tasks, device=obs.device,
                              dtype=obs.dtype)
        one_hot[:, task_id] = 1.0
        return torch.cat([obs, one_hot], dim=-1)

    @torch.no_grad()
    def store_eval_head(self, k: int) -> None:
        """Freeze task k's (1-based) EFFECTIVE actor head after it finishes training,
        so later evaluation of task k uses its own learned combination, not a later
        task's. Orchestrator calls this once after training each task, BEFORE the
        next add_task folds/reseeds the head."""
        w, b = self.actor._combined()
        self._eval_heads[k - 1] = (w.detach().clone(),
                                   b.detach().clone() if b is not None else None)

    def _actor_logits(self, feats: torch.Tensor, task_id: int) -> torch.Tensor:
        """Task task_id's frozen effective head if it has finished training, else the
        LIVE fused head (the currently-training task)."""
        head = self._eval_heads.get(task_id)
        if head is not None:
            return F.linear(feats, head[0], head[1])
        return self.actor(feats)

    def dist(self, obs: torch.Tensor, task_id: int) -> Categorical:
        feats = self.trunk(self._augment(obs, task_id))
        return Categorical(logits=self._actor_logits(feats, task_id))

    def value(self, obs: torch.Tensor, task_id: int) -> torch.Tensor:
        return self.critic(self.trunk(self._augment(obs, task_id))).squeeze(-1)

    def dist_value(self, obs: torch.Tensor, task_id: int):
        feats = self.trunk(self._augment(obs, task_id))
        return (Categorical(logits=self._actor_logits(feats, task_id)),
                self.critic(feats).squeeze(-1))
