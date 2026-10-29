"""CKA-RL (Hu et al., NeurIPS 2025) as an ATARI (image-obs) actor-critic policy.

The CNN analogue of :class:`crl.policies.cka_rl.CkaRlPolicy` (the MLP version used
for GridWorld). Everything about the CKA-RL knowledge-adaptation mechanism is
IDENTICAL -- we simply swap the Tanh-MLP trunk for a Nature-CNN trunk (uint8
[B,4,84,84] frames, /255 inside forward, like crl.policies.cnn_ac._NatureTrunk)
and reuse the already-verified :class:`crl.policies.cka_rl.FuseLinear` AS-IS for
the fused actor head. The critic is a plain linear head. Separate/per-task
structure (a fresh alpha + tau snapshot per task) is fine here: Atari is a
growth-allowed setting.

Fuse math (unchanged, see FuseLinear docstring):

    W_eff = W_base + sum_i  softmax(alpha * alpha_scale)_i * tau_i

with {tau_i} the FROZEN per-task deltas (theta_after_task_j - theta_base) kept in
a bounded pool merged by cosine-similarity averaging when it exceeds pool_size,
and (alpha, alpha_scale) the only new learnable combination params per task.

Orchestrator contract (same as the MLP version): call ``add_task(k)`` (k 1-based)
BEFORE training each task, run the normal Local PPO loop, then call
``store_eval_head(k)`` once after task k finishes -- BEFORE the next add_task
folds/reseeds the head -- so past tasks evaluate with their OWN effective head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from crl.policies.base import Policy
from crl.policies.cka_rl import FuseLinear  # reuse verified fuse head AS-IS


def _ortho(layer: nn.Module, gain: float) -> nn.Module:
    nn.init.orthogonal_(layer.weight, gain=gain)
    if getattr(layer, "bias", None) is not None:
        nn.init.zeros_(layer.bias)
    return layer


class _NatureTrunk(nn.Module):
    """Nature-CNN body -> ``[B, hidden_size]`` features (mirrors cnn_ac._NatureTrunk).

    task-conditioned variant appends a task one-hot to the flattened conv
    features before the FC, matching cnn_ac.
    """

    def __init__(self, obs_shape, hidden_size, num_tasks, task_conditioned) -> None:
        super().__init__()
        c, h, w = obs_shape
        self.task_conditioned = task_conditioned
        self.num_tasks = num_tasks
        self.conv = nn.Sequential(
            _ortho(nn.Conv2d(c, 32, kernel_size=8, stride=4), gain=2 ** 0.5),
            nn.ReLU(),
            _ortho(nn.Conv2d(32, 64, kernel_size=4, stride=2), gain=2 ** 0.5),
            nn.ReLU(),
            _ortho(nn.Conv2d(64, 64, kernel_size=3, stride=1), gain=2 ** 0.5),
            nn.ReLU(),
        )
        with torch.no_grad():
            conv_out = self.conv(torch.zeros(1, c, h, w)).flatten(1).shape[1]
        fc_in = conv_out + (num_tasks if task_conditioned else 0)
        self.fc = _ortho(nn.Linear(fc_in, hidden_size), gain=2 ** 0.5)

    def forward(self, obs: torch.Tensor, task_id: int) -> torch.Tensor:
        x = obs.float() / 255.0  # uint8 [0,255] -> [0,1]
        x = self.conv(x).flatten(start_dim=1)
        if self.task_conditioned:
            one_hot = torch.zeros(x.shape[0], self.num_tasks, device=x.device,
                                  dtype=x.dtype)
            one_hot[:, task_id] = 1.0
            x = torch.cat([x, one_hot], dim=-1)
        return torch.relu(self.fc(x))


class CkaRlCNNPolicy(Policy):
    """Discrete-action ATARI actor-critic with a CKA-RL FuseLinear actor head.

    Architecture: a shared Nature-CNN trunk -> a FuseLinear ACTOR head (logits
    over num_actions) + a plain critic head. The trunk and critic are ordinary
    trainable params; the actor head is the fused knowledge-adaptation head. Per
    the CKA-RL method, on each new task only the trunk/critic + a fresh short
    ``alpha`` (+ actor base) train, while the stored tau vectors stay frozen.
    """

    def __init__(
        self,
        obs_shape,
        num_actions: int,
        hidden_size: int = 512,
        num_tasks: int = 0,
        pool_size: int = 5,
        task_conditioned: bool = False,
    ) -> None:
        super().__init__()
        self.num_tasks = num_tasks
        self.pool_size = pool_size
        self.task_conditioned = task_conditioned
        if task_conditioned and num_tasks <= 0:
            raise ValueError("task_conditioned=True requires num_tasks > 0.")

        self.trunk = _NatureTrunk(obs_shape, hidden_size, num_tasks, task_conditioned)
        self.actor = FuseLinear(hidden_size, num_actions)   # fused head (ref FuseActor)
        self.critic = nn.Linear(hidden_size, 1)
        # Small actor init keeps the initial policy near-uniform (standard PPO init).
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        _ortho(self.critic, gain=1.0)
        # Fix the anchor theta_base = the near-uniform init (the paper's fixed base
        # that every tau delta is measured against). tau_j = theta_after_task_j -
        # theta_base, so task 1's residual is a real vector.
        self.actor.fix_base()

        self._tasks_added = 0
        # Per-task alpha vectors live here so they persist / are optimizable.
        self.alphas = nn.ParameterList()
        self.alpha_scales = nn.ParameterList()
        # Frozen per-task EFFECTIVE actor head (W_eff, b_eff), keyed by 0-based task.
        # CKA-RL evaluates each past task with ITS OWN learned combination over the
        # (shared, latest) trunk. Populated by store_eval_head(k) after task k trains.
        self._eval_heads: dict[int, tuple[torch.Tensor, torch.Tensor | None]] = {}

    # -- CKA-RL task lifecycle -------------------------------------------------

    def add_task(self, k: int) -> None:
        """Register task ``k`` (1-based) BEFORE training it. Mirrors the MLP version.

        For k == 1: no pool yet -> plain linear actor, everything trains.
        For k >= 2: fold the just-trained head (TAT), snapshot tau_new = theta -
        theta_base onto the pool, merge to pool_size, seed the head back to
        theta_base, and allocate a FRESH learnable alpha (len = pool size) +
        alpha_scale.
        """
        if k != self._tasks_added + 1:
            raise ValueError(f"add_task expects k={self._tasks_added + 1}, got {k}.")

        if k >= 2:
            # 1) Fold theta + alpha*tau into base (delta_theta_mode="TAT", ref).
            self.actor.fold_into_base()
            # 2) Push tau_new = theta - theta_base (fixed anchor) onto the pool.
            self.actor.snapshot_delta()
            # 3) Merge pool down to pool_size by cosine-sim averaging (ref merge_vectors).
            self.actor.merge_pool(self.pool_size)
            # 4) Seed the trainable head back to theta_base.
            self.actor.seed_from_base()
            # 5) Fresh learnable alpha (len = current pool) + scalar scale (ref
            #    setup_alpha Randn init: randn/num_vectors; alpha_scale=ones).
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
        (tau are buffers). Past-task alphas are frozen so only the CURRENT task's
        combination adapts (ref: one alpha per new task)."""
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

    @torch.no_grad()
    def store_eval_head(self, k: int) -> None:
        """Freeze task k's (1-based) EFFECTIVE actor head after it finishes training,
        so later evaluation of task k uses its own learned combination, not a later
        task's. Called once after training each task, BEFORE the next add_task
        folds/reseeds the head."""
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
        feats = self.trunk(obs, task_id)
        return Categorical(logits=self._actor_logits(feats, task_id))

    def value(self, obs: torch.Tensor, task_id: int) -> torch.Tensor:
        return self.critic(self.trunk(obs, task_id)).squeeze(-1)

    def dist_value(self, obs: torch.Tensor, task_id: int):
        feats = self.trunk(obs, task_id)
        return (Categorical(logits=self._actor_logits(feats, task_id)),
                self.critic(feats).squeeze(-1))
