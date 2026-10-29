"""Continual-Backprop (CBP) shared-head MLP actor-critic.

Faithful port of Continual Backprop / generate-and-test (Dohare et al. 2024,
"Loss of Plasticity in Deep Continual Learning") onto this
:class:`~crl.policies.mlp.MLPActorCriticPolicy` interface, for the GridWorld
fixed-capacity (no-growth) comparison.

Same interface as ``MLPActorCriticPolicy``: a SINGLE shared trunk feeding one
actor head and one critic head, optionally ``task_conditioned`` (task one-hot
appended to obs). The only difference is that the trunk's ``nn.Linear`` layers
are CBP-managed: each hidden unit tracks a running *utility*, ages, and once a
unit is *mature* (age > maturity_threshold) low-utility units are periodically
REINITIALIZED (generate-and-test) to preserve plasticity under naive sequential
(``method=finetune``) training.

Reference impl mirrored:
  their reference implementation  (class GnT)
                                                /cbpnet.py       (class CbpAgent)
                                          /run_sac.py:551-553    (call site)

WHEN THE RESET FIRES (important):
  In the reference, ``GnT.gen_and_test`` is called AFTER ``optimizer.step()``
  each update, consuming the activations captured on the last forward. It is NOT
  self-contained inside forward (it needs the post-step weights and the
  optimizer state). We therefore mirror that exactly and expose
  :meth:`cbp_step`, which the trainer must call ONCE after every
  ``optimizer.step()`` (i.e. after ``optimize_batches``). A forward hook captures
  the hidden activations transparently, so the trainer only needs to add that one
  ``policy.cbp_step(optimizer)`` line.
"""

from __future__ import annotations

from math import sqrt

import torch
import torch.nn as nn
from torch.distributions import Categorical

from crl.policies.base import Policy


def _init_head(head: nn.Linear) -> None:
    nn.init.orthogonal_(head.weight, gain=0.01)
    nn.init.zeros_(head.bias)


class _CbpTrunk(nn.Module):
    """ReLU MLP trunk whose Linear layers self-manage per-unit utility + resets.

    Mirrors ``GnT`` from the reference for a stack of hidden layers. The trunk is
    ``[Linear, ReLU, Linear, ReLU, ...]``; the "next layer" of the LAST hidden
    layer is the shared actor+critic heads (passed in via :meth:`bind_heads`), so
    the outgoing-weight-magnitude term in the utility, the bias correction on
    reset, and the outgoing-weight zeroing all act on BOTH heads -- the
    generalization of the reference's single-``next_layer`` logic to a two-head
    shared trunk.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: list[int],
        replacement_rate: float,
        maturity_threshold: int,
        decay_rate: float,
        util_type: str,
    ) -> None:
        super().__init__()
        if not hidden_sizes:
            raise ValueError("_CbpTrunk requires at least one hidden layer.")
        self.replacement_rate = replacement_rate
        self.maturity_threshold = maturity_threshold
        self.decay_rate = decay_rate
        self.util_type = util_type

        layers: list[nn.Module] = []
        last = input_dim
        for width in hidden_sizes:
            layers += [nn.Linear(last, width), nn.ReLU()]
            last = width
        self.net = nn.Sequential(*layers)
        self.out_dim = last
        self.n_hidden = len(hidden_sizes)

        # Kaiming-relu reinit bound per hidden layer (GnT.compute_bounds, default).
        gain = nn.init.calculate_gain("relu")
        self.bounds = [gain * sqrt(3.0 / self.net[i * 2].in_features)
                       for i in range(self.n_hidden)]

        # Per-unit CBP state (buffers so they move with .to(device) / state_dict).
        for i in range(self.n_hidden):
            w = self.net[i * 2].out_features
            self.register_buffer(f"util_{i}", torch.zeros(w))
            self.register_buffer(f"bcu_{i}", torch.zeros(w))      # bias-corrected util
            self.register_buffer(f"ages_{i}", torch.zeros(w))
            self.register_buffer(f"mfa_{i}", torch.zeros(w))      # mean feature act

        # Activation capture (forward hook on each ReLU), mirrors cbpshared.
        self._acts: dict[int, torch.Tensor] = {}
        for i in range(self.n_hidden):
            relu = self.net[i * 2 + 1]
            relu.register_forward_hook(self._make_hook(i))

        self._heads: list[nn.Linear] = []
        self.total_resets = 0
        self.last_reset_specs: list = []  # (param, index) slices from last cbp_step

    def _make_hook(self, i: int):
        def hook(_m, _inp, out):
            self._acts[i] = out
        return hook

    def bind_heads(self, *heads: nn.Linear) -> None:
        """Register the shared heads that read the last hidden layer (the
        'next layer' for utility/reset accounting)."""
        self._heads = list(heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _next_out_mag(self, i: int) -> torch.Tensor:
        """Mean |outgoing weight| per unit of hidden layer i, over its
        consumers: the next hidden Linear, or the bound heads for the last."""
        if i < self.n_hidden - 1:
            nxt = self.net[(i + 1) * 2]
            return nxt.weight.data.abs().mean(dim=0)
        # last hidden layer -> heads (concatenate consumers, mean over all rows)
        cols = torch.cat([h.weight.data for h in self._heads], dim=0)
        return cols.abs().mean(dim=0)

    @torch.no_grad()
    def cbp_step(self) -> int:
        """Generate-and-test over the activations from the last forward.

        Call ONCE after each ``optimizer.step()``. Returns number of units reset
        this call and records, in ``self.last_reset_specs``, the (param, index)
        slices whose optimizer moments should be zeroed. Mirrors
        GnT.test_features + gen_new_features."""
        self.last_reset_specs = []
        if self.replacement_rate == 0 or not self._acts:
            return 0
        n_reset_total = 0
        for i in range(self.n_hidden):
            feats = self._acts.get(i)
            if feats is None:
                continue
            util = getattr(self, f"util_{i}")
            bcu = getattr(self, f"bcu_{i}")
            ages = getattr(self, f"ages_{i}")
            mfa = getattr(self, f"mfa_{i}")

            ages += 1
            # --- update_utility (GnT.update_utility) ---
            util *= self.decay_rate
            bias_correction = 1 - self.decay_rate ** ages
            mfa *= self.decay_rate
            mfa += (1 - self.decay_rate) * feats.mean(dim=0)
            bias_corrected_act = mfa / bias_correction

            out_mag = self._next_out_mag(i)
            in_mag = self.net[i * 2].weight.data.abs().mean(dim=1)
            if self.util_type == "weight":
                new_util = out_mag
            elif self.util_type == "contribution":
                new_util = out_mag * feats.abs().mean(dim=0)
            elif self.util_type == "adaptation":
                new_util = 1.0 / in_mag
            elif self.util_type == "zero_contribution":
                new_util = out_mag * (feats - bias_corrected_act).abs().mean(dim=0)
            elif self.util_type == "adaptable_contribution":
                new_util = out_mag * (feats - bias_corrected_act).abs().mean(dim=0) / in_mag
            else:
                raise ValueError(f"unknown util_type {self.util_type!r}")
            util += (1 - self.decay_rate) * new_util
            bcu[:] = util / bias_correction

            # --- test_features: pick mature, lowest-utility units to replace ---
            eligible = torch.where(ages > self.maturity_threshold)[0]
            if eligible.numel() == 0:
                continue
            n_new = self.replacement_rate * eligible.numel()
            if n_new < 1:
                n_new = 1 if torch.rand(1).item() <= n_new else 0
            n_new = int(n_new)
            if n_new == 0:
                continue
            worst = torch.topk(-bcu[eligible], n_new)[1]
            to_reset = eligible[worst]

            util[to_reset] = 0
            mfa[to_reset] = 0.0

            # --- gen_new_features: reinit in-weights, zero out-weights, fix bias ---
            cur = self.net[i * 2]
            cur.weight.data[to_reset, :] = 0.0
            cur.weight.data[to_reset, :] += torch.empty(
                n_new, cur.in_features, device=cur.weight.device
            ).uniform_(-self.bounds[i], self.bounds[i])
            cur.bias.data[to_reset] = 0.0

            bias_add = (mfa[to_reset] / (1 - self.decay_rate ** ages[to_reset]))
            for h in self._heads if i == self.n_hidden - 1 else [self.net[(i + 1) * 2]]:
                h.bias.data += (h.weight.data[:, to_reset] * bias_add).sum(dim=1)
                h.weight.data[:, to_reset] = 0.0
            ages[to_reset] = 0

            # record optimizer-moment slices to zero (in-weights/bias rows,
            # out-weight cols of every consumer)
            self.last_reset_specs.append((cur.weight, (to_reset, slice(None))))
            self.last_reset_specs.append((cur.bias, (to_reset,)))
            consumers = self._heads if i == self.n_hidden - 1 else [self.net[(i + 1) * 2]]
            for h in consumers:
                self.last_reset_specs.append((h.weight, (slice(None), to_reset)))

            n_reset_total += n_new

        self.total_resets += n_reset_total
        return n_reset_total


class MLPActorCriticCbpPolicy(Policy):
    """CBP-managed shared-trunk MLP actor-critic (single actor + single critic).

    Drop-in for :class:`crl.policies.mlp.MLPActorCriticPolicy` with the same
    ``dist``/``value``/``dist_value(obs, task_id)`` interface. Trained with
    ``method=finetune``; the trainer must call :meth:`cbp_step` after each
    optimizer step for the generate-and-test resets to fire.
    """

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        hidden_sizes: list[int],
        num_tasks: int = 0,
        task_conditioned: bool = False,
        # CBP hyper-params (reference GnT defaults; replacement_rate=1e-3 per
        # run_sac.py:371, the value actually used for cbpnet).
        replacement_rate: float = 1e-3,
        maturity_threshold: int = 100,
        decay_rate: float = 0.99,
        util_type: str = "contribution",
    ) -> None:
        super().__init__()
        if task_conditioned and num_tasks <= 0:
            raise ValueError("task_conditioned=True requires num_tasks > 0.")
        self.task_conditioned = task_conditioned
        self.num_tasks = num_tasks
        input_dim = obs_dim + (num_tasks if task_conditioned else 0)

        self.trunk = _CbpTrunk(
            input_dim, hidden_sizes, replacement_rate, maturity_threshold,
            decay_rate, util_type,
        )
        last = self.trunk.out_dim
        self.actor = nn.Linear(last, num_actions)
        self.critic = nn.Linear(last, 1)
        _init_head(self.actor)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)
        self.trunk.bind_heads(self.actor, self.critic)

    def _augment(self, obs: torch.Tensor, task_id: int) -> torch.Tensor:
        if not self.task_conditioned:
            return obs
        one_hot = torch.zeros(obs.shape[0], self.num_tasks, device=obs.device,
                              dtype=obs.dtype)
        one_hot[:, task_id] = 1.0
        return torch.cat([obs, one_hot], dim=-1)

    def dist(self, obs: torch.Tensor, task_id: int) -> Categorical:
        return Categorical(logits=self.actor(self.trunk(self._augment(obs, task_id))))

    def value(self, obs: torch.Tensor, task_id: int) -> torch.Tensor:
        return self.critic(self.trunk(self._augment(obs, task_id))).squeeze(-1)

    def dist_value(self, obs: torch.Tensor, task_id: int):
        feats = self.trunk(self._augment(obs, task_id))
        return Categorical(logits=self.actor(feats)), self.critic(feats).squeeze(-1)

    @torch.no_grad()
    def cbp_step(self, optimizer: torch.optim.Optimizer | None = None) -> int:
        """Run generate-and-test on the trunk. Call once after each
        ``optimizer.step()`` (i.e. after ``optimize_batches``). Returns the number
        of units reset this call. Zeroes the Adam moment tensors of reset units'
        weights when ``optimizer`` is provided (reference AdamGnT behavior; stock
        Adam's scalar step is left as-is)."""
        n = self.trunk.cbp_step()
        if optimizer is not None:
            state = optimizer.state
            for param, idx in self.trunk.last_reset_specs:
                st = state.get(param)
                if not st:
                    continue
                for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    if key in st:
                        st[key][idx] = 0.0
        return n


if __name__ == "__main__":
    # Self-check: shapes, dist_value, and that a reset actually fires.
    torch.manual_seed(0)
    obs_dim, n_act, n_tasks = 6, 4, 3
    pol = MLPActorCriticCbpPolicy(
        obs_dim, n_act, [16, 16], num_tasks=n_tasks, task_conditioned=True,
        replacement_rate=0.5, maturity_threshold=2,  # aggressive: force resets
    )
    opt = torch.optim.Adam(pol.parameters(), lr=1e-3, eps=1e-5)
    resets = 0
    for step in range(10):
        obs = torch.randn(8, obs_dim)
        dist, val = pol.dist_value(obs, task_id=step % n_tasks)
        assert dist.logits.shape == (8, n_act), dist.logits.shape
        assert val.shape == (8,), val.shape
        loss = -dist.log_prob(dist.sample()).mean() + val.pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        resets += pol.cbp_step(opt)
    print(f"total resets over 10 steps: {resets} (trunk.total_resets={pol.trunk.total_resets})")
    assert resets > 0, "no CBP reset fired -- lower maturity_threshold / raise rate"
    # value-only path
    assert pol.value(torch.randn(5, obs_dim), 0).shape == (5,)
    print("self-check OK")
