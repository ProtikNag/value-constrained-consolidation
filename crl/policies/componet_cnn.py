"""CompoNet for Atari (image obs): CNN-encoder self-composing policies.

CNN analogue of :class:`crl.policies.componet.CompoNetPolicy` (the MLP version,
already code-verified). The dual-attention *composition* math is IDENTICAL to the
MLP version — copied verbatim — and only the per-module state encoder is swapped
from a Tanh MLP to a Nature-DQN CNN trunk (Mnih et al. 2015; same trunk shape as
``crl.policies.cnn_ac._NatureTrunk``). uint8 frames [B,4,84,84] are normalized
/255 inside the encoder.

Idea (Malagon et al., ICML 2024, Sec 4): task k adds a NEW trainable self-composing
module. Its actor output is composed from (a) the frozen logit outputs of all k-1
previous modules (Phi) and (b) its own internal policy, via two attention heads.
All previous modules/critics are FROZEN, so no forgetting by construction; params
grow linearly. Separate per-task CNN encoders + heads are the intended CompoNet
growth regime.

Paper eq -> code map (Sec 4.1-4.2; mirrors the MLP CompoNet):
  * Phi_{k;s}          -> `phi`: stacked frozen predecessor logits, [B, k-1, |A|]
  * Output att. head   -> `_head_out`: q=h_s Wq, K=(Phi+E_out) Wk, V=Phi;
                          v = softmax(qK^T/sqrt(d)) V   (tentative output, eq 299)
  * Input att. head    -> `_internal`: q=h_s Wq, K=([Phi;v]+E_in) Wk,
                          V=([Phi;v]) Wv; attends over Phi AND the output-head v
  * Internal policy    -> feed-forward MLP on hstack([att_out, h_s]) -> logits
  * Final module output-> `out = v + internal_policy_out`  (v adjusted, not
                          overwritten)
The FIRST module has no predecessors: encoder -> internal-policy MLP -> logits
(the paper's "single policy module from scratch", = FirstModuleWrapper).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from crl.policies.base import Policy
from crl.policies.cnn_ac import _ortho


def _pos_encoding(seq_len: int, d: int, n: int = 10_000) -> torch.Tensor:
    """Cosine positional encoding of Vaswani et al. (paper eq: E_out/E_in),
    matching reference `get_position_encoding`. Returns [seq_len, d]."""
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in range(d // 2):
            denom = np.power(n, 2 * i / d)
            P[k, 2 * i] = np.sin(k / denom)
            if 2 * i + 1 < d:
                P[k, 2 * i + 1] = np.cos(k / denom)
    return torch.tensor(P, dtype=torch.float32)


class _CnnEncoder(nn.Module):
    """Nature-DQN CNN body -> ``[B, hidden]`` features. uint8 -> /255 inside.

    Same trunk shape as ``crl.policies.cnn_ac._NatureTrunk`` but with no task
    one-hot (CompoNet routes by module identity, so conditioning is redundant).
    """

    def __init__(self, obs_shape: tuple[int, int, int], hidden: int) -> None:
        super().__init__()
        c, h, w = obs_shape
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
        self.fc = _ortho(nn.Linear(conv_out, hidden), gain=2 ** 0.5)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs.float() / 255.0  # uint8 [0,255] -> [0,1]
        x = self.conv(x).flatten(start_dim=1)
        return torch.relu(self.fc(x))


class _Module(nn.Module):
    """One self-composing policy module (task k), CNN encoder + attention.

    ``n_prev`` = number of frozen predecessor modules whose logit outputs feed the
    attention heads. When ``n_prev == 0`` this is the first module: encoder ->
    internal MLP -> logits, with no composition (paper: train from scratch).
    """

    def __init__(self, obs_shape: tuple[int, int, int], num_actions: int,
                 hidden: int, n_prev: int) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.hidden = hidden
        self.n_prev = n_prev
        self.att_temp = math.sqrt(hidden)  # scaled dot-product temperature (eq 299)

        # Per-module CNN state encoder h_s (swap vs MLP CompoNet: CNN not Tanh MLP).
        self.encoder = _CnnEncoder(obs_shape, hidden)

        if n_prev == 0:
            # First module: plain policy MLP on the encoded state -> logits.
            self.internal_policy = nn.Sequential(
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, num_actions),
            )
            return

        # --- Output attention head (proposes tentative logits v) ---
        self.headout_wq = nn.Linear(hidden, hidden)        # q = h_s Wq
        self.headout_wk = nn.Linear(num_actions, hidden)   # K = (Phi + E_out) Wk
        # values matrix = Phi itself, no linear transform (paper).

        # --- Input attention head (retrieves info from [Phi; v]) ---
        self.headin_wq = nn.Linear(hidden, hidden)
        self.headin_wk = nn.Linear(num_actions, hidden)
        self.headin_wv = nn.Linear(num_actions, hidden)

        # --- Internal policy: MLP on hstack([att_in_out, h_s]) -> logits ---
        self.internal_policy = nn.Sequential(
            nn.Linear(hidden + hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, num_actions),
        )

        # Positional encodings (fixed buffers; not learned).
        # E_out over the k-1 previous rows of Phi; E_in over the k rows [Phi; v].
        self.register_buffer("pe_out", _pos_encoding(n_prev, num_actions)[None])
        self.register_buffer("pe_in", _pos_encoding(n_prev + 1, num_actions)[None])

    def _head_out(self, hs: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """Output attention head -> tentative logits v, [B, |A|]. (paper eq 299)
        phi: [B, n_prev, |A|]."""
        query = self.headout_wq(hs)                       # [B, hidden]
        keys = self.headout_wk(phi + self.pe_out)         # [B, n_prev, hidden]
        w = torch.matmul(query[:, None, :], keys.permute(0, 2, 1))  # [B,1,n_prev]
        att = F.softmax(w / self.att_temp, dim=-1)
        v = torch.matmul(att, phi)                        # [B, 1, |A|], V = phi
        return v[:, 0, :]

    def _internal(self, hs: torch.Tensor, phi_v: torch.Tensor) -> torch.Tensor:
        """Input attention head + internal policy -> logits, [B, |A|].
        phi_v = [Phi; v] : [B, n_prev+1, |A|]."""
        query = self.headin_wq(hs)
        keys = self.headin_wk(phi_v + self.pe_in)
        values = self.headin_wv(phi_v)
        w = torch.matmul(query[:, None, :], keys.permute(0, 2, 1))  # [B,1,n_prev+1]
        att = F.softmax(w / self.att_temp, dim=-1)
        att_out = torch.matmul(att, values)[:, 0, :]      # [B, hidden]
        return self.internal_policy(torch.hstack([att_out, hs]))

    def forward(self, obs: torch.Tensor, phi: torch.Tensor | None) -> torch.Tensor:
        """Return this module's action logits, [B, |A|]. ``phi`` is the stacked
        previous-module logits [B, n_prev, |A|] (None/ignored for first module)."""
        hs = self.encoder(obs)
        if self.n_prev == 0:
            return self.internal_policy(hs)
        v = self._head_out(hs, phi)                        # tentative output
        phi_v = torch.cat([phi, v[:, None, :]], dim=1)     # [Phi; v]
        internal = self._internal(hs, phi_v)
        return v + internal                                # adjust v


class CompoNetCNNPolicy(Policy):
    """Self-composing continual Atari policy: one frozen CNN module per past task +
    one trainable module for the current task, composed via attention.

    Orchestrator contract (identical to the MLP :class:`CompoNetPolicy`):
      * ``add_task(k)`` (k 1-based) BEFORE training task k: builds module k wired to
        the k-1 frozen predecessors and freezes everything else. Only module k's
        params (+ its critic) require grad.
      * ``dist(obs, task_id)`` / ``value`` / ``dist_value`` (task_id 0-based) route
        through modules 0..task_id, feeding each predecessor's logits into module
        task_id's attention composition.
    """

    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        num_actions: int,
        hidden_size: int = 512,
        num_tasks: int = 0,
        task_conditioned: bool = False,
    ) -> None:
        super().__init__()
        if num_tasks <= 0:
            raise ValueError("CompoNetCNNPolicy requires num_tasks > 0.")
        # CompoNet routes by module identity, so a task one-hot is redundant; we
        # accept the flag for a uniform ctor signature but do not use it.
        self.task_conditioned = task_conditioned
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.num_tasks = num_tasks
        self.hidden = int(hidden_size)          # model dim d_model = encoder out dim
        self.modules_ = nn.ModuleList()         # policy modules, index = task_id
        self.critics = nn.ModuleList()          # per-task critic heads
        self._added = 0                         # number of tasks instantiated so far

    def add_task(self, k: int) -> None:
        """Instantiate task ``k`` (1-based) and freeze all previous modules.

        After this call only module k-1 and critic k-1 are trainable; every
        earlier module/critic has ``requires_grad=False`` and is in eval() mode
        (paper: freeze the trained module, add a new trainable one)."""
        if not 1 <= k <= self.num_tasks:
            raise ValueError(f"add_task: k={k} out of range [1, {self.num_tasks}]")
        if k != self._added + 1:
            raise ValueError(f"add_task must be called in order; expected "
                             f"{self._added + 1}, got {k}.")
        n_prev = k - 1
        device = next(self.parameters()).device if list(self.parameters()) else None
        module = _Module(self.obs_shape, self.num_actions, self.hidden, n_prev)
        # Critic: own CNN encoder + scalar head (per-task, standard for CompoNet).
        critic = nn.Sequential(
            _CnnEncoder(self.obs_shape, self.hidden),
            _ortho(nn.Linear(self.hidden, 1), gain=1.0),
        )
        if device is not None:
            module = module.to(device)
            critic = critic.to(device)
        self.modules_.append(module)
        self.critics.append(critic)
        self._added = k

        # Freeze everything, then re-enable ONLY the current module + its critic.
        for p in self.parameters():
            p.requires_grad_(False)
        for m in list(self.modules_)[:-1]:
            m.eval()
        for c in list(self.critics)[:-1]:
            c.eval()
        for p in module.parameters():
            p.requires_grad_(True)
        for p in critic.parameters():
            p.requires_grad_(True)

    def _phi(self, obs: torch.Tensor, task_id: int) -> torch.Tensor | None:
        """Stack the frozen predecessor modules' logits: [B, task_id, |A|].
        Predecessors run under no_grad (they are frozen references)."""
        if task_id == 0:
            return None
        outs = []
        with torch.no_grad():
            phi = None
            for j in range(task_id):
                out_j = self.modules_[j](obs, phi)          # [B, |A|]
                outs.append(out_j)
                phi = torch.stack(outs, dim=1)              # [B, j+1, |A|]
        return phi

    def _check(self, task_id: int) -> None:
        if not 0 <= task_id < self._added:
            raise IndexError(f"task_id {task_id} not instantiated "
                            f"(added={self._added}); call add_task first.")

    def dist(self, obs: torch.Tensor, task_id: int) -> Categorical:
        self._check(task_id)
        logits = self.modules_[task_id](obs, self._phi(obs, task_id))
        return Categorical(logits=logits)

    def value(self, obs: torch.Tensor, task_id: int) -> torch.Tensor:
        self._check(task_id)
        return self.critics[task_id](obs).squeeze(-1)

    def dist_value(self, obs: torch.Tensor, task_id: int):
        self._check(task_id)
        logits = self.modules_[task_id](obs, self._phi(obs, task_id))
        return Categorical(logits=logits), self.critics[task_id](obs).squeeze(-1)
