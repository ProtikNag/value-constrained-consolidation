"""MLP categorical policies: shared-body and multi-head variants."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical

from crl.policies.base import Policy


def _mlp_trunk(input_dim: int, hidden_sizes: list[int]) -> tuple[nn.Sequential, int]:
    """Tanh MLP trunk; returns the module and its output width."""
    layers: list[nn.Module] = []
    last = input_dim
    for width in hidden_sizes:
        layers += [nn.Linear(last, width), nn.Tanh()]
        last = width
    return nn.Sequential(*layers), last


def _init_head(head: nn.Linear) -> None:
    """Small final layer keeps the initial policy near-uniform (low-variance
    early gradients)."""
    nn.init.orthogonal_(head.weight, gain=0.01)
    nn.init.zeros_(head.bias)


class MLPPolicy(Policy):
    """Single-head feed-forward categorical policy.

    When ``task_conditioned`` is true a one-hot task indicator is appended to
    every observation, letting one shared network represent task-specific
    behavior in shared states.
    """

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        hidden_sizes: list[int],
        task_conditioned: bool = False,
        num_tasks: int = 0,
    ) -> None:
        super().__init__()
        if task_conditioned and num_tasks <= 0:
            raise ValueError("task_conditioned=True requires num_tasks > 0.")
        self.task_conditioned = task_conditioned
        self.num_tasks = num_tasks

        input_dim = obs_dim + (num_tasks if task_conditioned else 0)
        self.trunk, last = _mlp_trunk(input_dim, hidden_sizes)
        self.head = nn.Linear(last, num_actions)
        _init_head(self.head)

    def _augment(self, obs: torch.Tensor, task_id: int) -> torch.Tensor:
        if not self.task_conditioned:
            return obs
        one_hot = torch.zeros(obs.shape[0], self.num_tasks, device=obs.device,
                              dtype=obs.dtype)
        one_hot[:, task_id] = 1.0
        return torch.cat([obs, one_hot], dim=-1)

    def dist(self, obs: torch.Tensor, task_id: int) -> Categorical:
        return Categorical(logits=self.head(self.trunk(self._augment(obs, task_id))))


class MultiHeadMLPPolicy(Policy):
    """Shared trunk with one output head per task (hard task routing).

    Each task owns a separate linear head on top of a shared Tanh trunk, so a
    task never has to share its output mapping with a conflicting task -- the
    task identifier selects the head. The trunk is shared (that is where
    forward transfer and, without the constraint, forgetting live); the
    min-max constraint is what protects the shared trunk. Optionally the task
    one-hot is also appended to the trunk input (``task_conditioned``), giving
    the trunk explicit task awareness on top of the routed heads.
    """

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        hidden_sizes: list[int],
        num_tasks: int,
        task_conditioned: bool = False,
    ) -> None:
        super().__init__()
        if num_tasks <= 0:
            raise ValueError("MultiHeadMLPPolicy requires num_tasks > 0.")
        self.num_tasks = num_tasks
        self.task_conditioned = task_conditioned

        input_dim = obs_dim + (num_tasks if task_conditioned else 0)
        self.trunk, last = _mlp_trunk(input_dim, hidden_sizes)
        self.heads = nn.ModuleList(
            [nn.Linear(last, num_actions) for _ in range(num_tasks)]
        )
        for head in self.heads:
            _init_head(head)

    def _augment(self, obs: torch.Tensor, task_id: int) -> torch.Tensor:
        if not self.task_conditioned:
            return obs
        one_hot = torch.zeros(obs.shape[0], self.num_tasks, device=obs.device,
                              dtype=obs.dtype)
        one_hot[:, task_id] = 1.0
        return torch.cat([obs, one_hot], dim=-1)

    def dist(self, obs: torch.Tensor, task_id: int) -> Categorical:
        if not 0 <= task_id < self.num_tasks:
            raise IndexError(f"task_id {task_id} out of range [0, {self.num_tasks})")
        features = self.trunk(self._augment(obs, task_id))
        return Categorical(logits=self.heads[task_id](features))


class MLPActorCriticPolicy(Policy):
    """MLP actor-critic with a SINGLE shared actor head + single critic head.

    The shared-head counterpart of :class:`MLPMultiHeadActorCriticPolicy`: there are
    NO per-task heads, so ALL weights (trunk + the one actor/critic head) are shared
    across tasks and that is where forgetting lives -- the hard setting where naive
    fine-tuning must overwrite shared parameters. On the PPO/actor-critic backend a
    single critic cannot value distinct tasks from a signal-less observation, so this
    policy is meant to be used with ``task_conditioned: true`` (task one-hot appended
    to the input) or a goal-in-obs family, which keeps the critic well-posed while
    still forcing all tasks to contend for one shared head."""

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        hidden_sizes: list[int],
        num_tasks: int = 0,
        task_conditioned: bool = False,
    ) -> None:
        super().__init__()
        if task_conditioned and num_tasks <= 0:
            raise ValueError("task_conditioned=True requires num_tasks > 0.")
        self.task_conditioned = task_conditioned
        self.num_tasks = num_tasks
        input_dim = obs_dim + (num_tasks if task_conditioned else 0)
        self.trunk, last = _mlp_trunk(input_dim, hidden_sizes)
        self.actor = nn.Linear(last, num_actions)
        self.critic = nn.Linear(last, 1)
        _init_head(self.actor)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

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


class MLPMultiHeadActorCriticPolicy(Policy):
    """MLP actor-critic with a shared trunk + per-task actor AND critic heads.

    The MLP analogue of :class:`crl.policies.cnn_ac.AtariMultiHeadActorCriticPolicy`
    for the PPO backend on flat-vector families (gridworld). A narrow shared Tanh
    trunk is where cross-task interference / forgetting lives (protected by the
    min-max constraint); ``task_id`` routes to that task's own actor and critic
    head. Only the ACTOR is ever constrained (global phase); the critic is standard
    PPO value regression and feeds GAE advantages only -- the constraint value
    V_i stays Monte-Carlo (paper-faithful), exactly as on Atari.
    """

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        hidden_sizes: list[int],
        num_tasks: int,
        task_conditioned: bool = False,
    ) -> None:
        super().__init__()
        if num_tasks <= 0:
            raise ValueError("MLPMultiHeadActorCriticPolicy requires num_tasks > 0.")
        self.num_tasks = num_tasks
        self.task_conditioned = task_conditioned
        input_dim = obs_dim + (num_tasks if task_conditioned else 0)
        self.trunk, last = _mlp_trunk(input_dim, hidden_sizes)
        self.actors = nn.ModuleList(
            [nn.Linear(last, num_actions) for _ in range(num_tasks)])
        self.critics = nn.ModuleList(
            [nn.Linear(last, 1) for _ in range(num_tasks)])
        for actor in self.actors:
            _init_head(actor)               # small init -> near-uniform start
        for critic in self.critics:
            nn.init.orthogonal_(critic.weight, gain=1.0)
            nn.init.zeros_(critic.bias)

    def _augment(self, obs: torch.Tensor, task_id: int) -> torch.Tensor:
        if not self.task_conditioned:
            return obs
        one_hot = torch.zeros(obs.shape[0], self.num_tasks, device=obs.device,
                              dtype=obs.dtype)
        one_hot[:, task_id] = 1.0
        return torch.cat([obs, one_hot], dim=-1)

    def _check(self, task_id: int) -> None:
        if not 0 <= task_id < self.num_tasks:
            raise IndexError(f"task_id {task_id} out of range [0, {self.num_tasks})")

    def dist(self, obs: torch.Tensor, task_id: int) -> Categorical:
        self._check(task_id)
        return Categorical(logits=self.actors[task_id](self.trunk(self._augment(obs, task_id))))

    def value(self, obs: torch.Tensor, task_id: int) -> torch.Tensor:
        self._check(task_id)
        return self.critics[task_id](self.trunk(self._augment(obs, task_id))).squeeze(-1)

    def dist_value(self, obs: torch.Tensor, task_id: int):
        self._check(task_id)
        feats = self.trunk(self._augment(obs, task_id))
        return (Categorical(logits=self.actors[task_id](feats)),
                self.critics[task_id](feats).squeeze(-1))


class MLPSeparateActorCriticPolicy(Policy):
    """Actor and critic as two INDEPENDENT networks. Nothing is shared.

    :class:`MLPActorCriticPolicy` puts one Tanh trunk under both the policy and
    the value function, so the critic's regression loss reshapes the very
    features the actor reads from. That coupling is unwanted here for two
    reasons. The constraint coefficient gates only the policy-gradient term, so
    with a shared trunk the current task keeps influencing the actor's features
    through the value loss no matter what ``mu`` is. And DUEL hands a frozen
    specialist's critic to a *different* policy to score its rollouts, which is
    only clean when that critic carries no actor parameters.

    Each network keeps its own trunk of the same width, so the ACTOR's capacity
    is identical to the shared-trunk version (the extra parameters live entirely
    in the critic, which is discarded at evaluation time). Like its shared-trunk
    counterpart this is a SINGLE actor head and a SINGLE critic head serving all
    tasks, so pair it with ``task_conditioned: true``.
    """

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        hidden_sizes: list[int],
        num_tasks: int = 0,
        task_conditioned: bool = False,
    ) -> None:
        super().__init__()
        if task_conditioned and num_tasks <= 0:
            raise ValueError("task_conditioned=True requires num_tasks > 0.")
        self.task_conditioned = task_conditioned
        self.num_tasks = num_tasks
        input_dim = obs_dim + (num_tasks if task_conditioned else 0)
        # Two trunks, built independently -> disjoint parameter sets.
        self.actor_trunk, actor_last = _mlp_trunk(input_dim, hidden_sizes)
        self.critic_trunk, critic_last = _mlp_trunk(input_dim, hidden_sizes)
        self.actor = nn.Linear(actor_last, num_actions)
        self.critic = nn.Linear(critic_last, 1)
        _init_head(self.actor)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def _augment(self, obs: torch.Tensor, task_id: int) -> torch.Tensor:
        if not self.task_conditioned:
            return obs
        one_hot = torch.zeros(obs.shape[0], self.num_tasks, device=obs.device,
                              dtype=obs.dtype)
        one_hot[:, task_id] = 1.0
        return torch.cat([obs, one_hot], dim=-1)

    def dist(self, obs: torch.Tensor, task_id: int) -> Categorical:
        return Categorical(logits=self.actor(self.actor_trunk(self._augment(obs, task_id))))

    def value(self, obs: torch.Tensor, task_id: int) -> torch.Tensor:
        return self.critic(self.critic_trunk(self._augment(obs, task_id))).squeeze(-1)

    def dist_value(self, obs: torch.Tensor, task_id: int):
        # No shared features to reuse: this is two independent forward passes.
        aug = self._augment(obs, task_id)
        return (Categorical(logits=self.actor(self.actor_trunk(aug))),
                self.critic(self.critic_trunk(aug)).squeeze(-1))

    def actor_parameters(self):
        """Parameters the policy gradient (and the constraint) may touch."""
        yield from self.actor_trunk.parameters()
        yield from self.actor.parameters()

    def critic_parameters(self):
        """Parameters only the value regression may touch."""
        yield from self.critic_trunk.parameters()
        yield from self.critic.parameters()


class MLPSharedActorMultiCriticPolicy(Policy):
    """ONE shared actor head for every task, plus an INDEPENDENT critic per task.

    The actor is a single trunk and a single head serving all tasks, which is the
    architecture the headline claim rests on: no per-task parameters in the thing
    that acts, so every task contends for the same capacity and forgetting is
    forced rather than designed away.

    The critic is the opposite. Each task gets its own trunk and head, sharing
    nothing with the actor or with any other task's critic. One critic asked to
    hold fifty value functions, distinguished only by a one-hot, is both
    overloaded (the tasks interfere inside it) and starved (each is visited
    rarely). Splitting it removes the interference entirely.

    This costs nothing in fairness. The critic never selects an action and is
    discarded at evaluation, so per-task critics are training scaffolding rather
    than parameter isolation in the deployed policy. Memory grows with the
    sequence, which it already does since a specialist is stored per task.

    Safe as a variance-reduction device specifically because a state-dependent
    baseline cannot bias a policy gradient however wrong it is. Use with
    ``gae_lambda=1`` and whole episodes, where the critic is a baseline only. As
    a BOOTSTRAP a wrong critic is catastrophic, measured at PERF -0.23.
    """

    def __init__(
        self,
        obs_dim: int,
        num_actions: int,
        hidden_sizes: list[int],
        num_tasks: int,
        task_conditioned: bool = False,
    ) -> None:
        super().__init__()
        if num_tasks <= 0:
            raise ValueError("MLPSharedActorMultiCriticPolicy requires num_tasks > 0.")
        self.num_tasks = num_tasks
        self.task_conditioned = task_conditioned
        input_dim = obs_dim + (num_tasks if task_conditioned else 0)
        self.actor_trunk, actor_last = _mlp_trunk(input_dim, hidden_sizes)
        self.actor = nn.Linear(actor_last, num_actions)
        _init_head(self.actor)
        # One independent critic per task; nothing shared between them.
        self.critic_trunks = nn.ModuleList()
        self.critics = nn.ModuleList()
        for _ in range(num_tasks):
            trunk, last = _mlp_trunk(input_dim, hidden_sizes)
            head = nn.Linear(last, 1)
            nn.init.orthogonal_(head.weight, gain=1.0)
            nn.init.zeros_(head.bias)
            self.critic_trunks.append(trunk)
            self.critics.append(head)

    def _augment(self, obs: torch.Tensor, task_id: int) -> torch.Tensor:
        if not self.task_conditioned:
            return obs
        one_hot = torch.zeros(obs.shape[0], self.num_tasks, device=obs.device,
                              dtype=obs.dtype)
        one_hot[:, task_id] = 1.0
        return torch.cat([obs, one_hot], dim=-1)

    def _check(self, task_id: int) -> None:
        if not 0 <= task_id < self.num_tasks:
            raise IndexError(f"task_id {task_id} out of range [0, {self.num_tasks})")

    def dist(self, obs: torch.Tensor, task_id: int) -> Categorical:
        return Categorical(logits=self.actor(self.actor_trunk(self._augment(obs, task_id))))

    def value(self, obs: torch.Tensor, task_id: int) -> torch.Tensor:
        self._check(task_id)
        aug = self._augment(obs, task_id)
        return self.critics[task_id](self.critic_trunks[task_id](aug)).squeeze(-1)

    def dist_value(self, obs: torch.Tensor, task_id: int):
        self._check(task_id)
        aug = self._augment(obs, task_id)
        return (Categorical(logits=self.actor(self.actor_trunk(aug))),
                self.critics[task_id](self.critic_trunks[task_id](aug)).squeeze(-1))

    def actor_parameters(self):
        """The only parameters that act, and the only ones the constraint governs."""
        yield from self.actor_trunk.parameters()
        yield from self.actor.parameters()

    def critic_parameters(self):
        yield from self.critic_trunks.parameters()
        yield from self.critics.parameters()
