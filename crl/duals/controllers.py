"""Dual-variable controllers for the Lagrangian saddle problems.

Both saddle problems (eqs 10, 14) need a nonnegative multiplier driven by
the constraint violation. ``ProjectedAscentDual`` is the projected-ascent
rule from the README; ``PIDDual`` is the drop-in variant of Stooke et al.
(ICML 2020) for when plain ascent oscillates.

Two-timescale discipline (dual lr << primal lr) is a config responsibility;
nothing here enforces it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from crl.config import DualConfig


class DualController(ABC):
    """A scalar multiplier in [0, max_value] updated from F-hat - eps."""

    def __init__(self, init: float, max_value: float, warm_start: bool) -> None:
        self._init = init
        self._max_value = max_value
        self._warm_start = warm_start
        self._value = init

    @property
    def value(self) -> float:
        return self._value

    def reset(self) -> None:
        """Called at each phase start; warm-started controllers keep state."""
        if not self._warm_start:
            self._value = self._init
            self._reset_state()

    def _reset_state(self) -> None:
        """Clear controller-specific internals (overridden where needed)."""

    @abstractmethod
    def update(self, constraint_value: float, eps: float) -> float:
        """Consume F-hat and return the new multiplier."""


class ProjectedAscentDual(DualController):
    """lambda <- clip(lambda + lr * (F-hat - eps), 0, max)  (README eq block)."""

    def __init__(self, cfg: DualConfig) -> None:
        super().__init__(cfg.init, cfg.max_value, cfg.warm_start)
        self._lr = cfg.lr
        self._decay_when_stuck = cfg.decay_when_stuck
        self._stuck_patience = cfg.stuck_patience
        self._stuck_decay = cfg.stuck_decay
        self._stuck_frac = cfg.stuck_frac
        self._stuck = 0
        self._prev_c: float | None = None

    def _reset_state(self) -> None:
        self._stuck = 0
        self._prev_c = None

    def update(self, constraint_value: float, eps: float) -> float:
        self._value = min(
            max(self._value + self._lr * (constraint_value - eps), 0.0),
            self._max_value,
        )
        if self._decay_when_stuck and self._max_value > 0:
            pinned = self._value >= self._stuck_frac * self._max_value
            # "not improving" = the squared gap did not shrink since last refresh.
            improving = (self._prev_c is not None
                         and constraint_value < self._prev_c - 1e-6)
            self._stuck = self._stuck + 1 if (pinned and not improving) else 0
            if self._stuck >= self._stuck_patience:
                self._value *= self._stuck_decay  # release an unlearnable task
                self._stuck = 0
            self._prev_c = constraint_value
        return self._value


class PIDDual(DualController):
    """PID Lagrangian controller (Stooke et al. 2020).

    lambda = [Kp * e + Ki * I + Kd * de/dt]_+ with e = F-hat - eps and the
    integral I itself projected to stay nonnegative.
    """

    def __init__(self, cfg: DualConfig) -> None:
        super().__init__(cfg.init, cfg.max_value, cfg.warm_start)
        self._kp, self._ki, self._kd = cfg.kp, cfg.ki, cfg.kd
        self._integral = 0.0
        self._prev_error: float | None = None

    def _reset_state(self) -> None:
        self._integral = 0.0
        self._prev_error = None

    def update(self, constraint_value: float, eps: float) -> float:
        error = constraint_value - eps
        self._integral = max(self._integral + error, 0.0)
        derivative = 0.0 if self._prev_error is None else error - self._prev_error
        self._prev_error = error
        raw = self._kp * error + self._ki * self._integral + self._kd * derivative
        self._value = min(max(raw, 0.0), self._max_value)
        return self._value
