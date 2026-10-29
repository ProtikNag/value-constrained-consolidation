"""Dual-controller behavior: projection, growth under violation, decay under slack."""

from crl.config import DualConfig
from crl.duals import make_dual


def test_projected_ascent_grows_under_violation():
    dual = make_dual(DualConfig(kind="projected_ascent", lr=0.1, init=0.0))
    for _ in range(10):
        dual.update(constraint_value=1.0, eps=0.1)  # violated by 0.9
    assert abs(dual.value - 0.9) < 1e-9


def test_projected_ascent_decays_to_zero_under_slack():
    dual = make_dual(DualConfig(kind="projected_ascent", lr=0.1, init=0.5))
    for _ in range(100):
        dual.update(constraint_value=-1.0, eps=0.1)  # comfortably satisfied
    assert dual.value == 0.0


def test_projected_ascent_respects_cap():
    dual = make_dual(DualConfig(kind="projected_ascent", lr=10.0, max_value=3.0))
    for _ in range(50):
        dual.update(constraint_value=100.0, eps=0.0)
    assert dual.value == 3.0


def test_reset_semantics():
    cold = make_dual(DualConfig(kind="projected_ascent", lr=0.1, init=0.0,
                                warm_start=False))
    cold.update(constraint_value=5.0, eps=0.0)
    cold.reset()
    assert cold.value == 0.0

    warm = make_dual(DualConfig(kind="projected_ascent", lr=0.1, init=0.0,
                                warm_start=True))
    warm.update(constraint_value=5.0, eps=0.0)
    warm.reset()
    assert warm.value > 0.0


def test_pid_nonnegative_and_responsive():
    dual = make_dual(DualConfig(kind="pid", kp=0.5, ki=0.1, kd=0.0))
    assert dual.update(constraint_value=-1.0, eps=0.0) == 0.0  # no violation
    value_after_violation = dual.update(constraint_value=1.0, eps=0.0)
    assert value_after_violation > 0.0
