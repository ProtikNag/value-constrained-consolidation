"""Estimator registry and construction from config."""

from __future__ import annotations

from crl.buffers import BufferSet
from crl.config import EstimatorConfig
from crl.estimators.base import ValueEstimator
from crl.estimators.exact import ExactEstimator
from crl.estimators.rollout import MonteCarloEstimator
from crl.estimators.surrogate import FrozenReferenceSurrogate

ESTIMATOR_REGISTRY = {
    "exact": ExactEstimator,
    "monte_carlo": MonteCarloEstimator,
    "frozen_surrogate": FrozenReferenceSurrogate,  # placeholder, see module doc
}


def make_estimator(
    cfg: EstimatorConfig, buffer_set: BufferSet | None = None
) -> ValueEstimator:
    """Instantiate the estimator named in the config."""
    if cfg.kind == "exact":
        return ExactEstimator()
    if cfg.kind == "monte_carlo":
        return MonteCarloEstimator(
            episodes_per_eval=cfg.episodes_per_eval,
            episodes_per_grad=cfg.episodes_per_grad,
            episodes_per_ref=cfg.episodes_per_ref,
            baseline=cfg.baseline,
            time_discount_weighting=cfg.time_discount_weighting,
            buffer_set=buffer_set,
        )
    if cfg.kind == "frozen_surrogate":
        return FrozenReferenceSurrogate()
    raise KeyError(
        f"Unknown estimator '{cfg.kind}'; available: {sorted(ESTIMATOR_REGISTRY)}"
    )


__all__ = ["ESTIMATOR_REGISTRY", "make_estimator", "ValueEstimator"]
