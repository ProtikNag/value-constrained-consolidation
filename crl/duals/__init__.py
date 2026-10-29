"""Dual-controller registry and construction from config."""

from __future__ import annotations

from crl.config import DualConfig
from crl.duals.controllers import DualController, PIDDual, ProjectedAscentDual

DUAL_REGISTRY = {
    "projected_ascent": ProjectedAscentDual,
    "pid": PIDDual,
}


def make_dual(cfg: DualConfig) -> DualController:
    """Instantiate a fresh dual controller from config."""
    if cfg.kind not in DUAL_REGISTRY:
        raise KeyError(
            f"Unknown dual controller '{cfg.kind}'; available: {sorted(DUAL_REGISTRY)}"
        )
    return DUAL_REGISTRY[cfg.kind](cfg)


__all__ = ["DUAL_REGISTRY", "make_dual", "DualController"]
