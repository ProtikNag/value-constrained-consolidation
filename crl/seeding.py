"""Global seeding for reproducible runs."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed every RNG the project touches.

    Same seed must produce identical runs; ``tests/test_seeding.py``
    enforces this.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
