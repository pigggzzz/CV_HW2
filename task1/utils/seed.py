"""
Global random seed utilities for reproducibility.

Sets seeds for Python's random module, NumPy, and PyTorch (CPU + CUDA).
Also configures cuDNN for deterministic behaviour at the cost of some speed.
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Fix all random seeds for reproducible training.

    Args:
        seed: Integer seed value.
        deterministic: If True, forces cuDNN to use deterministic algorithms.
            This may reduce performance but ensures exact reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Allow cuDNN auto-tuner for faster (non-deterministic) training
        torch.backends.cudnn.benchmark = True
