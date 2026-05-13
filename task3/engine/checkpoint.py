"""
Checkpoint saving and loading utilities.

Stores both the most recent epoch (`last.pth`) and the best-by-metric
checkpoint (`best.pth`). For segmentation the tracked metric is mean IoU.
"""

import os
from typing import Any, Dict, Optional

import torch


class CheckpointManager:
    """
    Manages model checkpoints during training.

    Saves:
      - best.pth  : the epoch with highest validation mIoU so far.
      - last.pth  : the most recent epoch.

    Args:
        ckpt_dir:        Top-level checkpoint directory.
        experiment_name: Subdir for this experiment.
        metric_name:     Name of the metric used to determine best (default 'mIoU').
    """

    def __init__(
        self,
        ckpt_dir: str,
        experiment_name: str,
        metric_name: str = "mIoU",
    ) -> None:
        self.save_dir = os.path.join(ckpt_dir, experiment_name)
        os.makedirs(self.save_dir, exist_ok=True)
        self.metric_name = metric_name
        self.best_metric: float = -float("inf")
        self.best_epoch: int = -1

    @property
    def best_path(self) -> str:
        return os.path.join(self.save_dir, "best.pth")

    @property
    def last_path(self) -> str:
        return os.path.join(self.save_dir, "last.pth")

    def save(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        val_metric: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Save the latest checkpoint; also save best if val_metric improves.

        Returns True if this was the best checkpoint so far.
        """
        state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "val_metric": val_metric,
            "metric_name": self.metric_name,
            **(extra or {}),
        }
        torch.save(state, self.last_path)

        is_best = val_metric > self.best_metric
        if is_best:
            self.best_metric = val_metric
            self.best_epoch = epoch
            torch.save(state, self.best_path)
        return is_best

    def load_best(self, model: torch.nn.Module, device: torch.device) -> Dict[str, Any]:
        """Load best checkpoint into model; return the full state dict."""
        if not os.path.exists(self.best_path):
            raise FileNotFoundError(f"No best checkpoint found at {self.best_path}")
        state = torch.load(self.best_path, map_location=device)
        model.load_state_dict(state["model_state"])
        return state

    def load_last(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        scheduler,
        device: torch.device,
    ) -> Dict[str, Any]:
        """Load latest checkpoint for training resumption."""
        if not os.path.exists(self.last_path):
            raise FileNotFoundError(f"No checkpoint found at {self.last_path}")
        state = torch.load(self.last_path, map_location=device)
        model.load_state_dict(state["model_state"])
        if optimizer and state.get("optimizer_state"):
            optimizer.load_state_dict(state["optimizer_state"])
        if scheduler and state.get("scheduler_state"):
            scheduler.load_state_dict(state["scheduler_state"])
        return state
