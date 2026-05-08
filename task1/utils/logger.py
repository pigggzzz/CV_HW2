"""
Experiment logging: console output + optional W&B / SwanLab integration.

The ExperimentLogger wraps both a Python logging handler (for console/file)
and optional third-party experiment trackers (wandb or swanlab).
All metric logging goes through this single interface so that swapping or
disabling a tracker requires only a config change.
"""

import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional


class ExperimentLogger:
    """
    Unified logger for training experiments.

    Args:
        experiment_name: Name of the current experiment (used as run name).
        log_dir: Directory where log files are written.
        cfg: Full experiment config dict.
        use_wandb: Enable Weights & Biases logging.
        use_swanlab: Enable SwanLab logging.
    """

    def __init__(
        self,
        experiment_name: str,
        log_dir: str,
        cfg: Dict[str, Any],
        use_wandb: bool = False,
        use_swanlab: bool = False,
    ) -> None:
        self.experiment_name = experiment_name
        self.log_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)

        self._setup_file_logger()
        self._tracker = None
        self._tracker_type: Optional[str] = None

        if use_wandb:
            self._init_wandb(cfg)
        elif use_swanlab:
            self._init_swanlab(cfg)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_file_logger(self) -> None:
        log_path = os.path.join(self.log_dir, "train.log")
        fmt = "[%(asctime)s][%(levelname)s] %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"

        self._logger = logging.getLogger(self.experiment_name)
        self._logger.setLevel(logging.INFO)
        self._logger.handlers.clear()

        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter(fmt, datefmt=datefmt))

        self._logger.addHandler(fh)
        self._logger.addHandler(sh)
        self._logger.propagate = False

    def _init_wandb(self, cfg: Dict[str, Any]) -> None:
        try:
            import wandb
            project = cfg.get("logging", {}).get("project", "pet-classification")
            wandb.init(
                project=project,
                name=self.experiment_name,
                config=cfg,
                dir=self.log_dir,
            )
            self._tracker = wandb
            self._tracker_type = "wandb"
            self.info("Weights & Biases logging initialised.")
        except ImportError:
            self.warning("wandb not installed; skipping W&B logging.")

    def _init_swanlab(self, cfg: Dict[str, Any]) -> None:
        try:
            import swanlab
            project = cfg.get("logging", {}).get("project", "pet-classification")
            swanlab.init(
                project=project,
                experiment_name=self.experiment_name,
                config=cfg,
                logdir=self.log_dir,
            )
            self._tracker = swanlab
            self._tracker_type = "swanlab"
            self.info("SwanLab logging initialised.")
        except ImportError:
            self.warning("swanlab not installed; skipping SwanLab logging.")

    # ------------------------------------------------------------------
    # Core logging helpers
    # ------------------------------------------------------------------

    def info(self, msg: str) -> None:
        self._logger.info(msg)

    def warning(self, msg: str) -> None:
        self._logger.warning(msg)

    def _log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """Forward metrics to the active tracker if one is configured."""
        if self._tracker is None:
            return
        if self._tracker_type == "wandb":
            self._tracker.log(metrics, step=step)
        elif self._tracker_type == "swanlab":
            self._tracker.log(metrics, step=step)

    # ------------------------------------------------------------------
    # Structured epoch / batch logging
    # ------------------------------------------------------------------

    def log_epoch(
        self,
        epoch: int,
        total_epochs: int,
        train_loss: float,
        val_loss: float,
        val_top1: float,
        val_top5: float,
        backbone_lr: float,
        head_lr: float,
        elapsed: float,
        is_best: bool,
    ) -> None:
        """Log end-of-epoch summary."""
        best_flag = " [BEST]" if is_best else ""
        msg = (
            f"Epoch [{epoch:3d}/{total_epochs}] "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_top1={val_top1:.2f}%  "
            f"val_top5={val_top5:.2f}%  "
            f"bb_lr={backbone_lr:.2e}  "
            f"head_lr={head_lr:.2e}  "
            f"time={elapsed:.1f}s"
            f"{best_flag}"
        )
        self.info(msg)

        self._log_metrics(
            {
                "train/loss": train_loss,
                "val/loss": val_loss,
                "val/top1": val_top1,
                "val/top5": val_top5,
                "lr/backbone": backbone_lr,
                "lr/head": head_lr,
            },
            step=epoch,
        )

    def log_batch(
        self,
        epoch: int,
        batch: int,
        total_batches: int,
        loss: float,
        top1: float,
    ) -> None:
        """Log in-epoch batch progress (console only)."""
        self.info(
            f"  Epoch {epoch} [{batch:4d}/{total_batches}]  "
            f"loss={loss:.4f}  top1={top1:.2f}%"
        )

    def log_test(self, metrics: Dict[str, Any]) -> None:
        """Log final test set evaluation results."""
        self.info("=" * 60)
        self.info("TEST RESULTS")
        self.info(f"  top-1 accuracy : {metrics['top1']:.2f}%")
        self.info(f"  top-5 accuracy : {metrics['top5']:.2f}%")
        self.info(f"  loss           : {metrics['loss']:.4f}")
        self.info("=" * 60)

        self._log_metrics(
            {
                "test/top1": metrics["top1"],
                "test/top5": metrics["top5"],
                "test/loss": metrics["loss"],
            },
            step=0,
        )

        # Also persist to JSON for easy post-processing
        result_path = os.path.join(self.log_dir, "test_results.json")
        with open(result_path, "w") as f:
            json.dump(metrics, f, indent=2)
        self.info(f"Test results saved to {result_path}")

    def save_history(self, history: Dict[str, List]) -> None:
        """Persist training history curves to JSON."""
        path = os.path.join(self.log_dir, "history.json")
        with open(path, "w") as f:
            json.dump(history, f, indent=2)
        self.info(f"Training history saved to {path}")

    def finish(self) -> None:
        """Finalise the tracker (required for wandb/swanlab)."""
        if self._tracker_type == "wandb":
            self._tracker.finish()
        elif self._tracker_type == "swanlab":
            self._tracker.finish()
