"""
Experiment logger: console + file + optional wandb / swanlab.

The logger exposes a uniform `log_epoch` for the segmentation trainer and a
`log_images` helper to push qualitative samples to wandb. Falling back to a
no-op tracker when neither wandb nor swanlab is available keeps training
runs functional offline.
"""

import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Sequence


class ExperimentLogger:
    """
    Unified logger for segmentation experiments.

    Args:
        experiment_name: Name of the current experiment (used as wandb run name).
        log_dir:         Directory where local log files are written.
        cfg:             Full experiment config dict (logged to the tracker).
        use_wandb:       Enable Weights & Biases logging.
        use_swanlab:     Enable SwanLab logging.
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
            project = cfg.get("logging", {}).get("project", "pet-segmentation")
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
            project = cfg.get("logging", {}).get("project", "pet-segmentation")
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
    # Console helpers
    # ------------------------------------------------------------------

    def info(self, msg: str) -> None:
        self._logger.info(msg)

    def warning(self, msg: str) -> None:
        self._logger.warning(msg)

    # ------------------------------------------------------------------
    # Tracker forwarding
    # ------------------------------------------------------------------

    def _log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        if self._tracker is None:
            return
        if self._tracker_type == "wandb":
            self._tracker.log(metrics, step=step)
        elif self._tracker_type == "swanlab":
            self._tracker.log(metrics, step=step)

    def log_images(
        self,
        images: Sequence,
        step: int,
        tag: str = "val/samples",
        captions: Optional[Sequence[str]] = None,
    ) -> None:
        """
        Push a sequence of numpy uint8 HxWx3 images to the active tracker.

        Args:
            images:  iterable of np.ndarray (HxWx3 uint8).
            step:    epoch / global step index.
            tag:     log key under which the images appear.
            captions: optional list of captions (same length as images).
        """
        if self._tracker is None or not images:
            return
        captions = list(captions) if captions else [None] * len(images)
        if self._tracker_type == "wandb":
            wandb_imgs = [self._tracker.Image(img, caption=cap)
                          for img, cap in zip(images, captions)]
            self._tracker.log({tag: wandb_imgs}, step=step)
        elif self._tracker_type == "swanlab":
            sw_imgs = [self._tracker.Image(img, caption=cap)
                       for img, cap in zip(images, captions)]
            self._tracker.log({tag: sw_imgs}, step=step)

    # ------------------------------------------------------------------
    # Structured epoch / batch logging
    # ------------------------------------------------------------------

    def log_epoch(
        self,
        epoch: int,
        total_epochs: int,
        train_loss: float,
        train_miou: float,
        val_loss: float,
        val_miou: float,
        val_pixel_acc: float,
        per_class_iou: Sequence[float],
        class_names: Sequence[str],
        lr: float,
        elapsed: float,
        is_best: bool,
    ) -> None:
        """Print end-of-epoch summary and forward metrics to the tracker."""
        best_flag = " [BEST]" if is_best else ""
        iou_str = " ".join(
            f"{n}={v:.3f}" for n, v in zip(class_names, per_class_iou)
        )
        msg = (
            f"Epoch [{epoch:3d}/{total_epochs}] "
            f"train_loss={train_loss:.4f}  train_mIoU={train_miou:.4f}  "
            f"val_loss={val_loss:.4f}  val_mIoU={val_miou:.4f}  "
            f"val_pAcc={val_pixel_acc:.4f}  lr={lr:.2e}  "
            f"time={elapsed:.1f}s  [{iou_str}]"
            f"{best_flag}"
        )
        self.info(msg)

        metrics = {
            "train/loss": train_loss,
            "train/mIoU": train_miou,
            "val/loss": val_loss,
            "val/mIoU": val_miou,
            "val/pixel_acc": val_pixel_acc,
            "lr": lr,
        }
        for n, v in zip(class_names, per_class_iou):
            metrics[f"val/IoU_{n}"] = float(v)
        self._log_metrics(metrics, step=epoch)

    def log_batch(
        self,
        epoch: int,
        batch: int,
        total_batches: int,
        loss: float,
    ) -> None:
        """Console-only in-epoch progress log."""
        self.info(
            f"  Epoch {epoch} [{batch:4d}/{total_batches}]  loss={loss:.4f}"
        )

    def log_test(
        self,
        metrics: Dict[str, Any],
        class_names: Sequence[str],
    ) -> None:
        """Log final test-set evaluation results."""
        self.info("=" * 60)
        self.info("TEST RESULTS")
        self.info(f"  pixel accuracy : {metrics['pixel_acc']:.4f}")
        self.info(f"  mean IoU       : {metrics['mean_iou']:.4f}")
        self.info(f"  test loss      : {metrics['loss']:.4f}")
        for n, v in zip(class_names, metrics["per_class_iou"]):
            self.info(f"  IoU[{n:<11s}] = {v:.4f}")
        self.info("=" * 60)

        log_payload = {
            "test/loss": float(metrics["loss"]),
            "test/pixel_acc": float(metrics["pixel_acc"]),
            "test/mIoU": float(metrics["mean_iou"]),
        }
        for n, v in zip(class_names, metrics["per_class_iou"]):
            log_payload[f"test/IoU_{n}"] = float(v)
        self._log_metrics(log_payload, step=0)

        # Persist to JSON for downstream comparison plots.
        result_path = os.path.join(self.log_dir, "test_results.json")
        json_payload = {
            "loss": float(metrics["loss"]),
            "pixel_acc": float(metrics["pixel_acc"]),
            "mean_iou": float(metrics["mean_iou"]),
            "per_class_iou": {
                n: float(v) for n, v in zip(class_names, metrics["per_class_iou"])
            },
        }
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(json_payload, f, indent=2)
        self.info(f"Test results saved to {result_path}")

    def save_history(self, history: Dict[str, List]) -> None:
        """Persist training history curves to JSON."""
        path = os.path.join(self.log_dir, "history.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        self.info(f"Training history saved to {path}")

    def finish(self) -> None:
        """Finalise the tracker (required for wandb/swanlab)."""
        if self._tracker_type in ("wandb", "swanlab"):
            self._tracker.finish()
