"""
Training engine for one complete segmentation training run.

Responsibilities:
  - build the optimizer and scheduler
  - run the epoch training loop (with mixed precision)
  - run validation after each epoch
  - save checkpoints (by val mIoU)
  - log metrics through the unified ExperimentLogger
  - periodically push qualitative sample images to the tracker

Heavy artefacts (visualisation, per-class IoU, etc.) live in `utils/`; the
trainer only orchestrates them.
"""

import time
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader

from data import PET_SEG_CLASS_NAMES, denormalize
from utils.logger import ExperimentLogger
from utils.metrics import AverageMeter, SegmentationMetric
from utils.visualization import make_triplet

from .checkpoint import CheckpointManager
from .evaluator import SegmentationEvaluator
from .losses import build_criterion


class SegmentationTrainer:
    """
    Full training loop for a single segmentation experiment configuration.

    Args:
        model:         Model to train.
        dataloaders:   Dict with 'train' and 'val' DataLoaders.
        cfg:           Full experiment config dict.
        device:        Torch device.
        logger:        ExperimentLogger instance.
        ckpt_manager:  CheckpointManager instance.
    """

    def __init__(
        self,
        model: nn.Module,
        dataloaders: Dict[str, DataLoader],
        cfg: dict,
        device: torch.device,
        logger: ExperimentLogger,
        ckpt_manager: CheckpointManager,
    ) -> None:
        self.model = model.to(device)
        self.dataloaders = dataloaders
        self.cfg = cfg
        self.device = device
        self.logger = logger
        self.ckpt_manager = ckpt_manager

        train_cfg = cfg["training"]
        self.epochs = int(train_cfg["epochs"])
        self.mixed_precision = bool(train_cfg.get("mixed_precision", True)) and device.type == "cuda"
        self.gradient_clip = train_cfg.get("gradient_clip", None)
        self.log_interval = int(cfg.get("logging", {}).get("log_interval", 20))
        self.num_vis_samples = int(cfg.get("logging", {}).get("num_vis_samples", 6))
        self.val_vis_interval = int(train_cfg.get("val_vis_interval", 5))

        self.num_classes = int(cfg["model"].get("num_classes", 3))
        self.ignore_index = int(cfg.get("loss", {}).get("ignore_index", -100))
        self.class_names = list(PET_SEG_CLASS_NAMES[: self.num_classes])

        self.criterion = build_criterion(cfg)
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.scaler = GradScaler("cuda", enabled=self.mixed_precision)

    # ------------------------------------------------------------------
    # Optimizer / scheduler
    # ------------------------------------------------------------------

    def _build_optimizer(self) -> torch.optim.Optimizer:
        tc = self.cfg["training"]
        lr = float(tc.get("lr", 1e-3))
        weight_decay = float(tc.get("weight_decay", 1e-4))
        momentum = float(tc.get("momentum", 0.9))
        name = str(tc.get("optimizer", "adamw")).lower()
        param_groups = self.model.get_param_groups(lr)

        if name == "sgd":
            return torch.optim.SGD(param_groups, momentum=momentum, weight_decay=weight_decay)
        elif name == "adamw":
            return torch.optim.AdamW(param_groups, weight_decay=weight_decay)
        elif name == "adam":
            return torch.optim.Adam(param_groups, weight_decay=weight_decay)
        raise ValueError(f"Unknown optimizer: {name}")

    def _build_scheduler(self):
        tc = self.cfg["training"]
        name = str(tc.get("scheduler", "cosine")).lower()
        if name == "cosine":
            return CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=1e-6)
        if name == "step":
            return StepLR(
                self.optimizer,
                step_size=int(tc.get("step_size", 10)),
                gamma=float(tc.get("gamma", 0.5)),
            )
        if name in ("none", "null", ""):
            return None
        raise ValueError(f"Unknown scheduler: {name}")

    # ------------------------------------------------------------------
    # Training entry point
    # ------------------------------------------------------------------

    def train(self) -> Dict[str, list]:
        """Run the full training loop and return history dict."""
        history = {
            "train_loss": [],
            "train_miou": [],
            "train_pixel_acc": [],
            "val_loss": [],
            "val_miou": [],
            "val_pixel_acc": [],
            "val_per_class_iou": [],
            "lr": [],
        }

        evaluator = SegmentationEvaluator(
            model=self.model,
            loader=self.dataloaders["val"],
            criterion=self.criterion,
            device=self.device,
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
        )

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_stats = self._train_one_epoch(epoch)
            val_metrics = evaluator.evaluate()
            if self.scheduler is not None:
                self.scheduler.step()

            cur_lr = self.optimizer.param_groups[0]["lr"]

            history["train_loss"].append(train_stats["loss"])
            history["train_miou"].append(train_stats["mean_iou"])
            history["train_pixel_acc"].append(train_stats["pixel_acc"])
            history["val_loss"].append(val_metrics["loss"])
            history["val_miou"].append(val_metrics["mean_iou"])
            history["val_pixel_acc"].append(val_metrics["pixel_acc"])
            history["val_per_class_iou"].append(val_metrics["per_class_iou"])
            history["lr"].append(cur_lr)

            is_best = self.ckpt_manager.save(
                epoch=epoch,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                val_metric=val_metrics["mean_iou"],
            )

            elapsed = time.time() - t0
            self.logger.log_epoch(
                epoch=epoch,
                total_epochs=self.epochs,
                train_loss=train_stats["loss"],
                train_miou=train_stats["mean_iou"],
                val_loss=val_metrics["loss"],
                val_miou=val_metrics["mean_iou"],
                val_pixel_acc=val_metrics["pixel_acc"],
                per_class_iou=val_metrics["per_class_iou"],
                class_names=self.class_names,
                lr=cur_lr,
                elapsed=elapsed,
                is_best=is_best,
            )

            # Periodically push qualitative samples to the tracker.
            if (
                self.val_vis_interval > 0
                and (epoch % self.val_vis_interval == 0 or epoch == self.epochs)
            ):
                self._log_val_samples(evaluator, epoch)

        return history

    # ------------------------------------------------------------------
    # Single epoch
    # ------------------------------------------------------------------

    def _train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch; return summary stats."""
        self.model.train()
        loader = self.dataloaders["train"]
        loss_meter = AverageMeter("loss")
        metric = SegmentationMetric(self.num_classes, ignore_index=self.ignore_index)
        n_batches = len(loader)

        for batch_idx, (images, masks) in enumerate(loader):
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=self.mixed_precision):
                logits = self.model(images)
                loss = self.criterion(logits, masks)

            self.scaler.scale(loss).backward()

            if self.gradient_clip is not None:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), float(self.gradient_clip))

            self.scaler.step(self.optimizer)
            self.scaler.update()

            bsz = images.size(0)
            loss_meter.update(loss.item(), bsz)
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                metric.update(preds, masks)

            if (batch_idx + 1) % self.log_interval == 0:
                self.logger.log_batch(
                    epoch=epoch,
                    batch=batch_idx + 1,
                    total_batches=n_batches,
                    loss=loss_meter.avg,
                )

        result = metric.compute()
        return {
            "loss": loss_meter.avg,
            "mean_iou": result["mean_iou"],
            "pixel_acc": result["pixel_acc"],
        }

    # ------------------------------------------------------------------
    # Visualisation hook
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _log_val_samples(self, evaluator: SegmentationEvaluator, epoch: int) -> None:
        """Push a few (image, GT, Pred) triplets to the tracker."""
        if self.num_vis_samples <= 0:
            return
        samples = evaluator.collect_samples(max_samples=self.num_vis_samples)
        triplets = []
        for img_t, gt_t, pred_t in samples:
            img_np = (
                denormalize(img_t.unsqueeze(0))[0]
                .permute(1, 2, 0)
                .numpy()
            )
            img_np = (img_np * 255.0).astype(np.uint8)
            triplet = make_triplet(img_np, gt_t.numpy(), pred_t.numpy())
            triplets.append(triplet)
        captions = [f"epoch={epoch} sample={i}" for i in range(len(triplets))]
        self.logger.log_images(
            triplets,
            step=epoch,
            tag="val/samples",
            captions=captions,
        )
