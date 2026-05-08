"""
Training engine for one complete training run.

Handles:
  - optimizer and scheduler construction
  - the epoch training loop with mixed precision
  - validation after each epoch
  - checkpoint saving
  - metric logging (console + wandb/swanlab)
"""

import time
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader

from .checkpoint import CheckpointManager
from .evaluator import Evaluator
from .losses import build_criterion
from utils.logger import ExperimentLogger
from utils.metrics import AverageMeter, accuracy


class Trainer:
    """
    Full training loop for one experiment configuration.

    Args:
        model: Model to train (must implement get_param_groups()).
        dataloaders: Dict with 'train', 'val' DataLoaders.
        cfg: Full experiment config dict.
        device: Computation device.
        logger: ExperimentLogger instance.
        ckpt_manager: CheckpointManager instance.
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
        self.epochs = train_cfg["epochs"]
        self.mixed_precision = train_cfg.get("mixed_precision", True) and device.type == "cuda"
        self.gradient_clip = train_cfg.get("gradient_clip", None)
        self.log_interval = cfg.get("logging", {}).get("log_interval", 10)

        self.criterion = build_criterion(cfg)
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.scaler = GradScaler("cuda",enabled=self.mixed_precision)
        self.num_classes = cfg["model"].get("num_classes", 37)

    # ------------------------------------------------------------------
    # Optimizer and scheduler construction
    # ------------------------------------------------------------------

    def _build_optimizer(self) -> torch.optim.Optimizer:
        train_cfg = self.cfg["training"]
        backbone_lr = float(train_cfg["backbone_lr"])
        head_lr = float(train_cfg["head_lr"])
        weight_decay = float(train_cfg.get("weight_decay", 1e-4))
        momentum = float(train_cfg.get("momentum", 0.9))
        opt_name = train_cfg.get("optimizer", "sgd").lower()

        param_groups = self.model.get_param_groups(backbone_lr, head_lr)

        if opt_name == "sgd":
            return torch.optim.SGD(
                param_groups,
                momentum=momentum,
                weight_decay=weight_decay,
            )
        elif opt_name == "adamw":
            return torch.optim.AdamW(param_groups, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

    def _build_scheduler(self):
        train_cfg = self.cfg["training"]
        sched_name = train_cfg.get("scheduler", "cosine").lower()
        if sched_name == "cosine":
            return CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=1e-6)
        elif sched_name == "step":
            return StepLR(
                self.optimizer,
                step_size=int(train_cfg.get("step_size", 10)),
                gamma=float(train_cfg.get("gamma", 0.1)),
            )
        else:
            raise ValueError(f"Unknown scheduler: {sched_name}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> Dict[str, list]:
        """Run the full training loop and return history."""
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_top1": [],
            "val_top5": [],
            "backbone_lr": [],
            "head_lr": [],
        }

        evaluator = Evaluator(
            self.model,
            self.dataloaders["val"],
            self.criterion,
            self.device,
            self.num_classes,
        )

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_loss = self._train_one_epoch(epoch)
            val_metrics = evaluator.evaluate()
            self.scheduler.step()

            # Record current learning rates
            backbone_lr = self.optimizer.param_groups[0]["lr"]
            head_lr = self.optimizer.param_groups[1]["lr"]

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_metrics["loss"])
            history["val_top1"].append(val_metrics["top1"])
            history["val_top5"].append(val_metrics["top5"])
            history["backbone_lr"].append(backbone_lr)
            history["head_lr"].append(head_lr)

            is_best = self.ckpt_manager.save(
                epoch=epoch,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                val_acc=val_metrics["top1"],
            )

            elapsed = time.time() - t0
            self.logger.log_epoch(
                epoch=epoch,
                total_epochs=self.epochs,
                train_loss=train_loss,
                val_loss=val_metrics["loss"],
                val_top1=val_metrics["top1"],
                val_top5=val_metrics["top5"],
                backbone_lr=backbone_lr,
                head_lr=head_lr,
                elapsed=elapsed,
                is_best=is_best,
            )

        return history

    def _train_one_epoch(self, epoch: int) -> float:
        """Train for one epoch; return average loss."""
        self.model.train()
        loader = self.dataloaders["train"]
        loss_meter = AverageMeter("loss")
        top1_meter = AverageMeter("top1")
        n_batches = len(loader)

        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast("cuda",enabled=self.mixed_precision):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()

            if self.gradient_clip is not None:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            bsz = images.size(0)
            top1, _ = accuracy(logits.detach(), labels, topk=(1, 5))
            loss_meter.update(loss.item(), bsz)
            top1_meter.update(top1.item(), bsz)

            if (batch_idx + 1) % self.log_interval == 0:
                self.logger.log_batch(
                    epoch=epoch,
                    batch=batch_idx + 1,
                    total_batches=n_batches,
                    loss=loss_meter.avg,
                    top1=top1_meter.avg,
                )

        return loss_meter.avg
