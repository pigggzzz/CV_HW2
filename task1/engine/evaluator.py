"""
Model evaluation on a DataLoader split.

Computes top-1 accuracy, top-5 accuracy, and per-class accuracy.
Also supports optional confusion matrix generation.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.metrics import AverageMeter, accuracy


class Evaluator:
    """
    Runs inference on a DataLoader and computes classification metrics.

    Args:
        model: Model to evaluate.
        loader: DataLoader providing (images, labels).
        criterion: Loss function.
        device: Computation device.
        num_classes: Total number of classes.
    """

    def __init__(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        num_classes: int = 37,
    ) -> None:
        self.model = model
        self.loader = loader
        self.criterion = criterion
        self.device = device
        self.num_classes = num_classes

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        Run full evaluation over the DataLoader.

        Returns:
            Dictionary with keys: loss, top1, top5, per_class_acc (list).
        """
        self.model.eval()
        loss_meter = AverageMeter("loss")
        top1_meter = AverageMeter("top1")
        top5_meter = AverageMeter("top5")

        # Per-class tracking
        class_correct = torch.zeros(self.num_classes, dtype=torch.long)
        class_total = torch.zeros(self.num_classes, dtype=torch.long)

        for images, labels in self.loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            logits = self.model(images)
            loss = self.criterion(logits, labels)

            top1, top5 = accuracy(logits, labels, topk=(1, 5))
            bsz = images.size(0)

            loss_meter.update(loss.item(), bsz)
            top1_meter.update(top1.item(), bsz)
            top5_meter.update(top5.item(), bsz)

            # Accumulate per-class stats
            preds = logits.argmax(dim=1)
            for cls in range(self.num_classes):
                mask = labels == cls
                class_total[cls] += mask.sum().item()
                class_correct[cls] += ((preds == cls) & mask).sum().item()

        per_class = [
            (class_correct[c].item() / max(class_total[c].item(), 1)) * 100
            for c in range(self.num_classes)
        ]

        return {
            "loss": loss_meter.avg,
            "top1": top1_meter.avg,
            "top5": top5_meter.avg,
            "per_class_acc": per_class,
        }

    @torch.no_grad()
    def confusion_matrix(self) -> torch.Tensor:
        """Compute and return the confusion matrix as a (C×C) tensor."""
        self.model.eval()
        matrix = torch.zeros(self.num_classes, self.num_classes, dtype=torch.long)

        for images, labels in self.loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            preds = self.model(images).argmax(dim=1)

            for t, p in zip(labels.cpu(), preds.cpu()):
                matrix[t.item(), p.item()] += 1

        return matrix
