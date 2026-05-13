"""
Metric utilities for semantic segmentation.

Includes:
    - AverageMeter           : same as task1, running average of a scalar.
    - SegmentationMetric     : confusion-matrix-based pixel accuracy, per-class
                               IoU and mean IoU.
"""

from typing import Dict, List, Optional

import numpy as np
import torch


class AverageMeter:
    """Tracks running average of a scalar value."""

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.val: float = 0.0
        self.avg: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0

    def __repr__(self) -> str:
        return f"{self.name}: {self.avg:.4f}"


class SegmentationMetric:
    """
    Confusion-matrix-based metric tracker for semantic segmentation.
    """

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = -100,
    ) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        # Use torch tensor instead of numpy array
        self.confusion_matrix = torch.zeros(
            (num_classes, num_classes),
            dtype=torch.int64,
        )

    def reset(self) -> None:
        self.confusion_matrix.zero_()

    @torch.no_grad()
    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """
        Args:
            preds:   [N, H, W]
            targets: [N, H, W]
        """
        if preds.shape != targets.shape:
            raise ValueError(
                f"preds and targets must have the same shape, got "
                f"{preds.shape} vs {targets.shape}"
            )

        preds = preds.view(-1).to(torch.int64)
        targets = targets.view(-1).to(torch.int64)

        valid = (
            (targets != self.ignore_index)
            & (targets >= 0)
            & (targets < self.num_classes)
            & (preds >= 0)
            & (preds < self.num_classes)
        )

        if valid.sum() == 0:
            return

        preds = preds[valid]
        targets = targets[valid]

        indices = self.num_classes * targets + preds

        cm_update = torch.bincount(
            indices,
            minlength=self.num_classes * self.num_classes,
        ).reshape(self.num_classes, self.num_classes)

        self.confusion_matrix += cm_update.cpu()

    def compute(self) -> Dict[str, object]:
        """
        Returns:
            {
                pixel_acc,
                mean_iou,
                per_class_iou,
                confusion_matrix
            }
        """
        cm = self.confusion_matrix.to(torch.float64)

        tp = torch.diag(cm)

        fp = cm.sum(dim=0) - tp
        fn = cm.sum(dim=1) - tp

        total = cm.sum()

        pixel_acc = (
            float(tp.sum() / total)
            if total > 0
            else 0.0
        )

        denom = tp + fp + fn

        iou = torch.where(
            denom > 0,
            tp / torch.clamp(denom, min=1e-12),
            torch.tensor(float("nan"), device=cm.device),
        )

        per_class_iou: List[float] = [
            float(v) if not torch.isnan(v) else 0.0
            for v in iou
        ]

        valid_iou = iou[~torch.isnan(iou)]

        mean_iou = (
            float(valid_iou.mean())
            if valid_iou.numel() > 0
            else 0.0
        )

        return {
            "pixel_acc": pixel_acc,
            "mean_iou": mean_iou,
            "per_class_iou": per_class_iou,
            "confusion_matrix": cm.to(torch.int64).cpu().numpy(),
        }