"""
Segmentation evaluator.

Runs inference over a DataLoader and aggregates:
    - average loss
    - pixel accuracy
    - per-class IoU
    - mean IoU (mIoU)
    - (optionally) qualitative samples for visualisation
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.metrics import AverageMeter, SegmentationMetric


class SegmentationEvaluator:
    """
    Args:
        model:        Segmentation model.
        loader:       DataLoader yielding (image, mask) batches.
        criterion:    Loss function (segmentation).
        device:       Torch device.
        num_classes:  Number of classes for IoU computation.
        ignore_index: Label id excluded from metrics (default: -100).
    """

    def __init__(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        num_classes: int = 3,
        ignore_index: int = -100,
    ) -> None:
        self.model = model
        self.loader = loader
        self.criterion = criterion
        self.device = device
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    @torch.no_grad()
    def evaluate(self) -> Dict[str, Any]:
        """
        Run a full pass and return aggregated metrics.

        Returns:
            {
                "loss": float,
                "pixel_acc": float,
                "mean_iou": float,
                "per_class_iou": List[float],
            }
        """
        self.model.eval()
        loss_meter = AverageMeter("loss")
        metric = SegmentationMetric(self.num_classes, ignore_index=self.ignore_index)

        for images, masks in self.loader:
            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            logits = self.model(images)
            loss = self.criterion(logits, masks)

            preds = logits.argmax(dim=1)
            bsz = images.size(0)
            loss_meter.update(loss.item(), bsz)
            metric.update(preds, masks)

        result = metric.compute()
        return {
            "loss": loss_meter.avg,
            "pixel_acc": result["pixel_acc"],
            "mean_iou": result["mean_iou"],
            "per_class_iou": result["per_class_iou"],
        }

    @torch.no_grad()
    def collect_samples(
        self,
        max_samples: int = 6,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Collect `max_samples` (image, gt_mask, pred_mask) triplets from the loader.

        Returns tensors on CPU so they can be fed directly into visualisation.
        Images stay in normalised space; the caller is expected to denormalise.
        """
        self.model.eval()
        samples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for images, masks in self.loader:
            images_dev = images.to(self.device, non_blocking=True)
            logits = self.model(images_dev)
            preds = logits.argmax(dim=1).cpu()
            for i in range(images.size(0)):
                if len(samples) >= max_samples:
                    return samples
                samples.append((images[i].cpu(), masks[i].cpu(), preds[i]))
        return samples
