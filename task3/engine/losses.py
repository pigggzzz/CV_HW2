"""
Loss functions for semantic segmentation.

This module defines three loss variants used in task3:

    1. Cross-Entropy Loss        (`nn.CrossEntropyLoss`)
    2. Dice Loss                 (hand-written, no external library)
    3. Combined Loss             (CE + Dice with configurable weights)

`build_criterion(cfg)` dispatches between them based on `cfg["loss"]["type"]`.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Multi-class soft Dice loss for semantic segmentation.

    For each class c, the Dice coefficient between prediction P_c and target T_c is

                            2 * Σ (P_c * T_c) + ε
        Dice_c =  --------------------------------------
                   Σ P_c + Σ T_c + ε

    The loss is `1 - mean_c(Dice_c)`. We use the soft probability from softmax
    rather than a hard argmax so the loss is differentiable.

    Args:
        num_classes: number of classes C.
        smooth:      ε term to stabilise the denominator and gradient.
        ignore_index: optional label id whose pixels are excluded from
            both numerator and denominator (set to a value not in {0..C-1}
            to disable).
    """

    def __init__(
        self,
        num_classes: int,
        smooth: float = 1e-6,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  Tensor of shape [N, C, H, W] - raw model outputs.
            targets: LongTensor of shape [N, H, W] with values in {0, ..., C-1}
                     (or `ignore_index`).

        Returns:
            Scalar tensor, the Dice loss averaged over classes.
        """
        if logits.dim() != 4:
            raise ValueError(f"DiceLoss expects 4-D logits [N,C,H,W], got {logits.shape}")
        if targets.dim() != 3:
            raise ValueError(f"DiceLoss expects 3-D targets [N,H,W], got {targets.shape}")

        n, c, h, w = logits.shape
        assert c == self.num_classes, (
            f"DiceLoss configured for {self.num_classes} classes but logits has {c}"
        )

        # Soft probability map.
        probs = F.softmax(logits, dim=1)  # [N, C, H, W]

        # Build a valid mask that excludes `ignore_index` pixels.
        valid_mask = (targets != self.ignore_index)             # [N, H, W]
        # Clamp targets so that one-hot indexing is safe even where invalid.
        safe_targets = targets.clamp(min=0, max=self.num_classes - 1)
        # One-hot encode: [N, H, W] -> [N, H, W, C] -> [N, C, H, W]
        one_hot = F.one_hot(safe_targets, num_classes=self.num_classes)
        one_hot = one_hot.permute(0, 3, 1, 2).float()           # [N, C, H, W]

        # Apply the valid mask to both prediction and target.
        valid = valid_mask.unsqueeze(1).float()                 # [N, 1, H, W]
        probs = probs * valid
        one_hot = one_hot * valid

        # Sum over the spatial (and batch) dimensions per class.
        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dim=dims)          # [C]
        cardinality = probs.sum(dim=dims) + one_hot.sum(dim=dims)  # [C]

        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        # Per-class loss; we want minimisation, so 1 - dice.
        loss = 1.0 - dice_per_class
        return loss.mean()


class CombinedLoss(nn.Module):
    """
    Weighted sum of CrossEntropyLoss and DiceLoss.

    Args:
        num_classes:  number of classes C (forwarded to DiceLoss).
        ce_weight:    weight applied to the CE component.
        dice_weight:  weight applied to the Dice component.
        smooth:       ε for the Dice term.
        ignore_index: label id to ignore in CE and Dice.
    """

    def __init__(
        self,
        num_classes: int,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        smooth: float = 1e-6,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        # PyTorch CrossEntropyLoss expects ignore_index >= 0 to actually skip,
        # but it also accepts -100 as the default sentinel.
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(
            num_classes=num_classes,
            smooth=smooth,
            ignore_index=ignore_index,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        if self.ce_weight > 0:
            loss = loss + self.ce_weight * self.ce(logits, targets)
        if self.dice_weight > 0:
            loss = loss + self.dice_weight * self.dice(logits, targets)
        return loss


def build_criterion(cfg: dict) -> nn.Module:
    """
    Build the loss criterion based on `cfg["loss"]["type"]`.

    Recognised types:
        - "ce"      : standard CrossEntropyLoss
        - "dice"    : hand-written multi-class Dice loss
        - "ce_dice" : weighted sum of CE and Dice

    Other relevant keys in cfg["loss"]:
        ce_weight, dice_weight, dice_smooth, ignore_index.
    """
    loss_cfg = cfg.get("loss", {}) or {}
    loss_type = str(loss_cfg.get("type", "ce")).lower()
    num_classes = int(cfg["model"].get("num_classes", 3))
    smooth = float(loss_cfg.get("dice_smooth", 1e-6))
    ignore_index = int(loss_cfg.get("ignore_index", -100))

    if loss_type == "ce":
        return nn.CrossEntropyLoss(ignore_index=ignore_index)
    elif loss_type == "dice":
        return DiceLoss(num_classes=num_classes, smooth=smooth, ignore_index=ignore_index)
    elif loss_type in ("ce_dice", "combined"):
        return CombinedLoss(
            num_classes=num_classes,
            ce_weight=float(loss_cfg.get("ce_weight", 1.0)),
            dice_weight=float(loss_cfg.get("dice_weight", 1.0)),
            smooth=smooth,
            ignore_index=ignore_index,
        )
    else:
        raise ValueError(
            f"Unknown loss type '{loss_type}'. Choose from: ce, dice, ce_dice."
        )
