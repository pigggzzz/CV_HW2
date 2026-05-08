"""
Loss functions for pet classification training.
"""

import torch
import torch.nn as nn


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross-entropy loss with label smoothing.

    Args:
        smoothing: Label smoothing factor in [0, 1).
        reduction: 'mean' or 'sum'.
    """

    def __init__(self, smoothing: float = 0.1, reduction: str = "mean") -> None:
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = logits.size(-1)
        log_probs = torch.log_softmax(logits, dim=-1)

        # Smooth targets: (1 - ε) * one_hot + ε / K
        with torch.no_grad():
            smooth_targets = torch.zeros_like(log_probs)
            smooth_targets.fill_(self.smoothing / n_classes)
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        loss = -(smooth_targets * log_probs).sum(dim=-1)
        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()


def build_criterion(cfg: dict) -> nn.Module:
    """
    Build the loss criterion from config.

    Uses standard CrossEntropyLoss (suitable for 37-class classification).
    Label smoothing can optionally be applied.
    """
    smoothing = cfg.get("training", {}).get("label_smoothing", 0.0)
    if smoothing > 0:
        return LabelSmoothingCrossEntropy(smoothing=smoothing)
    return nn.CrossEntropyLoss()
