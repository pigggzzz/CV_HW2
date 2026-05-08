"""
Metric utilities: running averages and top-k accuracy computation.
"""

from typing import Tuple

import torch


class AverageMeter:
    """
    Tracks the running average and sum of a scalar value.

    Usage:
        meter = AverageMeter("loss")
        meter.update(loss.item(), batch_size)
        print(meter.avg)
    """

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


def accuracy(
    output: torch.Tensor,
    target: torch.Tensor,
    topk: Tuple[int, ...] = (1,),
) -> Tuple[torch.Tensor, ...]:
    """
    Compute top-k accuracy for the given logits and targets.

    Args:
        output: Logit tensor of shape (N, C).
        target: Ground-truth labels of shape (N,).
        topk: Tuple of k values for which to compute accuracy.

    Returns:
        Tuple of scalar tensors (one per k value), each representing
        the percentage of correct predictions (0–100).
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()  # (maxk, N)
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        results = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            results.append(correct_k.mul_(100.0 / batch_size))

        return tuple(results)
