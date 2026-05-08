"""
Squeeze-and-Excitation (SE) Block.

Reference:
    Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018.
    https://arxiv.org/abs/1709.01507

The SE block recalibrates channel-wise feature responses by:
  1. Squeeze: global average pooling → channel descriptor.
  2. Excitation: two FC layers with ReLU + Sigmoid → channel weights.
  3. Scale: multiply feature map by channel weights.
"""

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """
    Channel-wise Squeeze-and-Excitation block.

    Args:
        channels: Number of input (and output) channels.
        reduction: Reduction ratio for the bottleneck FC layer.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(1, channels // reduction)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        s = self.squeeze(x)              # (B, C, 1, 1)
        e = self.excitation(s)           # (B, C)
        e = e.view(b, c, 1, 1)          # (B, C, 1, 1)
        return x * e
