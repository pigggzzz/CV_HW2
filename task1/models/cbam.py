"""
Convolutional Block Attention Module (CBAM).

Reference:
    Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018.
    https://arxiv.org/abs/1807.06521

CBAM applies two sequential attention sub-modules:
  1. Channel Attention Module (CAM): focuses on *which* features matter.
  2. Spatial Attention Module (SAM): focuses on *where* informative regions are.
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """
    Channel Attention Module.

    Uses both max-pooled and average-pooled features, each passed through
    a shared MLP, then summed before sigmoid activation.

    Args:
        channels: Number of input channels.
        reduction: Bottleneck reduction ratio.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # Shared MLP (implemented as 1×1 convolutions to avoid flatten)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)  # (B, C, 1, 1)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module.

    Computes a 2D spatial attention map from channel-pooled features.

    Args:
        kernel_size: Convolution kernel size (typically 7).
    """

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)
        combined = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        return self.sigmoid(self.conv(combined))  # (B, 1, H, W)


class CBAM(nn.Module):
    """
    Full CBAM block: Channel Attention followed by Spatial Attention.

    Args:
        channels: Number of input channels.
        reduction: Channel attention reduction ratio.
        kernel_size: Spatial attention kernel size.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        kernel_size: int = 7,
    ) -> None:
        super().__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_attn(x)   # channel recalibration
        x = x * self.spatial_attn(x)   # spatial recalibration
        return x
