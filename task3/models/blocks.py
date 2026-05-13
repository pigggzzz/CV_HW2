"""
Building blocks for the U-Net architecture.

All blocks are plain `nn.Module`s built from basic PyTorch ops:
    Conv2d, BatchNorm2d, ReLU, MaxPool2d, ConvTranspose2d, Upsample.

No external segmentation library is used.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Two consecutive 3x3 convolutions with BN and ReLU.

    (Conv 3x3 → BN → ReLU) x 2

    Args:
        in_channels:  number of input feature channels.
        out_channels: number of output feature channels.
        mid_channels: number of intermediate channels (defaults to out_channels).
    """

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None) -> None:
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """Encoder step: 2x2 max-pool followed by a DoubleConv."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class Up(nn.Module):
    """
    Decoder step: upsample, concatenate with skip features, then DoubleConv.

    Two upsampling variants are supported:
      - bilinear=True:  nn.Upsample + 1x1 Conv to halve channels (parameter-light).
      - bilinear=False: nn.ConvTranspose2d (learnable upsampling, original U-Net).

    Args:
        in_channels:  channels coming up from the deeper level.
        out_channels: channels after this decoder step.
        bilinear:     choose upsampling strategy.
    """

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False) -> None:
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            # After upsample we still have `in_channels` channels; channel reduction
            # is done by DoubleConv(in_channels // 2 + skip_channels, out_channels).
            # Conventionally skip_channels == in_channels // 2, so input to conv
            # is (in_channels // 2) + (in_channels // 2) = in_channels.
            self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            # After transpose conv we have in_channels // 2 channels; with the skip
            # we get in_channels total feeding into the DoubleConv.
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # If the upsampled tensor has slightly different spatial size than the
        # skip connection (due to odd input sizes), pad to match.
        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_x != 0 or diff_y != 0:
            x = F.pad(
                x,
                [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
            )
        # Concatenate skip features along the channel dim.
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """1x1 convolution that produces the per-pixel class logits."""

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
