"""
U-Net for semantic segmentation, hand-written from scratch.

Reference:
    Ronneberger, Fischer, Brox.
    "U-Net: Convolutional Networks for Biomedical Image Segmentation." MICCAI 2015.

Architecture summary (with base_channels=c, bilinear=False):

    Encoder                         Decoder
    -----------------               --------------------------
    inc   :  3   ->   c             up1 : (16c , skip 8c ) -> 8c
    down1 :  c   -> 2c              up2 : (8c  , skip 4c ) -> 4c
    down2 : 2c   -> 4c              up3 : (4c  , skip 2c ) -> 2c
    down3 : 4c   -> 8c              up4 : (2c  , skip  c ) -> c
    down4 : 8c   -> 16c             outc:  c -> num_classes

When `bilinear=True` we halve the bottleneck width (16c -> 8c) to keep the
parameter count modest, mirroring the well-known "bilinear" variant.

All weights are randomly initialised: NO pretrained checkpoints are loaded.
"""

import torch
import torch.nn as nn

from .blocks import DoubleConv, Down, Up, OutConv


class UNet(nn.Module):
    """
    Args:
        in_channels:   number of input image channels (3 for RGB).
        num_classes:   number of output classes for per-pixel prediction.
        base_channels: width of the first stage; subsequent stages double.
        bilinear:      use bilinear upsampling instead of transposed conv.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 3,
        base_channels: int = 64,
        bilinear: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        c = base_channels
        factor = 2 if bilinear else 1

        # ----- Encoder -----
        self.inc = DoubleConv(in_channels, c)               # output: c
        self.down1 = Down(c, 2 * c)                         # output: 2c
        self.down2 = Down(2 * c, 4 * c)                     # output: 4c
        self.down3 = Down(4 * c, 8 * c)                     # output: 8c
        self.down4 = Down(8 * c, 16 * c // factor)          # output: 16c or 8c

        # ----- Decoder (with skip connections) -----
        # The first argument of `Up` is the channel count fed into the inner
        # DoubleConv, i.e. (upsampled_channels + skip_channels).
        #
        # bilinear=True : upsample keeps channel count; skip == upsample_in
        # bilinear=False: transposed conv halves channels first, so the sum
        #                 equals the original number going into the Up block.
        if bilinear:
            # bottleneck has 8c channels; after upsample still 8c, concat 8c skip → 16c
            self.up1 = Up(16 * c, 8 * c // factor, bilinear=True)   # cat in: 16c, out: 4c
            self.up2 = Up(8 * c, 4 * c // factor, bilinear=True)    # cat in: 8c,  out: 2c
            self.up3 = Up(4 * c, 2 * c // factor, bilinear=True)    # cat in: 4c,  out: c
            self.up4 = Up(2 * c, c, bilinear=True)                  # cat in: 2c,  out: c
        else:
            # bottleneck has 16c channels; after transposed conv 8c, concat 8c skip → 16c
            self.up1 = Up(16 * c, 8 * c, bilinear=False)            # cat in: 16c, out: 8c
            self.up2 = Up(8 * c, 4 * c, bilinear=False)             # cat in: 8c,  out: 4c
            self.up3 = Up(4 * c, 2 * c, bilinear=False)             # cat in: 4c,  out: 2c
            self.up4 = Up(2 * c, c, bilinear=False)                 # cat in: 2c,  out: c

        self.outc = OutConv(c, num_classes)

        self._init_weights()

    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Kaiming init for Conv layers; identity for BN."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)

    # ------------------------------------------------------------------

    def get_param_groups(self, lr: float) -> list:
        """Return a single parameter group with the given learning rate.

        Mirrors task1's `get_param_groups()` API so the Trainer code remains
        uniform; segmentation has no separate backbone / head so we return one
        group.
        """
        return [{"params": self.parameters(), "lr": lr}]


def build_unet(model_cfg: dict) -> UNet:
    """Construct a UNet from the `model:` section of a YAML config."""
    return UNet(
        in_channels=int(model_cfg.get("in_channels", 3)),
        num_classes=int(model_cfg.get("num_classes", 3)),
        base_channels=int(model_cfg.get("base_channels", 64)),
        bilinear=bool(model_cfg.get("bilinear", False)),
    )
