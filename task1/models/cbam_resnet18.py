"""
CBAM-ResNet18: ResNet-18 augmented with CBAM (Channel + Spatial Attention).

A CBAM block is inserted after each residual block's BN,
applying channel attention then spatial attention on the residual branch.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights
from torchvision.models.resnet import BasicBlock

from .cbam import CBAM


class CBAMBasicBlock(nn.Module):
    """
    ResNet BasicBlock with a CBAM block appended to the residual branch.
    """

    expansion = 1

    def __init__(
        self,
        original_block: BasicBlock,
        reduction: int = 16,
        kernel_size: int = 7,
    ) -> None:
        super().__init__()
        self.conv1 = original_block.conv1
        self.bn1 = original_block.bn1
        self.relu = original_block.relu
        self.conv2 = original_block.conv2
        self.bn2 = original_block.bn2
        self.downsample = original_block.downsample
        self.stride = original_block.stride

        out_channels = original_block.conv2.out_channels
        self.cbam = CBAM(out_channels, reduction=reduction, kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.cbam(out)            # CBAM recalibration on residual branch

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.relu(out + identity)
        return out


def _inject_cbam_blocks(
    layer: nn.Sequential,
    reduction: int,
    kernel_size: int,
) -> nn.Sequential:
    """Replace every BasicBlock in a ResNet layer with a CBAMBasicBlock."""
    new_blocks = []
    for block in layer:
        if isinstance(block, BasicBlock):
            new_blocks.append(
                CBAMBasicBlock(block, reduction=reduction, kernel_size=kernel_size)
            )
        else:
            new_blocks.append(block)
    return nn.Sequential(*new_blocks)


class CBAMPetResNet18(nn.Module):
    """
    ResNet-18 with CBAM blocks for 37-class pet classification.

    Args:
        pretrained: Load ImageNet pretrained weights before injecting CBAM.
        num_classes: Number of output classes.
        reduction: Channel attention reduction ratio.
        kernel_size: Spatial attention kernel size.
    """

    def __init__(
        self,
        pretrained: bool = True,
        num_classes: int = 37,
        reduction: int = 16,
        kernel_size: int = 7,
    ) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.resnet18(weights=weights)

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool

        self.layer1 = _inject_cbam_blocks(base.layer1, reduction, kernel_size)
        self.layer2 = _inject_cbam_blocks(base.layer2, reduction, kernel_size)
        self.layer3 = _inject_cbam_blocks(base.layer3, reduction, kernel_size)
        self.layer4 = _inject_cbam_blocks(base.layer4, reduction, kernel_size)

        self.avgpool = base.avgpool
        in_features = base.fc.in_features

        self.classifier = nn.Linear(in_features, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def get_param_groups(self, backbone_lr: float, head_lr: float) -> list:
        """Differential learning rates: backbone vs classifier head."""
        backbone_params = (
            list(self.conv1.parameters())
            + list(self.bn1.parameters())
            + list(self.layer1.parameters())
            + list(self.layer2.parameters())
            + list(self.layer3.parameters())
            + list(self.layer4.parameters())
        )
        return [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": self.classifier.parameters(), "lr": head_lr},
        ]


def build_cbam_resnet18(model_cfg: dict) -> CBAMPetResNet18:
    """Build a CBAMPetResNet18 from a model config dict."""
    return CBAMPetResNet18(
        pretrained=model_cfg.get("pretrained", True),
        num_classes=model_cfg.get("num_classes", 37),
        reduction=model_cfg.get("cbam_reduction", 16),
        kernel_size=model_cfg.get("cbam_kernel_size", 7),
    )
