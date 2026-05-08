"""
SE-ResNet18: ResNet-18 augmented with Squeeze-and-Excitation blocks.

SE blocks are inserted after each residual block's final BN layer,
recalibrating channel responses before the residual addition.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights
from torchvision.models.resnet import BasicBlock

from .se_block import SEBlock


class SEBasicBlock(nn.Module):
    """
    ResNet BasicBlock with a Squeeze-and-Excitation block appended.

    The SE block sits between the second BN and the residual addition,
    so the channel weights act on the residual branch only.
    """

    expansion = 1

    def __init__(
        self,
        original_block: BasicBlock,
        reduction: int = 16,
    ) -> None:
        super().__init__()
        # Copy all layers from the original BasicBlock
        self.conv1 = original_block.conv1
        self.bn1 = original_block.bn1
        self.relu = original_block.relu
        self.conv2 = original_block.conv2
        self.bn2 = original_block.bn2
        self.downsample = original_block.downsample
        self.stride = original_block.stride

        out_channels = original_block.conv2.out_channels
        self.se = SEBlock(out_channels, reduction=reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)              # SE recalibration on residual branch

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.relu(out + identity)
        return out


def _inject_se_blocks(layer: nn.Sequential, reduction: int) -> nn.Sequential:
    """Replace every BasicBlock in a ResNet layer with an SEBasicBlock."""
    new_blocks = []
    for block in layer:
        if isinstance(block, BasicBlock):
            new_blocks.append(SEBasicBlock(block, reduction=reduction))
        else:
            new_blocks.append(block)
    return nn.Sequential(*new_blocks)


class SEPetResNet18(nn.Module):
    """
    ResNet-18 with SE blocks for 37-class pet classification.

    Args:
        pretrained: Load ImageNet pretrained weights before injecting SE blocks.
        num_classes: Number of output classes.
        reduction: SE block reduction ratio.
    """

    def __init__(
        self,
        pretrained: bool = True,
        num_classes: int = 37,
        reduction: int = 16,
    ) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base = models.resnet18(weights=weights)

        # Stem layers (unchanged)
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool

        # Inject SE blocks into each residual stage
        self.layer1 = _inject_se_blocks(base.layer1, reduction)
        self.layer2 = _inject_se_blocks(base.layer2, reduction)
        self.layer3 = _inject_se_blocks(base.layer3, reduction)
        self.layer4 = _inject_se_blocks(base.layer4, reduction)

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


def build_se_resnet18(model_cfg: dict) -> SEPetResNet18:
    """Build an SEPetResNet18 from a model config dict."""
    return SEPetResNet18(
        pretrained=model_cfg.get("pretrained", True),
        num_classes=model_cfg.get("num_classes", 37),
        reduction=model_cfg.get("se_reduction", 16),
    )
