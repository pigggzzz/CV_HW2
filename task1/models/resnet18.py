"""
Baseline ResNet-18 model for Oxford-IIIT Pet classification.

Key design decisions:
  - Uses torchvision's pretrained ResNet-18 as backbone.
  - Replaces the final FC layer with a new head for 37 classes.
  - Supports differential learning rates via get_param_groups().
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


class PetResNet18(nn.Module):
    """
    ResNet-18 adapted for Oxford-IIIT Pet 37-class classification.

    Args:
        pretrained: If True, loads ImageNet pretrained backbone weights.
        num_classes: Number of output classes.
    """

    def __init__(self, pretrained: bool = True, num_classes: int = 37) -> None:
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)

        # Extract all layers except the original FC head
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            backbone.avgpool,
        )
        in_features = backbone.fc.in_features  # 512 for ResNet-18

        # New randomly-initialised classification head
        self.classifier = nn.Linear(in_features, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def get_param_groups(self, backbone_lr: float, head_lr: float) -> list:
        """
        Return parameter groups with differential learning rates.

        The backbone receives a smaller lr to preserve pretrained features;
        the classification head receives a larger lr for faster adaptation.
        """
        return [
            {"params": self.backbone.parameters(), "lr": backbone_lr},
            {"params": self.classifier.parameters(), "lr": head_lr},
        ]


def build_resnet18(model_cfg: dict) -> PetResNet18:
    """Build a PetResNet18 from a model config dict."""
    return PetResNet18(
        pretrained=model_cfg.get("pretrained", True),
        num_classes=model_cfg.get("num_classes", 37),
    )
