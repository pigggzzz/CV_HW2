from .resnet18 import build_resnet18
from .se_resnet18 import build_se_resnet18
from .cbam_resnet18 import build_cbam_resnet18

MODEL_REGISTRY = {
    "resnet18": build_resnet18,
    "se_resnet18": build_se_resnet18,
    "cbam_resnet18": build_cbam_resnet18,
}


def build_model(cfg: dict):
    """
    Factory function that builds a model from a config dictionary.

    Expected keys in cfg['model']:
        arch, pretrained, num_classes, and optional attention params.
    """
    model_cfg = cfg["model"]
    arch = model_cfg["arch"]
    if arch not in MODEL_REGISTRY:
        raise ValueError(f"Unknown architecture: {arch}. Choose from {list(MODEL_REGISTRY)}")
    builder = MODEL_REGISTRY[arch]
    return builder(model_cfg)


__all__ = [
    "build_resnet18",
    "build_se_resnet18",
    "build_cbam_resnet18",
    "build_model",
    "MODEL_REGISTRY",
]
