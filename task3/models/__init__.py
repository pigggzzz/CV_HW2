from .blocks import DoubleConv, Down, Up, OutConv
from .unet import UNet, build_unet


MODEL_REGISTRY = {
    "unet": build_unet,
}


def build_model(cfg: dict):
    """
    Factory function that builds a segmentation model from a config dictionary.

    Expected keys in cfg['model']:
        arch, in_channels, num_classes, base_channels, bilinear.
    """
    model_cfg = cfg["model"]
    arch = model_cfg["arch"]
    if arch not in MODEL_REGISTRY:
        raise ValueError(f"Unknown architecture: {arch}. Choose from {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[arch](model_cfg)


__all__ = [
    "DoubleConv",
    "Down",
    "Up",
    "OutConv",
    "UNet",
    "build_unet",
    "build_model",
    "MODEL_REGISTRY",
]
