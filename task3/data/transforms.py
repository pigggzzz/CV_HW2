"""
Joint image / mask transforms.

Geometric transforms (Resize, Crop, Flip, ...) MUST be applied with identical
random parameters to both the image and its segmentation mask. Photometric
transforms (ColorJitter, Normalize) are applied to the image only.

All transforms accept and return a tuple `(image, mask)` where:
  - image: PIL.Image (RGB) before ToTensor, torch.Tensor after.
  - mask : PIL.Image (mode 'L') before ToMaskTensor, torch.LongTensor after.
"""

import random
from typing import List, Tuple

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import ColorJitter


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Composition primitive
# ---------------------------------------------------------------------------


class JointCompose:
    """Compose multiple joint transforms in order."""

    def __init__(self, transforms: List):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


# ---------------------------------------------------------------------------
# Geometric transforms (applied to both image and mask)
# ---------------------------------------------------------------------------


class JointResize:
    """Resize image (bilinear) and mask (nearest) to the same target size."""

    def __init__(self, size: int):
        self.size = (size, size)

    def __call__(self, image, mask):
        image = TF.resize(image, self.size, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.size, interpolation=TF.InterpolationMode.NEAREST)
        return image, mask


class JointRandomResizedCrop:
    """
    Random resized crop applied identically to image and mask.

    Mirrors torchvision RandomResizedCrop but uses NEAREST for the mask.
    """

    def __init__(self, size: int, scale=(0.6, 1.0), ratio=(0.75, 1.333)):
        self.size = (size, size)
        self.scale = scale
        self.ratio = ratio

    def __call__(self, image, mask):
        from torchvision.transforms import RandomResizedCrop
        i, j, h, w = RandomResizedCrop.get_params(image, self.scale, self.ratio)
        image = TF.resized_crop(
            image, i, j, h, w, self.size, interpolation=TF.InterpolationMode.BILINEAR
        )
        mask = TF.resized_crop(
            mask, i, j, h, w, self.size, interpolation=TF.InterpolationMode.NEAREST
        )
        return image, mask


class JointRandomHorizontalFlip:
    """Random horizontal flip applied to both."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        return image, mask


# ---------------------------------------------------------------------------
# Photometric transforms (applied to image only)
# ---------------------------------------------------------------------------


class JointColorJitter:
    """Color jitter applied to the image only; mask untouched."""

    def __init__(self, brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05):
        self.jitter = ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
        )

    def __call__(self, image, mask):
        return self.jitter(image), mask


# ---------------------------------------------------------------------------
# Tensorization / normalisation
# ---------------------------------------------------------------------------


class JointToTensor:
    """
    Convert PIL image → float tensor [0, 1], and PIL mask → long tensor.

    The mask is converted via numpy to preserve integer class ids.
    """

    def __call__(self, image, mask):
        image_t = TF.to_tensor(image)  # [C, H, W] in [0, 1]
        # PIL "L" → torch.from_numpy preserves integers; we already remapped to {0,1,2}.
        import numpy as np
        mask_arr = np.array(mask, dtype="int64")
        mask_t = torch.from_numpy(mask_arr).long()
        return image_t, mask_t


class JointNormalize:
    """Normalize the image with the given mean/std; mask untouched."""

    def __init__(self, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask):
        image = TF.normalize(image, mean=self.mean, std=self.std)
        return image, mask


# ---------------------------------------------------------------------------
# Composed pipelines used by the dataset
# ---------------------------------------------------------------------------


def get_train_joint_transforms(image_size: int = 256) -> JointCompose:
    """Train-time augmentation: random crop + flip + color jitter + normalise."""
    return JointCompose([
        JointRandomResizedCrop(image_size, scale=(0.6, 1.0)),
        JointRandomHorizontalFlip(p=0.5),
        JointColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        JointToTensor(),
        JointNormalize(),
    ])


def get_val_joint_transforms(image_size: int = 256) -> JointCompose:
    """Validation/test-time: deterministic resize + normalise."""
    return JointCompose([
        JointResize(image_size),
        JointToTensor(),
        JointNormalize(),
    ])


def denormalize(image_tensor: torch.Tensor) -> torch.Tensor:
    """
    Invert ImageNet normalisation; useful for visualisation.

    Accepts [C, H, W] or [N, C, H, W] tensors in normalised space and returns
    a tensor in [0, 1] (clipped).
    """
    mean = torch.tensor(IMAGENET_MEAN, device=image_tensor.device).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=image_tensor.device).view(-1, 1, 1)
    if image_tensor.dim() == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    return (image_tensor * std + mean).clamp(0.0, 1.0)
