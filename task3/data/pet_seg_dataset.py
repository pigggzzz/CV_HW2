"""
Oxford-IIIT Pet semantic segmentation dataset and DataLoader factory.

Dataset homepage: https://www.robots.ox.ac.uk/~vgg/data/pets/

Expected directory layout:

    data/oxford_pets/
    ├── images/
    │   ├── Abyssinian_1.jpg
    │   └── ...
    └── annotations/
        ├── trimaps/
        │   ├── Abyssinian_1.png            # 1=pet, 2=background, 3=boundary
        │   └── ...
        ├── trainval.txt
        └── test.txt

The trimap is a PNG with three integer labels {1, 2, 3}. We remap them to
{0, 1, 2} = {foreground (pet), background, boundary} for the 3-class task.
"""

import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .splits import create_splits
from .transforms import get_train_joint_transforms, get_val_joint_transforms


# Class id → human-readable name (after remapping {1,2,3} → {0,1,2})
PET_SEG_CLASS_NAMES = ["foreground", "background", "boundary"]
NUM_SEG_CLASSES = 3


class OxfordPetSegDataset(Dataset):
    """
    PyTorch Dataset for Oxford-IIIT Pet segmentation.

    Args:
        samples: List of image stems (without extension) belonging to this split.
        images_dir: Directory containing JPEG images.
        masks_dir: Directory containing trimap PNGs.
        joint_transform: Callable that takes (PIL image, PIL mask) and returns
            (image tensor [C, H, W], mask tensor [H, W] of dtype long).
    """

    def __init__(
        self,
        samples: List[str],
        images_dir: str,
        masks_dir: str,
        joint_transform: Optional[Callable] = None,
    ) -> None:
        self.samples = samples
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.joint_transform = joint_transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        stem = self.samples[idx]
        img_path = self._resolve_image_path(stem)
        mask_path = os.path.join(self.masks_dir, stem + ".png")

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        # Trimaps are stored as a single-channel PNG. Convert to numpy first
        # (PIL "P" mode would otherwise apply a palette), then remap to 0/1/2.
        mask_arr = np.array(mask, dtype=np.int64)
        # Official trimap convention: 1=foreground, 2=background, 3=boundary.
        # Remap to a contiguous 0..C-1 label space.
        mask_arr = mask_arr - 1
        mask_arr = np.clip(mask_arr, 0, NUM_SEG_CLASSES - 1)
        mask = Image.fromarray(mask_arr.astype(np.uint8))

        if self.joint_transform is not None:
            image, mask = self.joint_transform(image, mask)

        return image, mask

    def _resolve_image_path(self, stem: str) -> str:
        for ext in (".jpg", ".jpeg", ".JPG", ".png"):
            candidate = os.path.join(self.images_dir, stem + ext)
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(
            f"Image for stem '{stem}' not found in {self.images_dir}"
        )


def build_dataloaders(
    data_root: str,
    batch_size: int = 16,
    image_size: int = 256,
    num_workers: int = 4,
    val_split: float = 0.15,
    seed: int = 42,
    pin_memory: bool = True,
) -> Dict[str, DataLoader]:
    """
    Build train / val / test DataLoaders for Oxford-IIIT Pet segmentation.

    Args:
        data_root: Root directory of the Oxford Pets dataset.
        batch_size: Mini-batch size.
        image_size: Spatial resolution (square crop).
        num_workers: Number of DataLoader workers.
        val_split: Fraction of the official trainval set used as validation.
        seed: Random seed for reproducible val split.
        pin_memory: Whether to pin memory in DataLoaders.

    Returns:
        Dict with keys 'train', 'val', 'test'.
    """
    images_dir = os.path.join(data_root, "images")
    masks_dir = os.path.join(data_root, "annotations", "trimaps")
    splits = create_splits(data_root, val_split=val_split, seed=seed)

    train_tf = get_train_joint_transforms(image_size)
    val_tf = get_val_joint_transforms(image_size)

    datasets = {
        "train": OxfordPetSegDataset(splits["train"], images_dir, masks_dir, train_tf),
        "val": OxfordPetSegDataset(splits["val"], images_dir, masks_dir, val_tf),
        "test": OxfordPetSegDataset(splits["test"], images_dir, masks_dir, val_tf),
    }

    loaders: Dict[str, DataLoader] = {}
    for split, dataset in datasets.items():
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=(split == "train"),
        )
    return loaders


def get_dataset_info(data_root: str, val_split: float = 0.15, seed: int = 42) -> dict:
    """Return a summary dict of dataset statistics."""
    splits = create_splits(data_root, val_split=val_split, seed=seed)
    return {
        "num_classes": NUM_SEG_CLASSES,
        "class_names": PET_SEG_CLASS_NAMES,
        "train_size": len(splits["train"]),
        "val_size": len(splits["val"]),
        "test_size": len(splits["test"]),
    }
