"""
Oxford-IIIT Pet Dataset wrapper and DataLoader factory.

Dataset homepage: https://www.robots.ox.ac.uk/~vgg/data/pets/
Expected directory layout after download and extraction:

    data/oxford_pets/
    ├── images/
    │   ├── Abyssinian_1.jpg
    │   └── ...
    └── annotations/
        ├── list.txt
        ├── trainval.txt
        └── test.txt
"""

import os
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .splits import create_splits, get_class_names
from .transforms import get_train_transforms, get_val_transforms


class OxfordPetsDataset(Dataset):
    """
    PyTorch Dataset for the Oxford-IIIT Pet Dataset.

    Args:
        samples: List of (image_name, class_id) tuples.
        images_dir: Directory containing JPEG images.
        transform: Optional transform applied to each PIL image.
    """

    def __init__(
        self,
        samples: List[Tuple[str, int]],
        images_dir: str,
        transform: Optional[Callable] = None,
    ) -> None:
        self.samples = samples
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_name, label = self.samples[idx]
        img_path = os.path.join(self.images_dir, image_name + ".jpg")

        # Some images may have uppercase extension
        if not os.path.exists(img_path):
            img_path = os.path.join(self.images_dir, image_name + ".JPG")

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @property
    def num_classes(self) -> int:
        return len(set(s[1] for s in self.samples))


def build_dataloaders(
    data_root: str,
    batch_size: int = 64,
    image_size: int = 224,
    num_workers: int = 4,
    val_split: float = 0.2,
    seed: int = 42,
    pin_memory: bool = True,
) -> Dict[str, DataLoader]:
    """
    Build train, val, and test DataLoaders for Oxford-IIIT Pets.

    Args:
        data_root: Root directory of the Oxford Pets dataset.
        batch_size: Mini-batch size for train and val loaders.
        image_size: Spatial resolution to resize images to.
        num_workers: Number of DataLoader worker processes.
        val_split: Fraction of trainval to use as validation.
        seed: Random seed for reproducible splits.
        pin_memory: Whether to pin memory in DataLoaders.

    Returns:
        Dictionary with keys 'train', 'val', 'test', each a DataLoader.
    """
    images_dir = os.path.join(data_root, "images")
    splits = create_splits(data_root, val_split=val_split, seed=seed)

    train_tf = get_train_transforms(image_size)
    val_tf = get_val_transforms(image_size)

    datasets = {
        "train": OxfordPetsDataset(splits["train"], images_dir, transform=train_tf),
        "val": OxfordPetsDataset(splits["val"], images_dir, transform=val_tf),
        "test": OxfordPetsDataset(splits["test"], images_dir, transform=val_tf),
    }

    loaders: Dict[str, DataLoader] = {}
    for split, dataset in datasets.items():
        shuffle = split == "train"
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=split == "train",  # drop incomplete last batch for training
        )

    return loaders


def get_dataset_info(data_root: str, val_split: float = 0.2, seed: int = 42) -> dict:
    """Return a summary dict of dataset statistics."""
    splits = create_splits(data_root, val_split=val_split, seed=seed)
    class_names = get_class_names(data_root)
    return {
        "num_classes": 37,
        "class_names": class_names,
        "train_size": len(splits["train"]),
        "val_size": len(splits["val"]),
        "test_size": len(splits["test"]),
    }
