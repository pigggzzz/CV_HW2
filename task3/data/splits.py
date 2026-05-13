"""
Dataset split utilities for Oxford-IIIT Pet segmentation.

The official annotation files (`trainval.txt`, `test.txt`) list image stems
along with a 1-indexed breed class id, species id and breed id. For the
segmentation task we only need the image stem; we further split `trainval`
into train / val.
"""

import os
import random
from typing import Dict, List


def _parse_stems(filepath: str) -> List[str]:
    """Read the first column of an Oxford Pets annotation file as image stems."""
    stems: List[str] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            stems.append(line.split()[0])
    return stems


def create_splits(
    data_root: str,
    val_split: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """
    Build train / val / test splits as lists of image stems.

    Args:
        data_root: Root dir containing `annotations/{trainval,test}.txt`.
        val_split: Fraction of trainval reserved as validation.
        seed: Random seed for the val carve-out (reproducible).

    Returns:
        Dict with keys 'train', 'val', 'test'.
    """
    ann_dir = os.path.join(data_root, "annotations")
    trainval_file = os.path.join(ann_dir, "trainval.txt")
    test_file = os.path.join(ann_dir, "test.txt")

    if not os.path.exists(trainval_file):
        raise FileNotFoundError(
            f"trainval.txt not found at {trainval_file}.\n"
            "Please download Oxford-IIIT Pet first:\n"
            "  wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz\n"
            "  wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"
        )

    trainval = _parse_stems(trainval_file)
    test = _parse_stems(test_file) if os.path.exists(test_file) else []

    rng = random.Random(seed)
    shuffled = list(trainval)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_split))
    val = shuffled[:n_val]
    train = shuffled[n_val:]

    return {"train": train, "val": val, "test": test}
