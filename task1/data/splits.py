"""
Dataset split utilities for Oxford-IIIT Pet Dataset.

The official dataset provides:
  - trainval.txt  (all training + validation samples)
  - test.txt      (held-out test samples)

We further split trainval into train / val subsets.
"""

import os
import random
from typing import Dict, List, Tuple


def parse_annotation_file(filepath: str) -> List[Tuple[str, int]]:
    """
    Parse an Oxford Pets annotation txt file.

    Each non-comment line has the format:
        <image_name> <class_id> <species> <breed_id>

    Returns a list of (image_name, class_id) tuples where class_id is 1-indexed.
    We convert to 0-indexed internally.
    """
    samples: List[Tuple[str, int]] = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            image_name = parts[0]
            class_id = int(parts[1]) - 1  # convert to 0-indexed
            samples.append((image_name, class_id))
    return samples


def create_splits(
    data_root: str,
    val_split: float = 0.2,
    seed: int = 42,
) -> Dict[str, List[Tuple[str, int]]]:
    """
    Create train / val / test splits from the Oxford Pets annotation files.

    Args:
        data_root: Root directory containing the 'annotations' subfolder.
        val_split: Fraction of the official trainval set to use as validation.
        seed: Random seed for reproducible splits.

    Returns:
        Dictionary with keys 'train', 'val', 'test', each mapping to a list of
        (image_name, class_id) tuples.
    """
    annotations_dir = os.path.join(data_root, "annotations")
    trainval_file = os.path.join(annotations_dir, "trainval.txt")
    test_file = os.path.join(annotations_dir, "test.txt")

    if not os.path.exists(trainval_file):
        raise FileNotFoundError(
            f"trainval.txt not found at {trainval_file}. "
            "Please download the Oxford-IIIT Pet Dataset first.\n"
            "  wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz\n"
            "  wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"
        )

    trainval = parse_annotation_file(trainval_file)
    test = parse_annotation_file(test_file)

    # Stratified split: split per class to maintain class distribution
    rng = random.Random(seed)
    class_to_samples: Dict[int, List[Tuple[str, int]]] = {}
    for sample in trainval:
        cls = sample[1]
        class_to_samples.setdefault(cls, []).append(sample)

    train_samples: List[Tuple[str, int]] = []
    val_samples: List[Tuple[str, int]] = []

    for cls_samples in class_to_samples.values():
        shuffled = list(cls_samples)
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * val_split))
        val_samples.extend(shuffled[:n_val])
        train_samples.extend(shuffled[n_val:])

    return {
        "train": train_samples,
        "val": val_samples,
        "test": test,
    }


def get_class_names(data_root: str) -> List[str]:
    """
    Return a list of class names ordered by class id (0-indexed).

    The class names are derived from the image filenames:
    e.g. 'Abyssinian_1.jpg' → class 'Abyssinian'.
    The annotations list.txt maps numeric IDs to breed names.
    """
    list_file = os.path.join(data_root, "annotations", "list.txt")
    if not os.path.exists(list_file):
        return [str(i) for i in range(37)]

    id_to_name: Dict[int, str] = {}
    with open(list_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            image_name = parts[0]
            class_id = int(parts[1]) - 1  # 0-indexed
            # breed name is the image_name up to the last underscore
            breed = "_".join(image_name.split("_")[:-1])
            id_to_name[class_id] = breed

    # Build sorted list
    max_id = max(id_to_name.keys()) if id_to_name else 36
    return [id_to_name.get(i, str(i)) for i in range(max_id + 1)]
