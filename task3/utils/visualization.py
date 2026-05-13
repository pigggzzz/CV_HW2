"""
Visualisation helpers for semantic segmentation outputs.

Functions provided:
    - decode_mask(mask, palette)       : int label map -> RGB image.
    - make_triplet(image, gt, pred)    : 3-up panel (image | GT | Pred).
    - make_overlay(image, mask, alpha) : image with semi-transparent mask overlay.
    - save_triplet_grid(samples, ...)  : save a grid of triplets to disk.
    - plot_per_class_iou(...)          : bar chart of per-class IoU.
    - plot_training_curves(history, ...): loss + mIoU curves.
    - plot_loss_comparison(histories,...): compare metric across loss types.
"""

import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# A fixed colour palette for the 3-class trimap.
# (foreground=red, background=green, boundary=yellow)
DEFAULT_PALETTE = np.array(
    [
        [220, 50, 50],     # 0 - foreground (pet)
        [50, 180, 80],     # 1 - background
        [255, 230, 0],     # 2 - boundary
    ],
    dtype=np.uint8,
)


# ---------------------------------------------------------------------------
# Low-level rendering helpers
# ---------------------------------------------------------------------------


def decode_mask(mask: np.ndarray, palette: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Map an integer label image to RGB using a colour palette.

    Args:
        mask:    np.ndarray of shape [H, W] with integer class ids.
        palette: np.ndarray of shape [C, 3] uint8; defaults to DEFAULT_PALETTE.

    Returns:
        np.ndarray of shape [H, W, 3] uint8.
    """
    if palette is None:
        palette = DEFAULT_PALETTE
    mask = np.clip(mask.astype(np.int64), 0, palette.shape[0] - 1)
    return palette[mask]


def _to_uint8_image(image: np.ndarray) -> np.ndarray:
    """Ensure an image is in [0,255] uint8 HxWx3."""
    img = np.asarray(image)
    if img.dtype != np.uint8:
        img = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    return img


def make_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    palette: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Blend a class mask on top of the original image.

    Args:
        image: np.ndarray [H, W, 3] uint8 or float in [0,1].
        mask:  np.ndarray [H, W] int class ids.
        alpha: blending factor for the mask (0 = no mask, 1 = mask only).

    Returns:
        np.ndarray [H, W, 3] uint8.
    """
    img = _to_uint8_image(image).astype(np.float32)
    color = decode_mask(mask, palette).astype(np.float32)
    out = (1 - alpha) * img + alpha * color
    return np.clip(out, 0, 255).astype(np.uint8)


def make_triplet(
    image: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    palette: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Build a horizontal 3-panel image: [original | GT mask | Pred mask].

    Returns: np.ndarray [H, 3*W, 3] uint8.
    """
    img = _to_uint8_image(image)
    gt_rgb = decode_mask(gt_mask, palette)
    pr_rgb = decode_mask(pred_mask, palette)
    return np.concatenate([img, gt_rgb, pr_rgb], axis=1)


# ---------------------------------------------------------------------------
# Saving multi-sample grids
# ---------------------------------------------------------------------------


def save_triplet_grid(
    samples: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    save_path: str,
    palette: Optional[np.ndarray] = None,
    titles: Tuple[str, str, str] = ("Image", "GT", "Prediction"),
) -> None:
    """
    Save a vertical stack of triplets, one row per sample, with column titles.

    Args:
        samples:   sequence of (image, gt_mask, pred_mask) numpy arrays.
        save_path: PNG file path. Parent dir is created if needed.
    """
    import matplotlib.pyplot as plt

    n = len(samples)
    if n == 0:
        return
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = np.array([axes])

    for r, (img, gt, pred) in enumerate(samples):
        axes[r, 0].imshow(_to_uint8_image(img))
        axes[r, 1].imshow(decode_mask(gt, palette))
        axes[r, 2].imshow(decode_mask(pred, palette))
        for c in range(3):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
        if r == 0:
            for c, title in enumerate(titles):
                axes[r, c].set_title(title, fontsize=11)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_overlay_grid(
    samples: Sequence[Tuple[np.ndarray, np.ndarray]],
    save_path: str,
    alpha: float = 0.5,
    palette: Optional[np.ndarray] = None,
) -> None:
    """Save a grid of (image, mask) overlays (one per row)."""
    import matplotlib.pyplot as plt

    n = len(samples)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axes = [axes]
    for ax, (img, mask) in zip(axes, samples):
        ax.imshow(make_overlay(img, mask, alpha=alpha, palette=palette))
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Metric plots
# ---------------------------------------------------------------------------


def plot_per_class_iou(
    per_class_iou: Sequence[float],
    class_names: Sequence[str],
    save_path: str,
    title: str = "Per-class IoU",
) -> None:
    """Bar chart of per-class IoU values."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(per_class_iou))
    bars = ax.bar(x, per_class_iou, color=["#dc3232", "#32b450", "#ffe600"][: len(x)])
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=0)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("IoU")
    ax.set_title(title)
    for b, v in zip(bars, per_class_iou):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(
    history: Dict[str, list],
    save_path: str,
    title: Optional[str] = None,
) -> None:
    """Plot train/val loss + train/val mIoU over epochs."""
    import matplotlib.pyplot as plt

    epochs = list(range(1, len(history.get("train_loss", [])) + 1))
    if not epochs:
        return

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, history["train_loss"], label="train loss", color="tab:red")
    ax1.plot(epochs, history.get("val_loss", []), label="val loss",
             color="tab:red", linestyle="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    if "train_miou" in history:
        ax2.plot(epochs, history["train_miou"], label="train mIoU", color="tab:blue")
    if "val_miou" in history:
        ax2.plot(epochs, history["val_miou"], label="val mIoU",
                 color="tab:blue", linestyle="--")
    ax2.set_ylabel("mIoU", color="tab:blue")
    ax2.set_ylim(0, 1.0)
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    # Merge legends from both axes.
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=9)

    if title:
        plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_loss_comparison(
    histories: Dict[str, Dict[str, list]],
    metric: str,
    save_path: str,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
) -> None:
    """
    Overlay the same metric (e.g. 'val_miou') from multiple experiments.

    Args:
        histories: {experiment_name: history_dict}.
        metric:    key inside each history dict (e.g. "val_miou", "val_loss").
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, hist in histories.items():
        values = hist.get(metric, [])
        if not values:
            continue
        ax.plot(range(1, len(values) + 1), values, label=name, linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel or metric)
    if title:
        ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_iou_comparison(
    results: Dict[str, Sequence[float]],
    class_names: Sequence[str],
    save_path: str,
    title: str = "Per-class IoU comparison",
) -> None:
    """Grouped bar chart of per-class IoU across multiple experiments."""
    import matplotlib.pyplot as plt

    exp_names = list(results.keys())
    n_exp = len(exp_names)
    n_cls = len(class_names)
    if n_exp == 0 or n_cls == 0:
        return

    x = np.arange(n_cls)
    width = 0.8 / n_exp
    fig, ax = plt.subplots(figsize=(2 + 1.5 * n_cls, 4))
    colours = ["#3279ff", "#dc3232", "#32b450", "#ff9900"]
    for i, name in enumerate(exp_names):
        vals = list(results[name]) + [0.0] * (n_cls - len(results[name]))
        ax.bar(x + i * width - 0.4 + width / 2, vals[:n_cls], width,
               label=name, color=colours[i % len(colours)])
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("IoU")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
