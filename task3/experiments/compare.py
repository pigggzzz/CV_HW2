"""
Compare the three loss-configuration experiments and produce summary figures.

Inputs (auto-discovered under outputs/logs/<exp_name>/):
    - history.json      : training curves per experiment
    - test_results.json : final test metrics per experiment

Outputs (written to outputs/figures/):
    - compare_val_loss.png
    - compare_val_miou.png
    - compare_train_loss.png
    - compare_train_miou.png
    - compare_per_class_iou.png
    - compare_summary.json
"""

import argparse
import json
import os
import sys
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import PET_SEG_CLASS_NAMES  # noqa: E402
from utils.visualization import (  # noqa: E402
    plot_loss_comparison,
    plot_per_class_iou_comparison,
)


DEFAULT_EXPERIMENTS = [
    "task3_unet_ce",
    "task3_unet_dice",
    "task3_unet_ce_dice",
]


def _load_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare task3 loss-ablation experiments.")
    parser.add_argument("--logs-dir", default="outputs/logs",
                        help="Directory containing per-experiment subfolders.")
    parser.add_argument("--figures-dir", default="outputs/figures",
                        help="Where to write comparison PNGs.")
    parser.add_argument("--experiments", nargs="*", default=DEFAULT_EXPERIMENTS,
                        help="Experiment names to compare.")
    return parser.parse_args()


def main():
    args = parse_args()

    histories: Dict[str, dict] = {}
    test_results: Dict[str, dict] = {}
    for name in args.experiments:
        h = _load_json(os.path.join(args.logs_dir, name, "history.json"))
        t = _load_json(os.path.join(args.logs_dir, name, "test_results.json"))
        if h is None:
            print(f"[warn] history.json missing for {name}; skipping.")
            continue
        histories[name] = h
        if t is not None:
            test_results[name] = t

    if not histories:
        print("No histories found. Did you run the three experiments?")
        return

    os.makedirs(args.figures_dir, exist_ok=True)

    # Curve comparison plots.
    plot_loss_comparison(
        histories, "val_loss",
        save_path=os.path.join(args.figures_dir, "compare_val_loss.png"),
        ylabel="val loss",
        title="Validation loss vs epoch",
    )
    plot_loss_comparison(
        histories, "val_miou",
        save_path=os.path.join(args.figures_dir, "compare_val_miou.png"),
        ylabel="val mIoU",
        title="Validation mIoU vs epoch",
    )
    plot_loss_comparison(
        histories, "train_loss",
        save_path=os.path.join(args.figures_dir, "compare_train_loss.png"),
        ylabel="train loss",
        title="Training loss vs epoch",
    )
    plot_loss_comparison(
        histories, "train_miou",
        save_path=os.path.join(args.figures_dir, "compare_train_miou.png"),
        ylabel="train mIoU",
        title="Training mIoU vs epoch",
    )

    # Per-class IoU bar comparison (test set).
    if test_results:
        class_names = PET_SEG_CLASS_NAMES
        per_class = {
            name: [tr["per_class_iou"].get(c, 0.0) for c in class_names]
            for name, tr in test_results.items()
        }
        plot_per_class_iou_comparison(
            per_class, class_names,
            save_path=os.path.join(args.figures_dir, "compare_per_class_iou.png"),
            title="Per-class IoU on test set",
        )

    # Numerical summary.
    summary = {}
    for name in histories:
        h = histories[name]
        t = test_results.get(name, {})
        summary[name] = {
            "best_val_miou": max(h.get("val_miou", [0.0])) if h.get("val_miou") else 0.0,
            "final_val_loss": h.get("val_loss", [None])[-1] if h.get("val_loss") else None,
            "test_loss": t.get("loss"),
            "test_pixel_acc": t.get("pixel_acc"),
            "test_mean_iou": t.get("mean_iou"),
            "test_per_class_iou": t.get("per_class_iou"),
        }

    summary_path = os.path.join(args.figures_dir, "compare_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Comparison summary written to {summary_path}")

    print("\nResults table:")
    print(f"{'experiment':<24s}  {'best_val_mIoU':>12s}  {'test_mIoU':>10s}  {'test_pAcc':>10s}")
    for name, row in summary.items():
        print(
            f"{name:<24s}  "
            f"{row['best_val_miou']:>12.4f}  "
            f"{(row['test_mean_iou'] or 0.0):>10.4f}  "
            f"{(row['test_pixel_acc'] or 0.0):>10.4f}"
        )


if __name__ == "__main__":
    main()
