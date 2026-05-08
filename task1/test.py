#!/usr/bin/env python3
"""
Standalone test-set evaluation script.

Loads a saved checkpoint and evaluates on the Oxford-IIIT Pet test set.
Outputs top-1 / top-5 accuracy, per-class accuracy, and optionally saves
a confusion matrix figure.

Examples
--------
# Evaluate the best checkpoint from the baseline experiment
python test.py --config configs/baseline.yaml \\
               --checkpoint outputs/checkpoints/E1_baseline/best.pth

# Save confusion matrix figure
python test.py --config configs/baseline.yaml \\
               --checkpoint outputs/checkpoints/E1_baseline/best.pth \\
               --save-cm
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from data import build_dataloaders
from engine.evaluator import Evaluator
from engine.losses import build_criterion
from models import build_model
from utils.config import load_config, merge_config_with_overrides
from utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the test set.")
    parser.add_argument("--config", "-c", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--checkpoint", "-ck",
        required=True,
        help="Path to model checkpoint (.pth).",
    )
    parser.add_argument(
        "--save-cm",
        action="store_true",
        help="Save confusion matrix as a PNG figure.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory for results (default: from config).",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional key=value config overrides.",
    )
    return parser.parse_args()


def parse_overrides(override_list: list) -> dict:
    result = {}
    for item in override_list:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        key, raw_val = item.split("=", 1)
        if raw_val.lower() in ("true", "false"):
            value = raw_val.lower() == "true"
        else:
            try:
                value = int(raw_val)
            except ValueError:
                try:
                    value = float(raw_val)
                except ValueError:
                    value = raw_val
        result[key] = value
    return result


def save_confusion_matrix(cm: torch.Tensor, class_names: list, save_path: str) -> None:
    """Render and save a confusion matrix heatmap."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(14, 12))
        data = cm.numpy().astype(float)
        # Normalise rows to percentages
        row_sums = data.sum(axis=1, keepdims=True)
        data_pct = data / (row_sums + 1e-8) * 100

        im = ax.imshow(data_pct, cmap="Blues", vmin=0, vmax=100)
        plt.colorbar(im, ax=ax, label="Accuracy (%)")

        n = len(class_names)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(class_names, rotation=90, fontsize=6)
        ax.set_yticklabels(class_names, fontsize=6)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix (row-normalised %)")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Confusion matrix saved to {save_path}")
    except ImportError:
        print("matplotlib not available; skipping confusion matrix plot.")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.overrides:
        from experiments.run import parse_overrides as _parse
        cfg = merge_config_with_overrides(cfg, _parse(args.overrides))

    if args.output_dir:
        cfg["experiment"]["output_dir"] = args.output_dir

    seed = cfg["experiment"].get("seed", 42)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Data ----
    data_cfg = cfg["data"]
    dataloaders = build_dataloaders(
        data_root=data_cfg["data_root"],
        batch_size=int(data_cfg["batch_size"]),
        image_size=int(data_cfg.get("image_size", 224)),
        num_workers=int(data_cfg.get("num_workers", 4)),
        val_split=float(data_cfg.get("val_split", 0.2)),
        seed=seed,
        pin_memory=bool(data_cfg.get("pin_memory", True)),
    )

    # ---- Model + checkpoint ----
    model = build_model(cfg)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model_state"])
    model = model.to(device)
    print(f"Loaded checkpoint from epoch {state.get('epoch', '?')}  "
          f"(val_acc={state.get('val_acc', 0):.2f}%)")

    # ---- Evaluation ----
    num_classes = cfg["model"].get("num_classes", 37)
    criterion = build_criterion(cfg)
    evaluator = Evaluator(
        model=model,
        loader=dataloaders["test"],
        criterion=criterion,
        device=device,
        num_classes=num_classes,
    )
    metrics = evaluator.evaluate()

    print("=" * 50)
    print(f"  top-1 accuracy : {metrics['top1']:.2f}%")
    print(f"  top-5 accuracy : {metrics['top5']:.2f}%")
    print(f"  test loss      : {metrics['loss']:.4f}")
    print("=" * 50)

    # Per-class accuracy
    print("\nPer-class accuracy:")
    from data.splits import get_class_names
    class_names = get_class_names(data_cfg["data_root"])
    for i, acc in enumerate(metrics["per_class_acc"]):
        name = class_names[i] if i < len(class_names) else str(i)
        print(f"  [{i:2d}] {name:<30s} {acc:.1f}%")

    # Save results
    output_dir = cfg["experiment"].get("output_dir", "outputs")
    exp_name = cfg["experiment"]["name"]
    result_dir = os.path.join(output_dir, "logs", exp_name)
    os.makedirs(result_dir, exist_ok=True)

    result_path = os.path.join(result_dir, "test_results.json")
    with open(result_path, "w") as f:
        json.dump(
            {k: v for k, v in metrics.items() if k != "per_class_acc"},
            f, indent=2,
        )
    print(f"\nResults saved to {result_path}")

    # Confusion matrix
    if args.save_cm:
        cm = evaluator.confusion_matrix()
        cm_path = os.path.join(output_dir, "figures", f"{exp_name}_confusion_matrix.png")
        os.makedirs(os.path.dirname(cm_path), exist_ok=True)
        save_confusion_matrix(cm, class_names, cm_path)


if __name__ == "__main__":
    main()
