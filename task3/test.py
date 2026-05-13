#!/usr/bin/env python3
"""
Standalone test-set evaluation script.

Loads a saved checkpoint and evaluates on the Oxford-IIIT Pet segmentation
test set. Outputs:
  - pixel accuracy, mean IoU, per-class IoU (printed and JSON-saved);
  - per-class IoU bar chart (PNG);
  - qualitative triplet grid (image / GT / Pred);
  - overlay grid (image with mask).

Examples
--------
python test.py --config configs/baseline_ce.yaml \\
               --checkpoint outputs/checkpoints/task3_unet_ce/best.pth

python test.py --config configs/baseline_ce_dice.yaml \\
               --checkpoint outputs/checkpoints/task3_unet_ce_dice/best.pth \\
               --num-samples 8
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from data import (
    PET_SEG_CLASS_NAMES,
    NUM_SEG_CLASSES,
    build_dataloaders,
    denormalize,
)
from engine.evaluator import SegmentationEvaluator
from engine.losses import build_criterion
from models import build_model
from utils.config import load_config, merge_config_with_overrides
from utils.seed import set_seed
from utils.visualization import (
    plot_per_class_iou,
    save_overlay_grid,
    save_triplet_grid,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained segmentation checkpoint.")
    parser.add_argument("--config", "-c", required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", "-ck", required=True,
                        help="Path to checkpoint .pth file (best.pth).")
    parser.add_argument("--num-samples", type=int, default=8,
                        help="Number of qualitative samples to save.")
    parser.add_argument("--output-dir", default=None,
                        help="Override output directory (default: from config).")
    parser.add_argument("overrides", nargs="*",
                        help="Optional key=value config overrides.")
    return parser.parse_args()


def parse_overrides(override_list):
    result = {}
    for item in override_list:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        key, raw = item.split("=", 1)
        if raw.lower() in ("true", "false"):
            v = raw.lower() == "true"
        else:
            try:
                v = int(raw)
            except ValueError:
                try:
                    v = float(raw)
                except ValueError:
                    v = raw
        result[key] = v
    return result


def _denormed_to_uint8(image_tensor):
    img = denormalize(image_tensor.unsqueeze(0))[0].permute(1, 2, 0).cpu().numpy()
    return (img * 255.0).astype(np.uint8)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.overrides:
        cfg = merge_config_with_overrides(cfg, parse_overrides(args.overrides))
    if args.output_dir:
        cfg["experiment"]["output_dir"] = args.output_dir

    seed = int(cfg["experiment"].get("seed", 42))
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Data ----
    data_cfg = cfg["data"]
    dataloaders = build_dataloaders(
        data_root=data_cfg["data_root"],
        batch_size=int(data_cfg["batch_size"]),
        image_size=int(data_cfg.get("image_size", 256)),
        num_workers=int(data_cfg.get("num_workers", 4)),
        val_split=float(data_cfg.get("val_split", 0.15)),
        seed=seed,
        pin_memory=bool(data_cfg.get("pin_memory", True)),
    )

    # ---- Model + checkpoint ----
    model = build_model(cfg)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model_state"])
    model = model.to(device)
    print(f"Loaded checkpoint from epoch {state.get('epoch', '?')} "
          f"(val_metric={state.get('val_metric', 0):.4f} "
          f"{state.get('metric_name', 'mIoU')})")

    # ---- Evaluation ----
    num_classes = int(cfg["model"].get("num_classes", NUM_SEG_CLASSES))
    ignore_index = int(cfg.get("loss", {}).get("ignore_index", -100))
    criterion = build_criterion(cfg)
    evaluator = SegmentationEvaluator(
        model=model,
        loader=dataloaders["test"],
        criterion=criterion,
        device=device,
        num_classes=num_classes,
        ignore_index=ignore_index,
    )
    metrics = evaluator.evaluate()
    class_names = list(PET_SEG_CLASS_NAMES[:num_classes])

    print("=" * 50)
    print(f"  pixel accuracy : {metrics['pixel_acc']:.4f}")
    print(f"  mean IoU       : {metrics['mean_iou']:.4f}")
    print(f"  test loss      : {metrics['loss']:.4f}")
    print("=" * 50)
    print("Per-class IoU:")
    for name, iou in zip(class_names, metrics["per_class_iou"]):
        print(f"  [{name:<11s}] {iou:.4f}")

    # ---- Persist results ----
    output_dir = cfg["experiment"].get("output_dir", "outputs")
    exp_name = cfg["experiment"]["name"]
    log_dir = os.path.join(output_dir, "logs", exp_name)
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    result_path = os.path.join(log_dir, "test_results.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({
            "loss": float(metrics["loss"]),
            "pixel_acc": float(metrics["pixel_acc"]),
            "mean_iou": float(metrics["mean_iou"]),
            "per_class_iou": {
                name: float(v) for name, v in zip(class_names, metrics["per_class_iou"])
            },
        }, f, indent=2)
    print(f"Results saved to {result_path}")

    # ---- Per-class IoU bar chart ----
    plot_per_class_iou(
        metrics["per_class_iou"], class_names,
        save_path=os.path.join(figures_dir, f"{exp_name}_per_class_iou.png"),
        title=f"{exp_name} - per-class IoU (test)",
    )

    # ---- Qualitative samples ----
    samples = evaluator.collect_samples(max_samples=args.num_samples)
    triplet_data = []
    overlay_data = []
    for img_t, gt_t, pred_t in samples:
        img_np = _denormed_to_uint8(img_t)
        triplet_data.append((img_np, gt_t.numpy(), pred_t.numpy()))
        overlay_data.append((img_np, pred_t.numpy()))
    if triplet_data:
        save_triplet_grid(triplet_data,
                          save_path=os.path.join(figures_dir, f"{exp_name}_test_triplets.png"))
        save_overlay_grid(overlay_data,
                          save_path=os.path.join(figures_dir, f"{exp_name}_test_overlays.png"),
                          alpha=0.5)
        print(f"Visualisations saved under {figures_dir}/")


if __name__ == "__main__":
    main()
