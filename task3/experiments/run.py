"""
Single segmentation experiment runner.

Pipeline:
    1. Load YAML config (+ optional key=value overrides).
    2. Fix seed; build ExperimentLogger (wandb / swanlab / file).
    3. Build train / val / test DataLoaders.
    4. Build U-Net (from scratch, no pretrained weights).
    5. Build loss / optimizer / scheduler.
    6. Train -> validate per epoch -> save best by mIoU.
    7. Load best checkpoint -> evaluate on test set.
    8. Save training curves + qualitative figures.

Usage:
    python experiments/run.py --config configs/baseline_ce.yaml
    python experiments/run.py --config configs/baseline_dice.yaml training.epochs=5
"""

import argparse
import os
import sys
from typing import Dict

import numpy as np
import torch

# Allow running from project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import (  # noqa: E402
    PET_SEG_CLASS_NAMES,
    NUM_SEG_CLASSES,
    build_dataloaders,
    denormalize,
    get_dataset_info,
)
from engine import (  # noqa: E402
    CheckpointManager,
    SegmentationEvaluator,
    SegmentationTrainer,
    build_criterion,
)
from models import build_model  # noqa: E402
from utils.config import (  # noqa: E402
    load_config,
    merge_config_with_overrides,
    save_config,
)
from utils.logger import ExperimentLogger  # noqa: E402
from utils.seed import set_seed  # noqa: E402
from utils.visualization import (  # noqa: E402
    plot_per_class_iou,
    plot_training_curves,
    save_triplet_grid,
    save_overlay_grid,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run a single Oxford-Pet segmentation experiment.")
    parser.add_argument("--config", "-c", required=True,
                        help="Path to YAML config file (e.g. configs/baseline_ce.yaml)")
    parser.add_argument("overrides", nargs="*",
                        help="Optional key=value overrides, e.g. training.epochs=5 loss.type=dice")
    return parser.parse_args()


def parse_overrides(override_list: list) -> dict:
    """Convert ['key=value', ...] -> {'key': value} with type inference."""
    result = {}
    for item in override_list:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        key, raw = item.split("=", 1)
        if raw.lower() in ("true", "false"):
            value = raw.lower() == "true"
        else:
            try:
                value = int(raw)
            except ValueError:
                try:
                    value = float(raw)
                except ValueError:
                    value = raw
        result[key] = value
    return result


def _denormed_to_uint8(image_tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalised [C,H,W] tensor to HxWx3 uint8 for visualisation."""
    img = denormalize(image_tensor.unsqueeze(0))[0].permute(1, 2, 0).cpu().numpy()
    return (img * 255.0).astype(np.uint8)


def run_experiment(cfg: dict) -> Dict:
    """Execute a complete training + evaluation run for one config."""
    exp_cfg = cfg["experiment"]
    exp_name = exp_cfg["name"]
    seed = int(exp_cfg.get("seed", 42))
    output_dir = exp_cfg.get("output_dir", "outputs")

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_cfg = cfg.get("logging", {})
    logger = ExperimentLogger(
        experiment_name=exp_name,
        log_dir=os.path.join(output_dir, "logs"),
        cfg=cfg,
        use_wandb=bool(log_cfg.get("use_wandb", False)),
        use_swanlab=bool(log_cfg.get("use_swanlab", False)),
    )

    logger.info(f"Experiment : {exp_name}")
    logger.info(f"Device     : {device}")
    logger.info(f"Seed       : {seed}")

    # Save a copy of the resolved config next to the log.
    config_save_path = os.path.join(output_dir, "logs", exp_name, "config.yaml")
    save_config(cfg, config_save_path)
    logger.info(f"Config saved to {config_save_path}")

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
    info = get_dataset_info(data_cfg["data_root"],
                            val_split=float(data_cfg.get("val_split", 0.15)),
                            seed=seed)
    logger.info(
        f"Dataset splits - train: {info['train_size']}, "
        f"val: {info['val_size']}, test: {info['test_size']}, "
        f"classes={info['num_classes']} {info['class_names']}"
    )

    # ---- Model ----
    model = build_model(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model      : {cfg['model']['arch']}  ({n_params:,} params, all randomly initialised)")

    # ---- Checkpoint manager ----
    ckpt_manager = CheckpointManager(
        ckpt_dir=os.path.join(output_dir, "checkpoints"),
        experiment_name=exp_name,
        metric_name="mIoU",
    )

    # ---- Training ----
    trainer = SegmentationTrainer(
        model=model,
        dataloaders=dataloaders,
        cfg=cfg,
        device=device,
        logger=logger,
        ckpt_manager=ckpt_manager,
    )
    logger.info("Starting training ...")
    history = trainer.train()
    logger.save_history(history)

    # ---- Save training curve figure ----
    figures_dir = os.path.join(output_dir, "figures")
    curve_path = os.path.join(figures_dir, f"{exp_name}_curves.png")
    plot_training_curves(history, save_path=curve_path,
                         title=f"{exp_name} - training curves")
    logger.info(f"Training curves saved to {curve_path}")

    # ---- Test evaluation ----
    logger.info("Loading best checkpoint for test evaluation ...")
    ckpt_manager.load_best(model, device)

    criterion = build_criterion(cfg)
    test_evaluator = SegmentationEvaluator(
        model=model,
        loader=dataloaders["test"],
        criterion=criterion,
        device=device,
        num_classes=cfg["model"].get("num_classes", NUM_SEG_CLASSES),
        ignore_index=int(cfg.get("loss", {}).get("ignore_index", -100)),
    )
    test_metrics = test_evaluator.evaluate()
    class_names = list(PET_SEG_CLASS_NAMES[: cfg["model"].get("num_classes", NUM_SEG_CLASSES)])
    logger.log_test(test_metrics, class_names)

    # ---- Per-class IoU bar chart ----
    bar_path = os.path.join(figures_dir, f"{exp_name}_per_class_iou.png")
    plot_per_class_iou(
        test_metrics["per_class_iou"], class_names, bar_path,
        title=f"{exp_name} - per-class IoU (test)",
    )
    logger.info(f"Per-class IoU bar chart saved to {bar_path}")

    # ---- Qualitative samples (test set triplets + overlays) ----
    samples = test_evaluator.collect_samples(
        max_samples=int(log_cfg.get("num_vis_samples", 6))
    )
    triplet_data = []
    overlay_data = []
    for img_t, gt_t, pred_t in samples:
        img_np = _denormed_to_uint8(img_t)
        triplet_data.append((img_np, gt_t.numpy(), pred_t.numpy()))
        overlay_data.append((img_np, pred_t.numpy()))
    if triplet_data:
        trip_path = os.path.join(figures_dir, f"{exp_name}_test_triplets.png")
        save_triplet_grid(triplet_data, save_path=trip_path)
        logger.info(f"Test triplets saved to {trip_path}")

        over_path = os.path.join(figures_dir, f"{exp_name}_test_overlays.png")
        save_overlay_grid(overlay_data, save_path=over_path, alpha=0.5)
        logger.info(f"Test overlays saved to {over_path}")

    logger.finish()
    return {"history": history, "test_metrics": test_metrics}


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.overrides:
        cfg = merge_config_with_overrides(cfg, parse_overrides(args.overrides))
    run_experiment(cfg)


if __name__ == "__main__":
    main()
