"""
Single-experiment runner.

Loads a YAML config, builds the model, dataloaders, trainer and logger,
runs the full training loop, then evaluates on the test set.

Usage:
    python experiments/run.py --config configs/baseline.yaml
    python experiments/run.py --config configs/baseline.yaml training.epochs=20
"""

import argparse
import os
import sys

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from data import build_dataloaders
from engine import CheckpointManager, Evaluator, Trainer
from engine.losses import build_criterion
from models import build_model
from utils.config import load_config, merge_config_with_overrides, save_config
from utils.logger import ExperimentLogger
from utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Run a single pet classification experiment.")
    parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to YAML config file (e.g. configs/baseline.yaml)",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional key=value overrides, e.g. training.epochs=20 model.pretrained=false",
    )
    return parser.parse_args()


def parse_overrides(override_list: list) -> dict:
    """Convert ['key=value', ...] → {'key': value} with type inference."""
    result = {}
    for item in override_list:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        key, raw_val = item.split("=", 1)
        # Type inference: bool, int, float, str
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


def run_experiment(cfg: dict) -> dict:
    """
    Execute a complete training + evaluation run.

    Args:
        cfg: Fully resolved experiment config dict.

    Returns:
        Dict containing training history and test metrics.
    """
    exp_cfg = cfg["experiment"]
    exp_name = exp_cfg["name"]
    seed = exp_cfg.get("seed", 42)
    output_dir = exp_cfg.get("output_dir", "outputs")

    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_cfg = cfg.get("logging", {})
    logger = ExperimentLogger(
        experiment_name=exp_name,
        log_dir=os.path.join(output_dir, "logs"),
        cfg=cfg,
        use_wandb=log_cfg.get("use_wandb", False),
        use_swanlab=log_cfg.get("use_swanlab", False),
    )

    logger.info(f"Experiment : {exp_name}")
    logger.info(f"Device     : {device}")
    logger.info(f"Seed       : {seed}")

    # Save a copy of the resolved config next to the log
    config_save_path = os.path.join(output_dir, "logs", exp_name, "config.yaml")
    save_config(cfg, config_save_path)
    logger.info(f"Config saved to {config_save_path}")

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
    logger.info(
        f"Dataset splits — "
        f"train: {len(dataloaders['train'].dataset)}, "
        f"val: {len(dataloaders['val'].dataset)}, "
        f"test: {len(dataloaders['test'].dataset)}"
    )

    # ---- Model ----
    model = build_model(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model      : {cfg['model']['arch']}  ({n_params:,} params, {n_trainable:,} trainable)")

    # ---- Checkpoint manager ----
    ckpt_manager = CheckpointManager(
        ckpt_dir=os.path.join(output_dir, "checkpoints"),
        experiment_name=exp_name,
    )

    # ---- Training ----
    trainer = Trainer(
        model=model,
        dataloaders=dataloaders,
        cfg=cfg,
        device=device,
        logger=logger,
        ckpt_manager=ckpt_manager,
    )

    logger.info("Starting training …")
    history = trainer.train()
    logger.save_history(history)

    # ---- Test evaluation ----
    logger.info("Loading best checkpoint for test evaluation …")
    ckpt_manager.load_best(model, device)

    criterion = build_criterion(cfg)
    evaluator = Evaluator(
        model=model,
        loader=dataloaders["test"],
        criterion=criterion,
        device=device,
        num_classes=cfg["model"].get("num_classes", 37),
    )
    test_metrics = evaluator.evaluate()
    logger.log_test(test_metrics)
    logger.finish()

    return {"history": history, "test_metrics": test_metrics}


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.overrides:
        overrides = parse_overrides(args.overrides)
        cfg = merge_config_with_overrides(cfg, overrides)
    run_experiment(cfg)


if __name__ == "__main__":
    main()
