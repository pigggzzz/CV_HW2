#!/usr/bin/env python3
"""
Training entrypoint for task3 - Oxford-IIIT Pet semantic segmentation.

Thin wrapper around experiments/run.py that can be invoked from the project
root. Supports config-file selection plus arbitrary key=value overrides.

Examples
--------
# CE-only baseline
python train.py --config configs/baseline_ce.yaml

# Dice-only baseline
python train.py --config configs/baseline_dice.yaml

# Combined CE + Dice
python train.py --config configs/baseline_ce_dice.yaml

# Quick override: fewer epochs
python train.py --config configs/baseline_ce.yaml training.epochs=5 logging.use_wandb=false
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.run import main

if __name__ == "__main__":
    main()
