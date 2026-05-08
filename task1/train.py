#!/usr/bin/env python3
"""
Training entrypoint.

Thin wrapper around experiments/run.py that can be invoked directly
from the project root.  Supports config-file selection plus arbitrary
key=value overrides for quick ablations without editing YAML files.

Examples
--------
# Baseline (E1)
python train.py --config configs/baseline.yaml

# Scratch ablation (E2)
python train.py --config configs/scratch.yaml

# SE-ResNet18 (E3)
python train.py --config configs/se_resnet18.yaml

# CBAM-ResNet18 (E4)
python train.py --config configs/cbam_resnet18.yaml

# Quick override: fewer epochs, enable wandb
python train.py --config configs/baseline.yaml training.epochs=10 logging.use_wandb=true
"""

import sys
import os

# Ensure project root is on the path when run as `python train.py`
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from experiments.run import main

if __name__ == "__main__":
    main()
