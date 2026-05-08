# Pet Classification — Oxford-IIIT Pet Dataset

ResNet-18 fine-tuning study for 37-class pet breed classification.  
Covers baseline transfer learning, hyperparameter search, pretraining ablation, and SE/CBAM attention mechanisms.

---

## Project Overview

| Experiment | Model | Pretrained | Attention | Purpose |
|---|---|---|---|---|
| E1 | ResNet-18 | ✓ | — | Baseline |
| E2 | ResNet-18 | ✗ | — | Pretraining ablation |
| E3 | SE-ResNet-18 | ✓ | SE | Attention comparison |
| E4 | CBAM-ResNet-18 | ✓ | CBAM | Attention comparison |

---

## Project Structure

```
task1/
├── configs/               # YAML experiment configs
│   ├── baseline.yaml
│   ├── scratch.yaml
│   ├── hyperparam_search.yaml
│   ├── se_resnet18.yaml
│   └── cbam_resnet18.yaml
├── data/                  # Dataset pipeline
│   ├── pet_dataset.py     # OxfordPetsDataset + build_dataloaders()
│   ├── transforms.py      # Train / val transforms
│   └── splits.py          # Stratified train/val/test splits
├── models/                # Model definitions
│   ├── resnet18.py        # Baseline ResNet-18
│   ├── se_block.py        # Squeeze-and-Excitation block
│   ├── cbam.py            # CBAM (channel + spatial attention)
│   ├── se_resnet18.py     # SE-ResNet-18
│   └── cbam_resnet18.py   # CBAM-ResNet-18
├── engine/                # Training engine
│   ├── trainer.py         # Full training loop
│   ├── evaluator.py       # Evaluation loop + metrics
│   ├── losses.py          # Loss functions
│   └── checkpoint.py      # Checkpoint save/load
├── utils/                 # Shared utilities
│   ├── metrics.py         # AverageMeter, top-k accuracy
│   ├── seed.py            # Global seed fixing
│   ├── logger.py          # Console + W&B/SwanLab logger
│   └── config.py          # YAML load/save/merge
├── experiments/
│   ├── run.py             # Single experiment runner
│   └── sweep.py           # Hyperparameter sweep runner
├── outputs/               # Auto-created at runtime
│   ├── checkpoints/       # best.pth + last.pth per experiment
│   ├── logs/              # train.log + history.json per experiment
│   └── figures/           # confusion matrix PNGs
├── train.py               # Training entrypoint
├── test.py                # Test-set evaluation entrypoint
└── requirements.txt
```

---

## Environment Setup

```bash
# 1. Create a virtual environment (Python ≥ 3.9 recommended)
python -m venv venv
source venv/bin/activate          # Linux / macOS
venv\Scripts\activate             # Windows PowerShell

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Install experiment tracker
pip install wandb      # Weights & Biases
# or
pip install swanlab    # SwanLab
```

CUDA is strongly recommended. The code falls back to CPU automatically.

---

## Dataset Preparation

Download the Oxford-IIIT Pet Dataset from the official site:

```bash
# Images (~800 MB)
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
# Annotations (~19 MB)
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz

# Extract into data/oxford_pets/
mkdir -p data/oxford_pets
tar -xzf images.tar.gz -C data/oxford_pets/
tar -xzf annotations.tar.gz -C data/oxford_pets/
```

Expected layout after extraction:

```
data/oxford_pets/
├── images/
│   ├── Abyssinian_1.jpg
│   └── ...
└── annotations/
    ├── list.txt
    ├── trainval.txt
    └── test.txt
```

The `data_root` key in each YAML config must point to this directory.  
Default: `data/oxford_pets`

**Dataset statistics (default 80/20 split):**
- Total trainval images: ~3,680
- Train: ~2,944 | Val: ~736 | Test: ~3,669
- Classes: 37 breeds (25 dog + 12 cat)

---

## Running Training

### Single experiment

```bash
# E1 — Baseline (pretrained, no attention)
python train.py --config configs/baseline.yaml

# E2 — From scratch
python train.py --config configs/scratch.yaml

# E3 — SE-ResNet18
python train.py --config configs/se_resnet18.yaml

# E4 — CBAM-ResNet18
python train.py --config configs/cbam_resnet18.yaml
```

### Command-line overrides

Any YAML key can be overridden via `key=value` arguments:

```bash
# Shorter run for debugging
python train.py --config configs/baseline.yaml training.epochs=5 data.batch_size=32

# Enable W&B logging
python train.py --config configs/baseline.yaml logging.use_wandb=true

# Enable SwanLab logging
python train.py --config configs/baseline.yaml logging.use_swanlab=true
```

---

## Hyperparameter Search

```bash
python experiments/sweep.py --config configs/hyperparam_search.yaml

# Preview planned runs without executing
python experiments/sweep.py --config configs/hyperparam_search.yaml --dry-run
```

The sweep grid (defined under `sweep:` in the config) runs all combinations of:
- `backbone_lr`: `[1e-5, 1e-4]`
- `head_lr`: `[1e-3, 1e-2]`
- `epochs`: `[20, 30]`

Results are written to `outputs/logs/hyperparam_search_sweep_results.{csv,md}`.

---

## Running Test Evaluation

```bash
# Evaluate best checkpoint from E1 baseline
python test.py \
    --config configs/baseline.yaml \
    --checkpoint outputs/checkpoints/E1_baseline/best.pth

# Include confusion matrix figure
python test.py \
    --config configs/baseline.yaml \
    --checkpoint outputs/checkpoints/E1_baseline/best.pth \
    --save-cm
```

Outputs:
- Console: top-1, top-5 accuracy, per-class accuracy
- File: `outputs/logs/<exp_name>/test_results.json`
- Figure: `outputs/figures/<exp_name>_confusion_matrix.png` (if `--save-cm`)

---

## Reproducing All Experiments

Run all four required experiments in order:

```bash
# Step 1: Baseline
python train.py --config configs/baseline.yaml

# Step 2: Hyperparameter search
python experiments/sweep.py --config configs/hyperparam_search.yaml

# Step 3: Pretraining ablation
python train.py --config configs/scratch.yaml

# Step 4: Attention models
python train.py --config configs/se_resnet18.yaml
python train.py --config configs/cbam_resnet18.yaml

# Evaluate all
for exp in E1_baseline E2_scratch E3_se_resnet18 E4_cbam_resnet18; do
    python test.py \
        --config configs/${exp#E?_}.yaml \
        --checkpoint outputs/checkpoints/${exp}/best.pth \
        --save-cm
done
```

---

## Output Locations

| Type | Location |
|---|---|
| Best model checkpoint | `outputs/checkpoints/<exp_name>/best.pth` |
| Latest checkpoint | `outputs/checkpoints/<exp_name>/last.pth` |
| Training log | `outputs/logs/<exp_name>/train.log` |
| Training curves (JSON) | `outputs/logs/<exp_name>/history.json` |
| Test results (JSON) | `outputs/logs/<exp_name>/test_results.json` |
| Saved config | `outputs/logs/<exp_name>/config.yaml` |
| Confusion matrix | `outputs/figures/<exp_name>_confusion_matrix.png` |
| Sweep summary | `outputs/logs/hyperparam_search_sweep_results.{csv,md}` |

---

## Experiment Tracking (W&B / SwanLab)

### Weights & Biases

```bash
pip install wandb
wandb login

python train.py --config configs/baseline.yaml logging.use_wandb=true
```

Or set `use_wandb: true` permanently in the YAML config.

### SwanLab

```bash
pip install swanlab
swanlab login

python train.py --config configs/baseline.yaml logging.use_swanlab=true
```

Logged metrics include:
- `train/loss` — per-epoch training loss
- `val/loss` — per-epoch validation loss
- `val/top1` — validation top-1 accuracy
- `val/top5` — validation top-5 accuracy
- `lr/backbone` — backbone learning rate schedule
- `lr/head` — head learning rate schedule
- `test/top1`, `test/top5`, `test/loss` — final test metrics

---

## Downloading Pretrained Checkpoints

If you want to skip training and download pre-trained checkpoints:

> **Note**: Upload your checkpoints to a cloud service (Google Drive, OneDrive, HuggingFace Hub, etc.) and update the links below.

```bash
# Example: download from HuggingFace Hub
pip install huggingface_hub
python - <<'EOF'
from huggingface_hub import hf_hub_download
# Update repo_id and filename to match your uploaded files
path = hf_hub_download(repo_id="your-username/pet-classification", filename="E1_baseline_best.pth")
print(f"Downloaded to: {path}")
EOF
```

Or manually place `.pth` files under `outputs/checkpoints/<exp_name>/best.pth`.

---

## Interpreting Outputs

### Training curves (`history.json`)

```python
import json, matplotlib.pyplot as plt

with open("outputs/logs/E1_baseline/history.json") as f:
    h = json.load(f)

epochs = range(1, len(h["train_loss"]) + 1)
plt.plot(epochs, h["train_loss"], label="train loss")
plt.plot(epochs, h["val_loss"],  label="val loss")
plt.plot(epochs, h["val_top1"], label="val top-1 (%)")
plt.legend(); plt.show()
```

### Comparison table (example)

| Model | Pretrained | Params | Val Top-1 | Test Top-1 |
|---|---|---|---|---|
| ResNet-18 | ✓ | 11.2M | ~88% | ~87% |
| ResNet-18 | ✗ | 11.2M | ~55% | ~54% |
| SE-ResNet-18 | ✓ | 11.3M | ~89% | ~88% |
| CBAM-ResNet-18 | ✓ | 11.3M | ~89% | ~88% |

*(Actual values depend on your hardware and random seed.)*

---

## Key Design Decisions

- **Differential learning rates**: backbone uses `1e-4`, head uses `1e-3` (10× larger).  
  This preserves pretrained features while fast-adapting the classifier.
- **Cosine annealing**: smoothly decays learning rate to `1e-6` over all epochs.
- **Mixed precision** (`torch.cuda.amp`): reduces GPU memory and speeds up training on CUDA.
- **Stratified val split**: each class contributes proportionally to validation set.
- **SE / CBAM injection**: attention blocks are inserted into existing BasicBlocks  
  (on the residual branch, before the skip addition) without rewriting the full network.

---

## Citation

```bibtex
@dataset{oxford_pets,
  author    = {Parkhi, Omkar and Vedaldi, Andrea and Zisserman, Andrew and Jawahar, C.V.},
  title     = {The Oxford-IIIT Pet Dataset},
  year      = {2012},
  url       = {https://www.robots.ox.ac.uk/~vgg/data/pets/}
}
```
