@echo off
echo Running E1 baseline
python train.py --config configs/baseline.yaml data.num_workers=0

echo Running hyperparameter search...
python experiments/sweep.py --config configs/hyperparam_search.yaml

echo Running E2 scratch...
python train.py --config configs/scratch.yaml data.num_workers=0

echo Running E3 SE-ResNet18...
python train.py --config configs/se_resnet18.yaml data.num_workers=0

echo Running E4 CBAM-ResNet18...
python train.py --config configs/cbam_resnet18.yaml data.num_workers=0

echo All experiments finished.
pause