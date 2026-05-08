"""
Hyperparameter sweep runner.

Reads a sweep config (configs/hyperparam_search.yaml), generates all
combinations from the 'sweep' grid, runs each as a separate experiment,
and produces a summary CSV + Markdown table of results.

Usage:
    python experiments/sweep.py --config configs/hyperparam_search.yaml
    python experiments/sweep.py --config configs/hyperparam_search.yaml --dry-run
"""

import argparse
import csv
import itertools
import os
import sys
import torch
import gc
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import load_config, merge_config_with_overrides
from experiments.run import run_experiment


def generate_grid(sweep_cfg: Dict[str, List]) -> List[Dict[str, Any]]:
    """
    Generate all combinations from a sweep dict.

    Example input:
        {"backbone_lr": [1e-5, 1e-4], "head_lr": [1e-3, 1e-2], "epochs": [20, 30]}

    Returns a list of flat dicts, one per combination:
        [{"backbone_lr": 1e-5, "head_lr": 1e-3, "epochs": 20}, ...]
    """
    keys = list(sweep_cfg.keys())
    values = [sweep_cfg[k] for k in keys]
    combos = []
    for combo in itertools.product(*values):
        combos.append(dict(zip(keys, combo)))
    return combos


def combo_to_overrides(combo: Dict[str, Any]) -> Dict[str, Any]:
    """Map sweep combo keys to config dot-path overrides."""
    key_map = {
        "backbone_lr": "training.backbone_lr",
        "head_lr": "training.head_lr",
        "epochs": "training.epochs",
        "batch_size": "data.batch_size",
    }
    return {key_map.get(k, k): v for k, v in combo.items()}


def combo_name(base_name: str, combo: Dict[str, Any]) -> str:
    """Generate a unique experiment name for a sweep combo."""
    parts = []
    for k, v in combo.items():
        short_k = k.replace("backbone_lr", "bblr").replace("head_lr", "hlr").replace("epochs", "ep")
        parts.append(f"{short_k}{v:.0e}" if isinstance(v, float) else f"{short_k}{v}")
    return f"{base_name}_{'_'.join(parts)}"


def save_summary(
    results: List[Dict[str, Any]],
    output_dir: str,
    base_name: str,
) -> None:
    """Write sweep results as CSV and Markdown table."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{base_name}_sweep_results.csv")
    md_path = os.path.join(output_dir, f"{base_name}_sweep_results.md")

    if not results:
        return

    fieldnames = list(results[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Markdown table
    header = " | ".join(fieldnames)
    sep = " | ".join(["---"] * len(fieldnames))
    rows = [" | ".join(str(r[k]) for k in fieldnames) for r in results]
    with open(md_path, "w") as f:
        f.write(f"# Hyperparameter Sweep Results\n\n")
        f.write(f"| {header} |\n| {sep} |\n")
        for row in rows:
            f.write(f"| {row} |\n")

    print(f"\nSweep summary saved:\n  CSV : {csv_path}\n  MD  : {md_path}")


def main():
    parser = argparse.ArgumentParser(description="Run hyperparameter sweep.")
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned experiments without running them.",
    )
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    sweep_cfg = base_cfg.pop("sweep", {})

    if not sweep_cfg:
        print("No 'sweep' section found in config. Running single experiment.")
        run_experiment(base_cfg)
        return

    combos = generate_grid(sweep_cfg)
    base_name = base_cfg["experiment"]["name"]
    print(f"Sweep over {len(combos)} combinations:")
    for i, combo in enumerate(combos, 1):
        print(f"  [{i:2d}] {combo}")

    if args.dry_run:
        print("\n--dry-run: no experiments were executed.")
        return

    results = []
    for i, combo in enumerate(combos, 1):
        exp_name = combo_name(base_name, combo)
        overrides = combo_to_overrides(combo)
        overrides["experiment.name"] = exp_name

        print(f"\n{'='*60}")
        print(f"[{i}/{len(combos)}] Running: {exp_name}")
        print(f"{'='*60}")

        cfg = merge_config_with_overrides(base_cfg, overrides)
        try:
            out = run_experiment(cfg)
            test_top1 = out["test_metrics"]["top1"]
            best_val = max(out["history"]["val_top1"])
        except Exception as e:
            print(f"ERROR in {exp_name}: {e}")
            test_top1 = -1.0
            best_val = -1.0
        finally:
            gc.collect()
            if(torch.cuda.is_available()):
                torch.cuda.empty_cache()    

        row = {**combo, "best_val_top1": round(best_val, 2), "test_top1": round(test_top1, 2)}
        results.append(row)

    output_dir = os.path.join(base_cfg["experiment"].get("output_dir", "outputs"), "logs")
    save_summary(results, output_dir, base_name)

    # Print final table
    print("\n" + "=" * 60)
    print("SWEEP COMPLETE — Summary")
    print("=" * 60)
    header_keys = list(results[0].keys())
    print("  ".join(f"{k:>14}" for k in header_keys))
    for row in sorted(results, key=lambda r: r["test_top1"], reverse=True):
        print("  ".join(f"{str(row[k]):>14}" for k in header_keys))


if __name__ == "__main__":
    main()
