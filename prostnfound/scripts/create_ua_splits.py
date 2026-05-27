"""
Create deterministic patient-level train/val/test splits for UA (Optimum) finetuning.

The split is at the patient (case directory) level so no patient's cores
appear in both the finetune and test sets.

Usage
-----
Run from the prostnfound/ directory:

    python scripts/create_ua_splits.py \\
        --root_dir /data/project/prostate-us/processed/processed/processed/UA_annotated_needles \\
        --output_file splits/ua_finetune_splits.json \\
        --finetune_fraction 0.3 \\
        --val_fraction_of_finetune 0.2 \\
        --seed 42

The resulting JSON file will have three keys:
    "train"  - patients used for finetuning
    "val"    - patients used for validation during training
    "test"   - held-out patients used for final evaluation (never seen during training)

With the defaults above and 1188 cores (~120 patients), you get roughly:
    train  ~24 % of patients  (finetune)
    val    ~6 % of patients   (validation during training)
    test   ~70 % of patients  (held-out, large set for reliable evaluation)
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np


def create_splits(
    root_dir: str,
    finetune_fraction: float = 0.3,
    val_fraction_of_finetune: float = 0.2,
    seed: int = 42,
) -> dict:
    """
    Return a dict with 'train', 'val', and 'test' patient ID lists.

    finetune_fraction          : fraction of ALL patients put aside for finetuning
    val_fraction_of_finetune   : fraction of the finetune patients used as val
    seed                       : numpy RandomState seed (fixed → reproducible)
    """
    root = Path(root_dir)
    cases = sorted(p.name for p in root.iterdir() if p.is_dir())

    if len(cases) == 0:
        raise ValueError(f"No case directories found under {root_dir}")

    # Shuffle with a fixed seed for full reproducibility
    rng = np.random.RandomState(seed)
    cases_shuffled = list(cases)
    rng.shuffle(cases_shuffled)

    n_total = len(cases_shuffled)
    n_finetune = round(n_total * finetune_fraction)
    n_val = round(n_finetune * val_fraction_of_finetune)
    n_train = n_finetune - n_val

    finetune_cases = cases_shuffled[:n_finetune]
    test_cases = cases_shuffled[n_finetune:]

    # Within finetune, first n_train go to train, the rest to val
    train_cases = finetune_cases[:n_train]
    val_cases = finetune_cases[n_train:]

    return {
        "train": sorted(train_cases),
        "val": sorted(val_cases),
        "test": sorted(test_cases),
        "_meta": {
            "root_dir": str(root_dir),
            "seed": seed,
            "finetune_fraction": finetune_fraction,
            "val_fraction_of_finetune": val_fraction_of_finetune,
            "n_total_patients": n_total,
            "n_train": n_train,
            "n_val": n_val,
            "n_test": len(test_cases),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Create deterministic UA patient-level splits for finetuning."
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Path to the UA_annotated_needles root directory.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="splits/ua_finetune_splits.json",
        help="Output JSON file (default: splits/ua_finetune_splits.json).",
    )
    parser.add_argument(
        "--finetune_fraction",
        type=float,
        default=0.3,
        help="Fraction of patients used for finetuning (train + val). Default: 0.3.",
    )
    parser.add_argument(
        "--val_fraction_of_finetune",
        type=float,
        default=0.2,
        help="Fraction of the finetune patients used for validation. Default: 0.2.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Must match the seed used in all subsequent training and test configs.",
    )
    args = parser.parse_args()

    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    splits = create_splits(
        root_dir=args.root_dir,
        finetune_fraction=args.finetune_fraction,
        val_fraction_of_finetune=args.val_fraction_of_finetune,
        seed=args.seed,
    )

    meta = splits["_meta"]
    n_total = meta["n_total_patients"]
    print(f"Total patients : {n_total}")
    print(
        f"  train  : {meta['n_train']:>4d}  ({100 * meta['n_train'] / n_total:5.1f}%)"
    )
    print(
        f"  val    : {meta['n_val']:>4d}  ({100 * meta['n_val'] / n_total:5.1f}%)"
    )
    print(
        f"  test   : {meta['n_test']:>4d}  ({100 * meta['n_test'] / n_total:5.1f}%)"
    )
    print(f"Saved → {output_file}")

    with open(output_file, "w") as f:
        json.dump(splits, f, indent=2)


if __name__ == "__main__":
    main()
