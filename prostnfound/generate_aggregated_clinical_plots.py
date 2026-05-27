#!/usr/bin/env python3
"""
Aggregated (cross-fold) clinical plots.

Loads metrics_by_core.csv from every fold for each model, concatenates them,
and runs the identical plotting pipeline from generate_clinical_plots.py on
the pooled data.

Usage:
    python generate_aggregated_clinical_plots.py \\
        --models \\
        "supervised=outputs_v2/cv_eval/fold{fold}" \\
        "rl=outputs/cv_eval_tune/fold{fold}" \\
        "pnf=../pnf/pnf_vanila_with_prompt{fold}" \\
        "pnf+=../pnf+/pnfplus_with_prompt{fold}" \\
        --folds 0 1 2 3 4 \\
        --output_dir plots/paper-base-aggregated/
"""

import argparse, os, sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Re-use everything from the existing script
sys.path.insert(0, str(Path(__file__).parent))
from generate_clinical_plots import load_model, run


# ─────────────────────────────────────────────────────────────────────────────
# Data loading — aggregate across folds
# ─────────────────────────────────────────────────────────────────────────────

def load_model_aggregated(path_template: str, folds: List[int]) -> pd.DataFrame:
    """
    Load and concatenate metrics_by_core.csv from all fold directories.
    `path_template` may contain a ``{fold}`` placeholder, e.g.
        outputs_v2/cv_eval/fold{fold}
    If there is no placeholder the same path is used for every fold (no-op
    repetition) — in practice you'll always use a placeholder.
    """
    frames = []
    for fold in folds:
        path = path_template.format(fold=fold)
        try:
            df = load_model(path)
            df["_fold"] = fold
            frames.append(df)
            print(f"    fold {fold}: {len(df)} cores  ({path})")
        except Exception as exc:
            print(f"    WARN fold {fold}: skip — {exc}  ({path})")
    if not frames:
        raise ValueError(f"No data loaded for template '{path_template}'")
    combined = pd.concat(frames, ignore_index=True)
    print(f"    → total: {len(combined)} cores across {len(frames)} folds")
    return combined


def load_all_aggregated(
    model_dict: Dict[str, str], folds: List[int]
) -> Dict[str, pd.DataFrame]:
    out = {}
    for name, path_template in model_dict.items():
        print(f"  Loading '{name}' …")
        try:
            out[name] = load_model_aggregated(path_template, folds)
        except Exception as exc:
            print(f"  WARN: skip '{name}': {exc}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(
        description="Generate aggregated (cross-fold) clinical comparison plots."
    )
    ap.add_argument(
        "--models",
        nargs="+",
        required=True,
        metavar="NAME=PATH_TEMPLATE",
        help=(
            "Model specs as NAME=PATH_TEMPLATE.  "
            "Use {fold} in the path to mark where the fold number is substituted, "
            "e.g. 'supervised=outputs_v2/cv_eval/fold{fold}'."
        ),
    )
    ap.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        metavar="N",
        help="Fold indices to aggregate (default: 0 1 2 3 4).",
    )
    ap.add_argument(
        "--output_dir",
        default="plots/paper-base-aggregated",
        help="Directory where all plots are saved.",
    )
    return ap.parse_args()


def parse_model_specs(specs):
    out = {}
    for s in specs:
        if "=" in s:
            n, p = s.split("=", 1)
            out[n.strip()] = p.strip()
        else:
            out[Path(s).name] = s
    return out


def main():
    args = parse_args()
    model_dict = parse_model_specs(args.models)

    print(f"Folds to aggregate: {args.folds}")
    print("Loading models …")
    data = load_all_aggregated(model_dict, args.folds)

    if not data:
        print("ERROR: No models loaded.")
        sys.exit(1)

    run(data, args.output_dir)
    print(f"\nAll figures saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
