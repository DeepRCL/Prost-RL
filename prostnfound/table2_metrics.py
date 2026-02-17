"""
Compute paper-style GG2+/GG3+ biopsy-level metrics from metrics_by_core.csv.

Usage:
  python table2_metrics.py --csv outputs/ua_zeroshot_eval/pnf_plus_baseline/metrics_by_core.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _sens_spec(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> tuple[float, float]:
    y_pred = (y_score >= threshold).astype(np.int64)
    tp = float(((y_pred == 1) & (y_true == 1)).sum())
    tn = float(((y_pred == 0) & (y_true == 0)).sum())
    fp = float(((y_pred == 1) & (y_true == 0)).sum())
    fn = float(((y_pred == 0) & (y_true == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    return sens, spec


def _best_youden_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j = tpr - fpr
    idx = int(np.nanargmax(j))
    return float(thresholds[idx])


def _task_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    auc = _safe_auc(y_true, y_score)
    thr = _best_youden_threshold(y_true, y_score)
    sens_05, spec_05 = _sens_spec(y_true, y_score, threshold=0.5)
    if np.isnan(thr):
        sens_best, spec_best = float("nan"), float("nan")
    else:
        sens_best, spec_best = _sens_spec(y_true, y_score, threshold=thr)
    return {
        "n": int(len(y_true)),
        "n_pos": int((y_true == 1).sum()),
        "n_neg": int((y_true == 0).sum()),
        "auroc": auc,
        "threshold_0.5": 0.5,
        "sens_at_0.5": float(sens_05),
        "spec_at_0.5": float(spec_05),
        "threshold_best_youden": float(thr),
        "sens_at_best_youden": float(sens_best),
        "spec_at_best_youden": float(spec_best),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute GG2+/GG3+ table metrics.")
    parser.add_argument("--csv", required=True, help="Path to metrics_by_core.csv")
    parser.add_argument(
        "--score-col",
        default="average_needle_heatmap_value",
        help="Prediction score column to use.",
    )
    parser.add_argument(
        "--out-json",
        default=None,
        help="Optional output JSON path (default: <csv_dir>/table2_metrics_<score-col>.json)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if args.score_col not in df.columns:
        raise KeyError(f"Score column '{args.score_col}' not in CSV. Available: {list(df.columns)}")

    if "grade_group" not in df.columns:
        raise KeyError(
            "grade_group column not found in CSV. Cannot compute GG2+/GG3+ metrics."
        )

    scores = pd.to_numeric(df[args.score_col], errors="coerce").fillna(0.0).to_numpy()
    gg = pd.to_numeric(df["grade_group"], errors="coerce").fillna(0.0).to_numpy()

    labels_gg2 = (gg >= 2).astype(np.int64)
    labels_gg3 = (gg >= 3).astype(np.int64)

    out = {
        "score_column": args.score_col,
        "gg2_plus_vs_non_pca": _task_metrics(labels_gg2, scores),
        "gg3_plus_vs_non_cspca": _task_metrics(labels_gg3, scores),
        "notes": (
            "Missing grade_group is treated as 0 (non-cancer / non-csPCa) "
            "for GG-threshold tasks."
        ),
    }

    out_json = (
        Path(args.out_json)
        if args.out_json
        else csv_path.parent / f"table2_metrics_{args.score_col}.json"
    )
    with out_json.open("w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()

