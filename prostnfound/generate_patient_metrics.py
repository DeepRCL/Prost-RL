#!/usr/bin/env python3
"""
Generate patient-level metrics JSON for each model and a comparison table.

Saves  <model_output_dir>/metrics_patient.json  per model (matching the
structure of the existing metrics.json / test_rl output), then writes a
comparison table to <output_dir>/patient_metrics_comparison.{json,txt}.

Usage:
    python generate_patient_metrics.py \
        --models "APO=outputs_v2/V3-APO-continuous-fixed" \
                 "ProstNFound=outputs_v2/pnfplus-final" \
        --output_dir plots/patient_metrics
"""

import argparse, json, os, sys, warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.metrics import (
    roc_auc_score, roc_curve,
    balanced_accuracy_score, accuracy_score,
    f1_score,
)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SPEC_TARGETS = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
SCORE_COLS = {
    "heatmap":         "average_needle_heatmap_value",
    "heatmap_thresh":  "thresholded_needle_involvement",
    "clf":             "image_level_cancer_logits",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _auc(y, s):
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, s))


def _sens_at_spec(y, s, spec_target):
    if len(np.unique(y)) < 2:
        return None
    fpr, tpr, _ = roc_curve(y, s)
    spec = 1.0 - fpr
    idx = int(np.clip(np.searchsorted(-spec, -spec_target), 0, len(tpr) - 1))
    return float(tpr[idx])


def _opt_threshold(y, s, metric="balanced_accuracy"):
    threshs = np.linspace(0.01, 0.99, 99)
    best_score, best_t = -1.0, 0.5
    for t in threshs:
        yp = (s >= t).astype(int)
        if metric == "balanced_accuracy":
            sc = balanced_accuracy_score(y, yp)
        elif metric == "f1":
            sc = f1_score(y, yp, zero_division=0)
        else:
            sc = balanced_accuracy_score(y, yp)
        if sc > best_score:
            best_score, best_t = sc, t
    return float(best_t), float(best_score)


def _patient_agg(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """Aggregate core-level rows to patient level."""
    needed = [score_col, "label", "patient_id"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for patient aggregation: {missing}")
    sub = df.dropna(subset=needed)
    g = sub.groupby("patient_id")
    rows = {
        "patient_id":       list(g.groups.keys()),
        "max_score":        g[score_col].max().values,
        "mean_score":       g[score_col].mean().values,
        "label":            g["label"].max().values.astype(int),   # 1 if any core cancer
        "max_involvement":  g["involvement"].max().values if "involvement" in sub else np.nan,
    }
    if "grade_group" in sub.columns:
        rows["has_cspca"] = g["grade_group"].apply(lambda x: int((x > 2).any())).values
    else:
        rows["has_cspca"] = np.zeros(len(rows["patient_id"]), dtype=int)
    return pd.DataFrame(rows)


def _subgroup_patient(pat: pd.DataFrame, grp: str) -> pd.DataFrame:
    if grp == "all":
        return pat
    if grp == "high_inv":
        return pat[(pat["max_involvement"] >= 0.4) | (pat["label"] == 0)]
    if grp == "cspca":
        return pat[(pat["has_cspca"] == 1) | (pat["label"] == 0)]
    raise ValueError(grp)


def _compute_metrics_for_scores(y: np.ndarray, s: np.ndarray) -> dict:
    """Full set of metrics for a (label, score) pair."""
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {}
    aucroc = _auc(y, s)
    t_ba, ba = _opt_threshold(y, s, "balanced_accuracy")
    t_f1, f1 = _opt_threshold(y, s, "f1")
    # sensitivity at fixed 0.5 threshold
    yp_05 = (s >= 0.5).astype(int)
    yp_ba = (s >= t_ba).astype(int)
    sens_at_specs = {
        f"sensitivity_at_spec_{int(sp*100)}": _sens_at_spec(y, s, sp)
        for sp in SPEC_TARGETS
    }
    return {
        "n_patients": int(len(y)),
        "n_cancer_patients": int(y.sum()),
        "n_benign_patients": int((y == 0).sum()),
        "auroc": aucroc,
        "optimal_threshold_balanced_accuracy": t_ba,
        "balanced_accuracy_at_optimal_threshold": ba,
        "accuracy_at_optimal_threshold": float(accuracy_score(y, yp_ba)),
        "f1_at_optimal_threshold": float(f1),
        "balanced_accuracy_at_0.5": float(balanced_accuracy_score(y, yp_05)),
        "accuracy_at_0.5": float(accuracy_score(y, yp_05)),
        **sens_at_specs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Core-level balanced accuracy (summary only)
# ─────────────────────────────────────────────────────────────────────────────

def _core_balanced_accuracy(df: pd.DataFrame, score_col: str) -> Optional[dict]:
    if score_col not in df.columns:
        return None
    sub = df.dropna(subset=[score_col, "label"])
    if len(np.unique(sub["label"])) < 2:
        return None
    y = sub["label"].values
    s = sub[score_col].values
    t, ba = _opt_threshold(y, s, "balanced_accuracy")
    yp_05 = (s >= 0.5).astype(int)
    return {
        "n_cores": int(len(y)),
        "auroc": _auc(y, s),
        "optimal_threshold_balanced_accuracy": t,
        "balanced_accuracy_at_optimal_threshold": ba,
        "balanced_accuracy_at_0.5": float(balanced_accuracy_score(y, yp_05)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Per-model computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_model_patient_metrics(output_dir: str) -> dict:
    csv_path = os.path.join(output_dir, "metrics_by_core.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)

    result = {}

    for head, col in SCORE_COLS.items():
        if col not in df.columns:
            print(f"    skip {head} (column '{col}' not found)")
            continue

        head_result = {}

        # ── core-level summary ──────────────────────────────────────────────
        core_ba = _core_balanced_accuracy(df, col)
        if core_ba:
            head_result["core_level"] = core_ba

        # ── patient-level ───────────────────────────────────────────────────
        try:
            pat = _patient_agg(df, col)
        except KeyError as e:
            print(f"    WARN: patient aggregation failed for {head}: {e}")
            continue

        for grp in ("all", "high_inv", "cspca"):
            sub = _subgroup_patient(pat, grp)
            if len(sub) < 5 or len(np.unique(sub["label"])) < 2:
                head_result[f"patient_{grp}"] = {"n_patients": int(len(sub)), "note": "insufficient data"}
                continue
            metrics = _compute_metrics_for_scores(sub["label"].values, sub["max_score"].values)
            head_result[f"patient_{grp}"] = metrics
            # also with mean score
            metrics_mean = _compute_metrics_for_scores(sub["label"].values, sub["mean_score"].values)
            head_result[f"patient_{grp}_mean_agg"] = metrics_mean

        result[head] = head_result

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Text table helpers
# ─────────────────────────────────────────────────────────────────────────────

_METRIC_ROWS = [
    ("auroc",                                    "AUROC"),
    ("balanced_accuracy_at_optimal_threshold",   "Bal.Acc @opt.thresh"),
    ("balanced_accuracy_at_0.5",                 "Bal.Acc @0.50"),
    ("sensitivity_at_spec_40",                   "Sens @Spec40%"),
    ("sensitivity_at_spec_50",                   "Sens @Spec50%"),
    ("sensitivity_at_spec_60",                   "Sens @Spec60%"),
    ("sensitivity_at_spec_70",                   "Sens @Spec70%"),
    ("sensitivity_at_spec_80",                   "Sens @Spec80%"),
    ("sensitivity_at_spec_90",                   "Sens @Spec90%"),
    ("n_patients",                               "N patients"),
    ("n_cancer_patients",                        "N cancer"),
    ("n_benign_patients",                        "N benign"),
]


def _fmt(v):
    if v is None:
        return "  —  "
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def build_comparison_table(all_metrics: dict, head: str, grp: str) -> str:
    """Build a pretty text table for a given head + subgroup across all models."""
    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f"  HEAD: {head}   |   SUBGROUP: {grp}   (patient-level, max-score aggregation)")
    lines.append(f"{'='*80}")

    model_names = list(all_metrics.keys())
    col_w = max(18, max(len(n) for n in model_names) + 2)
    header = f"{'Metric':<30}" + "".join(f"{n:>{col_w}}" for n in model_names)
    lines.append(header)
    lines.append("-" * len(header))

    key = f"patient_{grp}"
    for mkey, mlabel in _METRIC_ROWS:
        row = f"{mlabel:<30}"
        for name in model_names:
            v = all_metrics[name].get(head, {}).get(key, {}).get(mkey)
            row += f"{_fmt(v):>{col_w}}"
        lines.append(row)

    # core-level
    lines.append("")
    lines.append(f"{'--- Core-level ---':<30}")
    for mkey, mlabel in [
        ("auroc",                                  "Core AUROC"),
        ("balanced_accuracy_at_optimal_threshold", "Core Bal.Acc @opt"),
        ("balanced_accuracy_at_0.5",               "Core Bal.Acc @0.50"),
        ("n_cores",                                "N cores"),
    ]:
        row = f"{mlabel:<30}"
        for name in model_names:
            v = all_metrics[name].get(head, {}).get("core_level", {}).get(mkey)
            row += f"{_fmt(v):>{col_w}}"
        lines.append(row)

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, metavar="NAME=PATH")
    ap.add_argument("--output_dir", default="plots/patient_metrics",
                    help="Directory to save the comparison table (JSON + TXT)")
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
    os.makedirs(args.output_dir, exist_ok=True)

    all_metrics = {}
    for name, output_dir in model_dict.items():
        print(f"\n[{name}]  {output_dir}")
        try:
            m = compute_model_patient_metrics(output_dir)
            all_metrics[name] = m
            # Save per-model JSON
            out_path = os.path.join(output_dir, "metrics_patient.json")
            with open(out_path, "w") as f:
                json.dump(m, f, indent=2)
            print(f"  → saved {out_path}")
        except Exception as e:
            print(f"  ERROR: {e}")
            all_metrics[name] = {}

    # Save combined JSON
    combined_json = os.path.join(args.output_dir, "patient_metrics_comparison.json")
    with open(combined_json, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved combined JSON: {combined_json}")

    # Build and save text table
    table_lines = []
    table_lines.append("PATIENT-LEVEL METRICS COMPARISON")
    table_lines.append(f"Models: {', '.join(all_metrics.keys())}")
    table_lines.append("")

    for head in list(SCORE_COLS.keys()):
        for grp in ("all", "high_inv", "cspca"):
            table_lines.append(build_comparison_table(all_metrics, head, grp))

    table_str = "\n".join(table_lines)

    txt_path = os.path.join(args.output_dir, "patient_metrics_comparison.txt")
    with open(txt_path, "w") as f:
        f.write(table_str)
    print(f"Saved text table: {txt_path}")
    print("\n" + table_str)


if __name__ == "__main__":
    main()
