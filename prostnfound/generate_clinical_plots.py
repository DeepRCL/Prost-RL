#!/usr/bin/env python3
"""
Clinical Model Comparison Plots — comprehensive edition.

Each figure is saved as its own file inside subfolders:
  <output_dir>/
    heatmap_head/
    clf_head/
    patient_level/
    shared/         ← cross-head / raw scatter plots

Usage:
    python generate_clinical_plots.py \\
        --models "APO=outputs_v2/V3-APO-continuous-fixed" \\
                 "Noise0.3=outputs_v2/V3-DRPO-continuous-noise-0.3" \\
                 "ProstNFound=outputs_v2/pnfplus-final" \\
        --output_dir plots/clinical_comparison
"""

import argparse, json, os, sys, warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import pearsonr
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    balanced_accuracy_score, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix,
)
warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────────────────────────────────────
# Palette
# ─────────────────────────────────────────────────────────────────────────────
COLORS  = ["#2166ac","#d6604d","#4dac26","#7b2d8b","#f4a582","#018571","#b2182b","#e08a00"]
MARKERS = ["o","^","s","D","v","<",">","p"]
STYLES  = ["-","--","-.",":","-","--","-."]


def _c(i): return COLORS[i % len(COLORS)]
def _m(i): return MARKERS[i % len(MARKERS)]
def _s(i): return STYLES[i % len(STYLES)]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(output_dir: str) -> pd.DataFrame:
    path = os.path.join(output_dir, "metrics_by_core.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}")
    df = pd.read_csv(path)
    df["clinically_significant"] = df["clinically_significant"].astype(str).str.lower() == "true"
    df["high_inv_mask"]  = (df["involvement"] > 0.4) | (df["label"] == 0)
    df["cspca_mask"]     = (df["grade_group"] > 2) | (df["label"] == 0)
    return df


def load_all(model_dict: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    out = {}
    for name, path in model_dict.items():
        try:
            out[name] = load_model(path)
            print(f"  Loaded '{name}': {len(out[name])} cores")
        except Exception as e:
            print(f"  WARN: skip '{name}': {e}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Stat helpers
# ─────────────────────────────────────────────────────────────────────────────

def _auc(y, s):
    if len(np.unique(y)) < 2: return np.nan
    return roc_auc_score(y, s)

def _roc(y, s):
    if len(np.unique(y)) < 2: return np.array([0,1]), np.array([0,1]), np.nan
    fpr, tpr, _ = roc_curve(y, s)
    return fpr, tpr, roc_auc_score(y, s)

def _sens_at_spec(y, s, spec_target):
    if len(np.unique(y)) < 2: return np.nan
    fpr, tpr, _ = roc_curve(y, s)
    spec = 1.0 - fpr
    # Use the best achievable sensitivity while satisfying specificity >= target.
    valid = spec >= spec_target
    if np.any(valid):
        return float(np.max(tpr[valid]))
    # Fallback (should be rare): return sensitivity at maximum specificity point.
    return float(tpr[np.argmax(spec)])

def _opt_threshold(y, s, metric="balanced_accuracy"):
    threshs = np.linspace(0.01, 0.99, 99)
    best_score, best_t = -1, 0.5
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
    return best_t, best_score

def _get_score(df: pd.DataFrame, col: str) -> Optional[np.ndarray]:
    if col not in df.columns:
        return None
    return df[col].values

def _subgroup(df: pd.DataFrame, grp: str) -> pd.DataFrame:
    if grp == "all":      return df
    if grp == "high_inv": return df[df["high_inv_mask"]]
    if grp == "cspca":    return df[df["cspca_mask"]]
    raise ValueError(grp)

GRPLABEL = {"all":"All Cores","high_inv":"High-Inv (>40%)","cspca":"csPCa (GG≥3)"}
SPEC_TARGETS = [0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
INV_THRESH_PCT = [5, 10, 20, 30, 40, 50, 60, 70]


# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────

def save(fig, directory: str, name: str, dpi: int = 200):
    os.makedirs(directory, exist_ok=True)
    for ext in ("png", "pdf"):
        p = os.path.join(directory, f"{name}.{ext}")
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
    print(f"    → {os.path.join(directory, name)}.{{png,pdf}}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Shared: ROC curves  (one file per subgroup)
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc(data, score_col, head_label, subgroup, out_dir, fname):
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, (name, df) in enumerate(data.items()):
        if score_col not in df.columns: continue
        sub = _subgroup(df, subgroup).dropna(subset=[score_col, "label"])
        fpr, tpr, auc = _roc(sub["label"].values, sub[score_col].values)
        ax.plot(fpr, tpr, color=_c(i), ls=_s(i), lw=2, marker=_m(i), markevery=0.2,
                markersize=5, label=f"{name}  (AUC={auc:.3f})")
    ax.plot([0,1],[0,1],"k--",lw=0.8,alpha=0.4)
    ax.axhspan(0.7, 1.02, color="#d4e6f1", alpha=0.12, zorder=0)
    ax.set_xlabel("1 − Specificity (FPR)", fontsize=11)
    ax.set_ylabel("Sensitivity (TPR)", fontsize=11)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.88)
    ax.set_xlim(0,1); ax.set_ylim(0,1.02)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    save(fig, out_dir, fname)


# ─────────────────────────────────────────────────────────────────────────────
# Shared: Sensitivity bars  (one file per subgroup)
# ─────────────────────────────────────────────────────────────────────────────

def plot_sensitivity_bars(data, score_col, head_label, subgroup, out_dir, fname):
    n = len(data)
    x = np.arange(len(SPEC_TARGETS))
    width = 0.8 / n
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (name, df) in enumerate(data.items()):
        if score_col not in df.columns: continue
        sub = _subgroup(df, subgroup).dropna(subset=[score_col, "label"])
        sens = [_sens_at_spec(sub["label"].values, sub[score_col].values, s)
                for s in SPEC_TARGETS]
        offset = (i - n/2 + 0.5) * width
        bars = ax.bar(x + offset, sens, width=width*0.9, color=_c(i), alpha=0.85,
                      label=name, edgecolor="white", lw=0.5)
        for bar, v in zip(bars, sens):
            if not np.isnan(v):
                ax.text(bar.get_x()+bar.get_width()/2, v+0.008, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=6, color=_c(i), fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(s*100)}%" for s in SPEC_TARGETS], fontsize=10)
    ax.set_xlabel("Specificity", fontsize=11); ax.set_ylabel("Sensitivity", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.axhline(0.8, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.legend(fontsize=8, framealpha=0.88)
    ax.grid(True, axis="y", alpha=0.2)
    plt.tight_layout()
    save(fig, out_dir, fname)


# ─────────────────────────────────────────────────────────────────────────────
# Shared: AUROC vs involvement threshold  (one file per subgroup × variant)
# Uses the CORRECT approach: new binary label = involvement >= thresh
# ─────────────────────────────────────────────────────────────────────────────

def plot_auroc_vs_threshold(data, score_col, head_label, subgroup, variant_label, out_dir, fname):
    """
    For each involvement threshold T (%), define a new binary label:
        y_new = 1  if involvement >= T/100
        y_new = 0  otherwise  (includes all benign + low-involvement cancer)
    Then compute AUROC(y_new, model_score) to see how well the model
    separates cores with >= T% involvement from everything else.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (name, df) in enumerate(data.items()):
        if score_col not in df.columns: continue
        # Respect subgroup masks consistently (including benign negatives where defined).
        sub = _subgroup(df, subgroup).copy()
        sub = sub.dropna(subset=[score_col])
        scores   = sub[score_col].values
        involv   = sub["involvement"].values
        aucs, xs = [], []
        for tp in INV_THRESH_PCT:
            t = tp / 100.0
            new_labels = (involv >= t).astype(int)
            if 0 < new_labels.sum() < len(new_labels):
                aucs.append(_auc(new_labels, scores) * 100)
                xs.append(tp)
        if xs:
            ax.plot(xs, aucs, color=_c(i), ls=_s(i), lw=2.2,
                    marker=_m(i), markersize=7, label=name)
    ax.set_xlabel("Involvement Threshold (%)", fontsize=11)
    ax.set_ylabel("AUROC (%)", fontsize=11)
    ax.legend(fontsize=8, framealpha=0.88, loc="lower right")
    ax.set_xlim(-2, max(INV_THRESH_PCT)+3); ax.set_ylim(None, None)
    ax.grid(True, alpha=0.25, ls="--")
    plt.tight_layout()
    save(fig, out_dir, fname)


# ─────────────────────────────────────────────────────────────────────────────
# Shared: Prediction (activation) vs involvement bins
# ─────────────────────────────────────────────────────────────────────────────

def plot_prediction_vs_involvement(data, score_col, head_label, variant_label, out_dir, fname):
    bins = np.array([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.01])
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (name, df) in enumerate(data.items()):
        if score_col not in df.columns: continue
        sub = df.dropna(subset=[score_col])
        means, sems, xs = [], [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            b = sub[(sub["involvement"] >= lo) & (sub["involvement"] < hi)]
            if len(b) >= 3:
                v = b[score_col].values
                means.append(np.nanmean(v))
                sems.append(np.nanstd(v) / np.sqrt(len(v)))
                xs.append((lo+hi)/2*100)
        if not xs: continue
        xs = np.array(xs); means = np.array(means); sems = np.array(sems)
        ax.plot(xs, means, color=_c(i), ls=_s(i), lw=2, marker=_m(i), ms=5, label=name)
        ax.fill_between(xs, means-sems, means+sems, color=_c(i), alpha=0.10)
    ax.axvline(40, color="gray", ls="--", lw=1, alpha=0.5)
    ax.text(41, 0.02, "High-Inv ↑", fontsize=8, color="gray")
    ax.set_xlabel("True Involvement (%)", fontsize=11)
    ax.set_ylabel("Mean Predicted Score ± SEM", fontsize=11)
    ax.set_xlim(-2, 102); ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, framealpha=0.88)
    ax.grid(True, alpha=0.22)
    plt.tight_layout()
    save(fig, out_dir, fname)


# ─────────────────────────────────────────────────────────────────────────────
# Shared: Score distributions (violin)
# ─────────────────────────────────────────────────────────────────────────────

def plot_score_distributions(data, score_col, head_label, subgroup, out_dir, fname):
    n = len(data)
    fig, ax = plt.subplots(figsize=(max(8, n*2.5), 5))
    for i, (name, df) in enumerate(data.items()):
        if score_col not in df.columns: continue
        sub = _subgroup(df, subgroup).dropna(subset=[score_col])
        benign = sub[sub["label"]==0][score_col].values
        cancer = sub[sub["label"]==1][score_col].values
        for scores, xpos, hatch, alpha in [
                (benign, i*3,     "///", 0.5),
                (cancer, i*3+1.1, "",    0.85)]:
            if len(scores) > 1:
                vp = ax.violinplot([scores], positions=[xpos], widths=0.9,
                                   showmedians=True, showextrema=False)
                for pc in vp["bodies"]:
                    pc.set_facecolor(_c(i)); pc.set_alpha(alpha)
                    if hatch: pc.set_hatch(hatch); pc.set_edgecolor("white")
                vp["cmedians"].set_color("white"); vp["cmedians"].set_linewidth(2)
        ax.text(i*3+0.55, -0.07, name, ha="center", va="top",
                fontsize=7, rotation=30, color=_c(i))
    legend_el = [Patch(fc="gray",alpha=0.5,hatch="///",label="Benign"),
                 Patch(fc="gray",alpha=0.85,label="Cancer")]
    ax.legend(handles=legend_el, fontsize=9, loc="upper left")
    ax.set_xlim(-1, n*3); ax.set_ylim(-0.1, 1.05)
    ax.set_xticks([]); ax.set_ylabel("Predicted Score", fontsize=11)
    ax.grid(True, axis="y", alpha=0.22)
    plt.tight_layout()
    save(fig, out_dir, fname)


# ─────────────────────────────────────────────────────────────────────────────
# Shared: Error by involvement bins
# ─────────────────────────────────────────────────────────────────────────────

def plot_error_by_involvement(data, score_col, head_label, subgroup, variant_label, out_dir, fname):
    inv_bins  = [(0,.2),(.2,.4),(.4,.6),(.6,.8),(.8,1.0)]
    all_bins  = inv_bins + [(0.0,1.0)]
    x = np.arange(len(all_bins))
    n = len(data); width = 0.8 / n
    fig, ax = plt.subplots(figsize=(12, 5))
    # sample counts from first model
    first_df = next(iter(data.values()))
    for i, (name, df) in enumerate(data.items()):
        if score_col not in df.columns: continue
        sub = _subgroup(df, subgroup).dropna(subset=[score_col,"involvement"])
        errors = []
        for lo, hi in all_bins:
            if (lo,hi) == (0.0,1.0):
                b = sub
            else:
                b = sub[(sub["involvement"] >= lo) & (sub["involvement"] < hi)]
            if len(b) > 0:
                errors.append(np.abs(b[score_col].values - b["involvement"].values).mean())
            else:
                errors.append(np.nan)
        offset = (i - n/2 + 0.5)*width
        bars = ax.bar(x+offset, errors, width=width*0.9, color=_c(i),
                      alpha=0.85, label=name, edgecolor="white", lw=0.5)
        for bar, v in zip(bars, errors):
            if not np.isnan(v):
                ax.text(bar.get_x()+bar.get_width()/2, v+0.005, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=6.5, rotation=45, color=_c(i))
    xlbls = [f"{int(lo*100)}–{int(hi*100)}%" for lo,hi in inv_bins] + ["Overall"]
    ax.set_xticks(x); ax.set_xticklabels(xlbls, fontsize=9)
    ax.axvline(len(inv_bins)-0.5, color="gray", ls="--", lw=0.8, alpha=0.5)
    # sample counts
    fsub = _subgroup(first_df, subgroup)
    for j,(lo,hi) in enumerate(all_bins):
        b = fsub if (lo,hi)==(0.0,1.0) else fsub[(fsub["involvement"]>=lo)&(fsub["involvement"]<hi)]
        ax.annotate(f"n={len(b)}", xy=(j,0), xycoords=("data","axes fraction"),
                    xytext=(0,-22), textcoords="offset points",
                    ha="center", va="top", fontsize=7, color="gray")
    ax.set_xlabel("True Involvement Range", fontsize=11)
    ax.set_ylabel("MAE (|Pred − Involvement|)", fontsize=11)
    ax.legend(fontsize=8, framealpha=0.88)
    ax.grid(True, axis="y", alpha=0.22)
    plt.tight_layout()
    save(fig, out_dir, fname)


# ─────────────────────────────────────────────────────────────────────────────
# CLF head: Accuracy by involvement bins
#
# Each bin = all cores with involvement in [lo, hi).
# Benign cores (involvement≈0) land naturally in the 0–20% bin.
# Higher bins (≥20%) are cancer-dominated, so accuracy ≈ sensitivity there.
# "Overall" bar always uses balanced_accuracy_score (benign >> cancer globally).
# For high_inv / cspca subgroups, involvement bins start at 40%.
# ─────────────────────────────────────────────────────────────────────────────

def plot_accuracy_by_involvement(data, score_col, subgroup, out_dir, fname,
                                 balanced=False, fixed_thresh=None):
    # Bins: for focused subgroups skip trivial low-involvement range
    if subgroup in ("high_inv", "cspca"):
        inv_bins = [(0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    else:
        inv_bins = [(0, .2), (.2, .4), (.4, .6), (.6, .8), (.8, 1.0)]
    all_bins = inv_bins + [(0.0, 1.0)]
    x = np.arange(len(all_bins))
    n = len(data); width = 0.8 / n
    fig, ax = plt.subplots(figsize=(max(9, len(all_bins) * 2), 6))

    # Compute per-model thresholds on FULL dataset for a fair comparison
    if fixed_thresh is not None:
        opt_thresh = {name: fixed_thresh for name in data}
        thresh_legend = f"fixed t={fixed_thresh:.2f}"
    else:
        opt_thresh = {}
        for name, df in data.items():
            if score_col not in df.columns: continue
            full = df.dropna(subset=[score_col, "label"])
            t, _ = _opt_threshold(full["label"].values, full[score_col].values, "balanced_accuracy")
            opt_thresh[name] = t
        thresh_legend = "optimal t"

    first_df = next(iter(data.values()))
    for i, (name, df) in enumerate(data.items()):
        if score_col not in df.columns: continue
        t = opt_thresh.get(name, 0.5)
        sub = _subgroup(df, subgroup).dropna(subset=[score_col, "label", "involvement"])
        accs = []
        for lo, hi in all_bins:
            if (lo, hi) == (0.0, 1.0):
                b = sub
                if len(b) < 2 or len(np.unique(b["label"].values)) < 2:
                    accs.append(np.nan); continue
                yp = (b[score_col].values >= t).astype(int)
                # Overall always uses balanced accuracy to account for class imbalance
                accs.append(balanced_accuracy_score(b["label"].values, yp) * 100)
            else:
                b = sub[(sub["involvement"] >= lo) & (sub["involvement"] < hi)]
                if len(b) < 2:
                    accs.append(np.nan); continue
                yp = (b[score_col].values >= t).astype(int)
                has_both = len(np.unique(b["label"].values)) > 1
                if balanced and has_both:
                    accs.append(balanced_accuracy_score(b["label"].values, yp) * 100)
                else:
                    accs.append(accuracy_score(b["label"].values, yp) * 100)
        offset = (i - n / 2 + 0.5) * width
        bars = ax.bar(x + offset, accs, width=width * 0.9, color=_c(i),
                      alpha=0.85, label=f"{name} (t={t:.2f})", edgecolor="white", lw=0.5)
        for bar, v in zip(bars, accs):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5, f"{v:.1f}",
                        ha="center", va="bottom", fontsize=6, rotation=45, color=_c(i))

    xlbls = [f"{int(lo*100)}\u2013{int(hi*100)}%" for lo, hi in inv_bins] + ["Overall\n(bal.acc)"]
    ax.set_xticks(x); ax.set_xticklabels(xlbls, fontsize=9)
    ax.axvline(len(inv_bins) - 0.5, color="gray", ls="--", lw=0.8, alpha=0.5)

    # Sample counts from first model
    fsub = _subgroup(first_df, subgroup).dropna(subset=["involvement"])
    for j, (lo, hi) in enumerate(all_bins):
        b = fsub if (lo, hi) == (0.0, 1.0) else fsub[
            (fsub["involvement"] >= lo) & (fsub["involvement"] < hi)]
        ax.annotate(f"n={len(b)}", xy=(j, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -22), textcoords="offset points",
                    ha="center", va="top", fontsize=7, color="gray")

    metric_name = "Balanced Accuracy" if balanced else "Accuracy"
    suffix = f" [{thresh_legend}]"
    ax.set_xlabel("True Involvement Range", fontsize=11)
    ax.set_ylabel(f"{metric_name} (%)", fontsize=11)
    ax.set_ylim(0, 115)
    ax.legend(fontsize=8, framealpha=0.88)
    ax.grid(True, axis="y", alpha=0.22)
    plt.tight_layout()
    save(fig, out_dir, fname)

# ─────────────────────────────────────────────────────────────────────────────
# CLF head: Threshold analysis — balanced accuracy + F1 vs threshold
# ─────────────────────────────────────────────────────────────────────────────

def plot_threshold_analysis(data, score_col, out_dir, fname):
    threshs = np.linspace(0.01, 0.99, 99)
    fig, axes = plt.subplots(2, 1, figsize=(10, 9))
    opt_info = {}
    for i, (name, df) in enumerate(data.items()):
        if score_col not in df.columns: continue
        sub = df.dropna(subset=[score_col,"label"])
        s = sub[score_col].values; y = sub["label"].values
        bal_accs = [balanced_accuracy_score(y,(s>=t).astype(int)) for t in threshs]
        f1s      = [f1_score(y,(s>=t).astype(int), zero_division=0) for t in threshs]
        t_ba, sc_ba = _opt_threshold(y, s, "balanced_accuracy")
        t_f1, sc_f1 = _opt_threshold(y, s, "f1")
        opt_info[name] = {"t_bal_acc":t_ba,"bal_acc":sc_ba,"t_f1":t_f1,"f1":sc_f1}
        c = _c(i)
        axes[0].plot(threshs, bal_accs, color=c, ls=_s(i), lw=2, label=name)
        axes[0].scatter([t_ba],[sc_ba], s=120, color=c, marker="*", edgecolor="k", lw=1, zorder=10)
        axes[1].plot(threshs, f1s, color=c, ls=_s(i), lw=2, label=name)
        axes[1].scatter([t_f1],[sc_f1], s=120, color=c, marker="*", edgecolor="k", lw=1, zorder=10)
    for ax, ylabel, title in [
        (axes[0],"Balanced Accuracy","Balanced Accuracy vs Threshold (★ = Optimal)"),
        (axes[1],"F1 Score","F1 Score vs Threshold (★ = Optimal)"),
    ]:
        ax.set_xlabel("Classification Threshold", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(fontsize=8, framealpha=0.88)
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.grid(True, alpha=0.22)
    plt.tight_layout()
    save(fig, out_dir, fname)
    # Print summary
    print("\n  Optimal thresholds:")
    for name, info in opt_info.items():
        print(f"    {name}: bal_acc threshold={info['t_bal_acc']:.3f} (score={info['bal_acc']:.3f}) | "
              f"F1 threshold={info['t_f1']:.3f} (score={info['f1']:.3f})")


# ─────────────────────────────────────────────────────────────────────────────
# Shared: Per-sample scatter — involvement vs score, colored by GT label,
#         shape by predicted class (at optimal balanced-accuracy threshold)
# ─────────────────────────────────────────────────────────────────────────────

def plot_involvement_scatter(data, score_col, head_label, out_dir, fname):
    """
    X: true involvement
    Y: predicted score
    Color: ground-truth label (blue=benign, red=cancer)
    Shape: ▲ correctly predicted, ✕ misclassified  (at optimal threshold)
    """
    n = len(data)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5), squeeze=False)
    axes = axes[0]
    for i, (name, df) in enumerate(data.items()):
        ax = axes[i]
        if score_col not in df.columns:
            ax.set_visible(False); continue
        sub = df.dropna(subset=[score_col,"label","involvement"])
        s = sub[score_col].values; y = sub["label"].values
        inv = sub["involvement"].values
        t, _ = _opt_threshold(y, s, "balanced_accuracy")
        yp = (s >= t).astype(int)
        correct = (yp == y)
        # Benign: blue; Cancer: red
        colors_pt  = ["#2166ac" if lb==0 else "#d6604d" for lb in y]
        markers_pt = ["o" if c else "x" for c in correct]
        for marker in ["o","x"]:
            mask = np.array(markers_pt) == marker
            ax.scatter(inv[mask]*100, s[mask], c=np.array(colors_pt)[mask],
                       marker=marker, s=15 if marker=="o" else 30,
                       alpha=0.55, linewidths=0.5 if marker=="o" else 1.2)
        ax.axhline(t, color="k", ls="--", lw=1, alpha=0.6, label=f"Thresh={t:.2f}")
        ax.axvline(40, color="gray", ls=":", lw=0.8, alpha=0.5)
        ax.set_title(f"{name}", fontsize=10, fontweight="bold")
        ax.set_xlabel("True Involvement (%)", fontsize=9)
        ax.set_ylabel("Predicted Score" if i==0 else "", fontsize=9)
        ax.set_xlim(-2,102); ax.set_ylim(-0.05,1.05)
        ax.grid(True, alpha=0.2)
        legend_el = [Line2D([0],[0],marker="o",color="w",markerfacecolor="#2166ac",ms=7,label="Benign"),
                     Line2D([0],[0],marker="o",color="w",markerfacecolor="#d6604d",ms=7,label="Cancer"),
                     Line2D([0],[0],marker="o",color="gray",ms=5,alpha=0.6,label="Correct"),
                     Line2D([0],[0],marker="x",color="gray",ms=7,lw=1.5,label="Wrong")]
        ax.legend(handles=legend_el, fontsize=7, framealpha=0.85, loc="upper left")

    plt.tight_layout()
    save(fig, out_dir, fname)


# ─────────────────────────────────────────────────────────────────────────────
# Patient-level: aggregate max score per patient, then compute patient AUC
# ─────────────────────────────────────────────────────────────────────────────

def _patient_agg(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """Aggregate to patient level: max score, patient has cancer if any core is cancer."""
    g = df.dropna(subset=[score_col,"label","patient_id"]).groupby("patient_id")
    pat = pd.DataFrame({
        "max_score": g[score_col].max(),
        "mean_score": g[score_col].mean(),
        "label": g["label"].max(),  # 1 if any core is cancer
        "max_involvement": g["involvement"].max(),
        "has_cspca": g["grade_group"].apply(lambda x: int((x > 2).any())) if "grade_group" in df.columns else 0,
    }).reset_index()
    return pat


def plot_patient_roc(data, score_col, head_label, out_dir, fname, agg="max"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    score_key = "max_score" if agg=="max" else "mean_score"
    for ax, grp, title in [(axes[0],"all","All Patients"),
                            (axes[1],"cspca","csPCa Patients (GG≥3)")]:
        for i,(name,df) in enumerate(data.items()):
            if score_col not in df.columns: continue
            pat = _patient_agg(df, score_col)
            if grp == "cspca":
                pat = pat[(pat["has_cspca"]==1) | (pat["label"]==0)]
            pat = pat.dropna(subset=[score_key,"label"])
            fpr, tpr, auc = _roc(pat["label"].values, pat[score_key].values)
            ax.plot(fpr, tpr, color=_c(i), ls=_s(i), lw=2, marker=_m(i),
                    markevery=0.2, ms=5, label=f"{name} (AUC={auc:.3f})")
        ax.plot([0,1],[0,1],"k--",lw=0.8,alpha=0.4)
        ax.set_xlabel("1 − Specificity"); ax.set_ylabel("Sensitivity")
        ax.legend(fontsize=8, loc="lower right", framealpha=0.88)
        ax.set_xlim(0,1); ax.set_ylim(0,1.02)
        ax.grid(True, alpha=0.2)
    plt.tight_layout()
    save(fig, out_dir, fname)


def plot_patient_score_distribution(data, score_col, head_label, out_dir, fname):
    """Violin of patient max scores by cancer/benign patient status."""
    n = len(data)
    fig, ax = plt.subplots(figsize=(max(8,n*2.5), 5))
    for i,(name,df) in enumerate(data.items()):
        if score_col not in df.columns: continue
        pat = _patient_agg(df, score_col)
        benign = pat[pat["label"]==0]["max_score"].values
        cancer = pat[pat["label"]==1]["max_score"].values
        for scores, xpos, hatch, alpha in [(benign,i*3,"///",0.5),(cancer,i*3+1.1,"",0.85)]:
            if len(scores) > 1:
                vp = ax.violinplot([scores], positions=[xpos], widths=0.9,
                                   showmedians=True, showextrema=False)
                for pc in vp["bodies"]:
                    pc.set_facecolor(_c(i)); pc.set_alpha(alpha)
                    if hatch: pc.set_hatch(hatch); pc.set_edgecolor("white")
                vp["cmedians"].set_color("white"); vp["cmedians"].set_linewidth(2)
        ax.text(i*3+0.55, -0.07, name, ha="center", va="top", fontsize=7, rotation=30, color=_c(i))
    legend_el = [Patch(fc="gray",alpha=0.5,hatch="///",label="Benign Patient"),
                 Patch(fc="gray",alpha=0.85,label="Cancer Patient")]
    ax.legend(handles=legend_el, fontsize=9, loc="upper left")
    ax.set_xlim(-1, n*3); ax.set_ylim(-0.1, 1.05)
    ax.set_xticks([]); ax.set_ylabel("Max Core Score per Patient", fontsize=11)
    ax.grid(True, axis="y", alpha=0.22)
    plt.tight_layout()
    save(fig, out_dir, fname)


def plot_patient_sensitivity_bars(data, score_col, head_label, out_dir, fname):
    n = len(data); x = np.arange(len(SPEC_TARGETS)); width = 0.8/n
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, grp, title in [(axes[0],"all","All Patients"),(axes[1],"cspca","csPCa Patients")]:
        for i,(name,df) in enumerate(data.items()):
            if score_col not in df.columns: continue
            pat = _patient_agg(df, score_col)
            if grp == "cspca":
                pat = pat[(pat["has_cspca"]==1) | (pat["label"]==0)]
            pat = pat.dropna(subset=["max_score","label"])
            sens = [_sens_at_spec(pat["label"].values, pat["max_score"].values, s)
                    for s in SPEC_TARGETS]
            offset = (i-n/2+0.5)*width
            bars = ax.bar(x+offset, sens, width=width*0.9, color=_c(i),
                          alpha=0.85, label=name, edgecolor="white", lw=0.5)
            for bar,v in zip(bars,sens):
                if not np.isnan(v):
                    ax.text(bar.get_x()+bar.get_width()/2, v+0.008, f"{v:.2f}",
                            ha="center", va="bottom", fontsize=6, color=_c(i), fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels([f"{int(s*100)}%" for s in SPEC_TARGETS], fontsize=9)
        ax.set_xlabel("Specificity"); ax.set_ylabel("Sensitivity")
        ax.set_ylim(0,1.15); ax.axhline(0.8,color="gray",ls="--",lw=0.8,alpha=0.5)
        ax.legend(fontsize=8,framealpha=0.88); ax.grid(True,axis="y",alpha=0.22)
    plt.tight_layout()
    save(fig, out_dir, fname)


# ─────────────────────────────────────────────────────────────────────────────
# Patient-level: balanced-accuracy-focused plots
# ─────────────────────────────────────────────────────────────────────────────

def _safe_bal_acc(y, yp):
    if len(y) == 0 or len(np.unique(y)) < 2:
        return np.nan
    return balanced_accuracy_score(y, yp)


def plot_patient_balanced_accuracy_curve(
    data, score_col, head_label, out_dir, fname, agg="max"
):
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    threshs = np.linspace(0.01, 0.99, 99)

    for i, (name, df) in enumerate(data.items()):
        if score_col not in df.columns:
            continue
        pat = _patient_agg(df, score_col).dropna(subset=["label"])
        score_key = "max_score" if agg == "max" else "mean_score"
        pat = pat.dropna(subset=[score_key])
        if len(np.unique(pat["label"].values)) < 2:
            continue

        y = pat["label"].values.astype(int)
        s = pat[score_key].values.astype(float)
        bas = np.array([_safe_bal_acc(y, (s >= t).astype(int)) for t in threshs])
        if np.all(np.isnan(bas)):
            continue
        best_idx = int(np.nanargmax(bas))
        ax.plot(
            threshs,
            bas * 100,
            color=_c(i),
            ls=_s(i),
            lw=2.1,
            label=f"{name} (best={bas[best_idx]*100:.1f}% @ t={threshs[best_idx]:.2f})",
        )
        ax.scatter(
            [threshs[best_idx]],
            [bas[best_idx] * 100],
            color=_c(i),
            marker="*",
            s=120,
            edgecolor="k",
            linewidth=0.8,
            zorder=5,
        )

    ax.set_xlabel("Decision Threshold", fontsize=11)
    ax.set_ylabel("Balanced Accuracy (%)", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.22)
    ax.legend(fontsize=8, framealpha=0.88, loc="best")
    plt.tight_layout()
    save(fig, out_dir, fname)


def plot_patient_balanced_accuracy_summary(
    data, score_col, head_label, out_dir, fname, agg="max"
):
    labels = []
    best_bas = []
    best_t = []
    yerrs = []

    for name, df in data.items():
        if score_col not in df.columns:
            continue
        pat = _patient_agg(df, score_col).dropna(subset=["label"])
        score_key = "max_score" if agg == "max" else "mean_score"
        pat = pat.dropna(subset=[score_key])
        if len(np.unique(pat["label"].values)) < 2:
            continue

        y = pat["label"].values.astype(int)
        s = pat[score_key].values.astype(float)
        threshs = np.linspace(0.01, 0.99, 99)
        bas = np.array([_safe_bal_acc(y, (s >= t).astype(int)) for t in threshs])
        if np.all(np.isnan(bas)):
            continue

        idx = int(np.nanargmax(bas))
        t = float(threshs[idx])
        yp = (s >= t).astype(int)
        sens = recall_score(y, yp, zero_division=0)
        spec = recall_score(y, yp, pos_label=0, zero_division=0)

        labels.append(name)
        best_bas.append(float(bas[idx] * 100))
        best_t.append(t)
        yerrs.append(abs(sens - spec) * 50.0)

    if not labels:
        return

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8.5, len(labels) * 1.8), 5.2))
    bars = ax.bar(
        x,
        best_bas,
        yerr=yerrs,
        capsize=3,
        color=[_c(i) for i in range(len(labels))],
        alpha=0.88,
        edgecolor="white",
        linewidth=0.8,
    )
    for i, (b, ba, t) in enumerate(zip(bars, best_bas, best_t)):
        ax.text(
            b.get_x() + b.get_width() / 2,
            ba + 1.2,
            f"{ba:.1f}%\nt={t:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=_c(i),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Balanced Accuracy at Optimal Threshold (%)", fontsize=11)
    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.22)
    ax.axhline(50, color="gray", ls="--", lw=0.9, alpha=0.5)
    plt.tight_layout()
    save(fig, out_dir, fname)


# ─────────────────────────────────────────────────────────────────────────────
# Per-patient checkerboard-style map (model score + biopsy result)
# ─────────────────────────────────────────────────────────────────────────────

def _core_status_from_grade_group(gg):
    if pd.isna(gg):
        return 0
    gg = float(gg)
    if gg <= 0:
        return 0  # Benign / no cancer
    if gg < 3:
        return 1  # isPCa
    return 2      # csPCa


def _summarize_patients_for_selection(df):
    needed = {"patient_id", "grade_group", "involvement", "core_id"}
    if any(c not in df.columns for c in needed):
        return pd.DataFrame(columns=["patient_id", "status", "max_involvement", "n_cores"])
    g = (
        df.dropna(subset=["patient_id"])
        .groupby("patient_id", dropna=True)
        .agg(
            max_grade_group=("grade_group", "max"),
            max_involvement=("involvement", "max"),
            n_cores=("core_id", "nunique"),
        )
        .reset_index()
    )
    g["status"] = g["max_grade_group"].apply(_core_status_from_grade_group)
    return g[["patient_id", "status", "max_involvement", "n_cores"]]


def _select_shared_patients_across_models(data, n_patients_per_group=8):
    """
    Build one consistent patient list per status across all models.
    Selection strategy:
      1) patient must exist in ALL models
      2) patient status must be identical in ALL models
      3) rank by highest mean max-involvement across models (descending)
    """
    summaries = {}
    for name, df in data.items():
        s = _summarize_patients_for_selection(df)
        if len(s) == 0:
            continue
        summaries[name] = s.set_index("patient_id")
    if not summaries:
        return {0: [], 1: [], 2: []}

    # Intersection of patients present in all models
    common_ids = set(next(iter(summaries.values())).index)
    for s in summaries.values():
        common_ids &= set(s.index)
    common_ids = sorted(common_ids)
    if not common_ids:
        return {0: [], 1: [], 2: []}

    selected = {0: [], 1: [], 2: []}
    for status_id in [0, 1, 2]:
        rows = []
        for pid in common_ids:
            statuses = [int(s.loc[pid, "status"]) for s in summaries.values()]
            if any(st != status_id for st in statuses):
                continue
            mean_inv = float(np.mean([float(s.loc[pid, "max_involvement"]) for s in summaries.values()]))
            mean_n = float(np.mean([float(s.loc[pid, "n_cores"]) for s in summaries.values()]))
            rows.append((pid, mean_inv, mean_n))
        rows.sort(key=lambda x: (x[1], x[2], str(x[0])), reverse=True)
        selected[status_id] = [pid for pid, _, _ in rows[:n_patients_per_group]]
    return selected


def _plot_patient_checkerboard_single_model(
    df,
    score_col,
    model_name,
    head_label,
    out_dir,
    fname,
    selected_patients_by_status,
    max_cores=18,
):
    if score_col not in df.columns:
        return
    needed_cols = {"patient_id", "core_id", "grade_group", score_col, "PRI-MUS", "PI-RADS"}
    if any(c not in df.columns for c in needed_cols):
        return

    fig, axes = plt.subplots(1, 3, figsize=(9, 0.13 * 40), sharey=False)
    status_titles = ["No Cancer", "isPCa", "csPCa"]
    risk_colors = ["#b2df8a", "#ffffbf", "#fdae61", "#f46d43", "#d73027"]
    cmap = matplotlib.colors.ListedColormap(risk_colors, name="risk_cmap")
    bounds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    rgba_gray = matplotlib.colors.to_rgba("gray", 0.25)

    def _score_to_bin_linear(values):
        v = np.asarray(values, dtype=float)
        if len(v) == 0:
            return np.array([], dtype=float)
        v = np.clip(v, 0.0, 1.0)
        return np.clip(np.floor(v * 5).astype(int) + 1, 1, 5).astype(float)

    def _fit_primus_matched_bin_edges(df_all, score_column):
        """
        Notebook-style binning: match score-bin proportions to PRI-MUS distribution.
        Returns 5 monotonic upper edges (q1..q5), or None if unavailable.
        """
        if "PRI-MUS" not in df_all.columns or score_column not in df_all.columns:
            return None

        fit_df = df_all.dropna(subset=["PRI-MUS", score_column]).copy()
        if len(fit_df) < 10:
            return None

        primus = fit_df["PRI-MUS"].astype(float).values
        scores = fit_df[score_column].astype(float).values
        if len(scores) == 0:
            return None

        counts = (
            pd.Series(primus)
            .value_counts(normalize=True)
            .reindex([1, 2, 3, 4, 5], fill_value=0.0)
        )
        if float(counts.sum()) <= 0:
            return None

        cum_probs = np.cumsum(counts.values)
        cum_probs = np.clip(cum_probs, 0.0, 1.0)
        edges = np.quantile(scores, cum_probs)
        edges = np.maximum.accumulate(edges.astype(float))
        return edges

    def _score_to_bin_primus_matched(values, edges):
        """
        Map continuous scores to bins 1..5 using upper edges q1..q5.
        """
        v = np.asarray(values, dtype=float)
        if len(v) == 0:
            return np.array([], dtype=float)
        if edges is None:
            return _score_to_bin_linear(v)

        bins = np.searchsorted(edges, v, side="left") + 1
        bins = np.clip(bins, 1, 5)
        return bins.astype(float)

    primus_matched_edges = _fit_primus_matched_bin_edges(df, score_col)

    def _gg_to_risk_bin(gg):
        if pd.isna(gg):
            return np.nan
        gg = float(gg)
        if gg == 0:
            return 1
        if gg < 3:
            return 3
        return 5

    for status_id, ax in enumerate(axes):
        pats = selected_patients_by_status.get(status_id, [])
        n_rows = max(1, 4 * len(pats))
        mat = np.full((n_rows, max_cores), np.nan, dtype=float)
        row_labels = []

        for i, pid in enumerate(pats):
            sub = df[df["patient_id"] == pid].copy()
            sub = sub.dropna(subset=[score_col, "grade_group", "PRI-MUS", "PI-RADS"])
            sub = sub.sort_values("core_id").head(max_cores)
            if len(sub) == 0:
                row_labels.extend([""] * 4)
                continue

            model_scores = _score_to_bin_primus_matched(
                sub[score_col].astype(float).values,
                primus_matched_edges,
            )
            human_scores = sub["PRI-MUS"].astype(float).values
            pi_rads = sub["PI-RADS"].astype(float).values
            gt_scores = sub["grade_group"].apply(_gg_to_risk_bin).astype(float).values

            mat[i * 4, : len(model_scores)] = model_scores
            mat[i * 4 + 1, : len(human_scores)] = human_scores
            mat[i * 4 + 2, : len(pi_rads)] = pi_rads
            mat[i * 4 + 3, : len(gt_scores)] = gt_scores

            if i == 0 and status_id == 0:
                row_labels.extend(["Model", "PRI-MUS", "PI-RADS", "Path. GT"])
            else:
                row_labels.append(str(pid))
                row_labels.extend([""] * 3)

        if len(pats) == 0:
            row_labels = [""]

        ax.imshow(
            mat,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            norm=norm,
            vmin=1,
            vmax=5,
        )

        rows = mat.shape[0]
        for y in range(0, rows + 4, 4):
            ax.axhline(y - 0.5, color="black", lw=4)
        for y in range(3, rows + 4, 4):
            ax.axhline(y + 0.5, color="black", lw=1.5, ls="--")

        for x in range(0, max_cores + 1):
            ax.axvline(x - 0.5, color=rgba_gray, lw=0.5)

        ax.set_title(status_titles[status_id], fontsize=11, fontweight="bold")
        ax.set_xlim(-0.5, max_cores - 0.5)
        ax.set_xticks([])
        ax.set_yticks(np.arange(len(row_labels)) + 0.5)
        ax.set_yticklabels(row_labels, fontsize=9, rotation=0)
        ax.tick_params(axis="y", length=0)

    fig.suptitle(f"{head_label} — {model_name}", fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()
    save(fig, out_dir, fname)


def plot_patient_checkerboard_maps(data, score_col, head_label, out_dir, fname_prefix):
    selected = _select_shared_patients_across_models(data, n_patients_per_group=8)
    status_names = {0: "No Cancer", 1: "isPCa", 2: "csPCa"}
    print(f"\n[Checkerboard selection — {head_label}]")
    for status_id in [0, 1, 2]:
        ids = selected.get(status_id, [])
        print(f"  {status_names[status_id]} ({len(ids)}): {', '.join(map(str, ids)) if ids else 'None'}")

    for name, df in data.items():
        safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(name))
        _plot_patient_checkerboard_single_model(
            df=df,
            score_col=score_col,
            model_name=name,
            head_label=head_label,
            out_dir=out_dir,
            fname=f"{fname_prefix}_{safe}",
            selected_patients_by_status=selected,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run(data: Dict[str, pd.DataFrame], output_dir: str):
    hm_dir  = os.path.join(output_dir, "heatmap_head")
    clf_dir = os.path.join(output_dir, "clf_head")
    pat_dir = os.path.join(output_dir, "patient_level")
    sh_dir  = os.path.join(output_dir, "shared")

    # ── Heatmap head normal score ──────────────────────────────────────────
    hm_col = "average_needle_heatmap_value"
    hm_thr = "thresholded_needle_involvement"
    print("\n[Heatmap Head — normal score]")
    for grp in ["all","high_inv","cspca"]:
        print(f"  subgroup: {grp}")
        plot_roc(data, hm_col, "Heatmap Head", grp,
                 hm_dir, f"roc_{grp}")
        plot_sensitivity_bars(data, hm_col, "Heatmap Head", grp,
                              hm_dir, f"sensitivity_bars_{grp}")
        plot_score_distributions(data, hm_col, "Heatmap Head", grp,
                                 hm_dir, f"score_distributions_{grp}")
        plot_error_by_involvement(data, hm_col, "Heatmap Head", grp, "normal",
                                  hm_dir, f"error_by_involvement_{grp}_normal")
        plot_auroc_vs_threshold(data, hm_col, "Heatmap Head", grp, "normal",
                                hm_dir, f"auroc_vs_threshold_{grp}_normal")

    print("\n[Heatmap Head — thresholded score]")
    for grp in ["all","high_inv","cspca"]:
        print(f"  subgroup: {grp}")
        plot_roc(data, hm_thr, "Heatmap Head (Thresh.)", grp,
                 hm_dir, f"roc_{grp}_thresholded")
        plot_sensitivity_bars(data, hm_thr, "Heatmap Head (Thresh.)", grp,
                              hm_dir, f"sensitivity_bars_{grp}_thresholded")
        plot_error_by_involvement(data, hm_thr, "Heatmap Head (Thresh.)", grp, "thresholded",
                                  hm_dir, f"error_by_involvement_{grp}_thresholded")
        plot_auroc_vs_threshold(data, hm_thr, "Heatmap Head (Thresh.)", grp, "thresholded",
                                hm_dir, f"auroc_vs_threshold_{grp}_thresholded")

    print("\n[Heatmap Head — prediction vs involvement]")
    plot_prediction_vs_involvement(data, hm_col, "Heatmap Head", "continuous",
                                   hm_dir, "prediction_vs_involvement_normal")
    plot_prediction_vs_involvement(data, hm_thr, "Heatmap Head", "thresholded",
                                   hm_dir, "prediction_vs_involvement_thresholded")

    print("\n[Heatmap Head — per-sample scatter]")
    plot_involvement_scatter(data, hm_col, "Heatmap Head", hm_dir, "involvement_scatter")

    # ── Classification head ────────────────────────────────────────────────
    clf_col = "image_level_cancer_logits"
    print("\n[Classification Head]")
    for grp in ["all","high_inv","cspca"]:
        print(f"  subgroup: {grp}")
        plot_roc(data, clf_col, "Classification Head", grp,
                 clf_dir, f"roc_{grp}")
        plot_sensitivity_bars(data, clf_col, "Classification Head", grp,
                              clf_dir, f"sensitivity_bars_{grp}")
        plot_score_distributions(data, clf_col, "Classification Head", grp,
                                 clf_dir, f"score_distributions_{grp}")
        plot_error_by_involvement(data, clf_col, "Classification Head", grp, "",
                                  clf_dir, f"error_by_involvement_{grp}")
        # Keep legacy filename and also save a "_normal" variant for parity with
        # heatmap-head outputs (easier side-by-side lookup by filename).
        plot_auroc_vs_threshold(data, clf_col, "Classification Head", grp, "normal",
                                clf_dir, f"auroc_vs_threshold_{grp}")
        plot_auroc_vs_threshold(data, clf_col, "Classification Head", grp, "normal",
                                clf_dir, f"auroc_vs_threshold_{grp}_normal")
        # Fixed threshold (fair cross-model comparison)
        plot_accuracy_by_involvement(data, clf_col, grp,
                                     clf_dir, f"accuracy_by_involvement_{grp}_fixed0.5",
                                     balanced=False, fixed_thresh=0.5)
        plot_accuracy_by_involvement(data, clf_col, grp,
                                     clf_dir, f"balanced_accuracy_by_involvement_{grp}_fixed0.5",
                                     balanced=True, fixed_thresh=0.5)
        # Per-model optimal threshold
        plot_accuracy_by_involvement(data, clf_col, grp,
                                     clf_dir, f"accuracy_by_involvement_{grp}_optimal",
                                     balanced=False, fixed_thresh=None)
        plot_accuracy_by_involvement(data, clf_col, grp,
                                     clf_dir, f"balanced_accuracy_by_involvement_{grp}_optimal",
                                     balanced=True, fixed_thresh=None)

    print("\n[Classification Head — threshold analysis]")
    plot_threshold_analysis(data, clf_col, clf_dir, "threshold_analysis")
    print("\n[Classification Head — per-sample scatter]")
    plot_involvement_scatter(data, clf_col, "Classification Head", clf_dir, "involvement_scatter")
    print("\n[Classification Head — prediction vs involvement]")
    plot_prediction_vs_involvement(data, clf_col, "Classification Head", "",
                                   clf_dir, "prediction_vs_involvement")

    # ── Patient level ──────────────────────────────────────────────────────
    print("\n[Patient level — heatmap head]")
    plot_patient_roc(data, hm_col, "Heatmap Head", pat_dir, "patient_roc_heatmap")
    plot_patient_score_distribution(data, hm_col, "Heatmap Head",
                                    pat_dir, "patient_score_dist_heatmap")
    plot_patient_sensitivity_bars(data, hm_col, "Heatmap Head",
                                  pat_dir, "patient_sensitivity_bars_heatmap")
    plot_patient_balanced_accuracy_curve(
        data, hm_col, "Heatmap Head", pat_dir, "patient_balanced_accuracy_curve_heatmap", agg="max"
    )
    plot_patient_balanced_accuracy_summary(
        data, hm_col, "Heatmap Head", pat_dir, "patient_balanced_accuracy_summary_heatmap", agg="max"
    )
    plot_patient_checkerboard_maps(
        data, hm_col, "Heatmap Head", pat_dir, "patient_checkerboard_heatmap"
    )

    print("\n[Patient level — classification head]")
    plot_patient_roc(data, clf_col, "Classification Head", pat_dir, "patient_roc_clf")
    plot_patient_score_distribution(data, clf_col, "Classification Head",
                                    pat_dir, "patient_score_dist_clf")
    plot_patient_sensitivity_bars(data, clf_col, "Classification Head",
                                  pat_dir, "patient_sensitivity_bars_clf")
    plot_patient_balanced_accuracy_curve(
        data, clf_col, "Classification Head", pat_dir, "patient_balanced_accuracy_curve_clf", agg="max"
    )
    plot_patient_balanced_accuracy_summary(
        data, clf_col, "Classification Head", pat_dir, "patient_balanced_accuracy_summary_clf", agg="max"
    )
    plot_patient_checkerboard_maps(
        data, clf_col, "Classification Head", pat_dir, "patient_checkerboard_clf"
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, metavar="NAME=PATH")
    ap.add_argument("--output_dir", default="plots/clinical_comparison")
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
    print("Loading models …")
    data = load_all(model_dict)
    if not data:
        print("ERROR: No models loaded."); sys.exit(1)
    run(data, args.output_dir)
    print(f"\nAll figures saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
