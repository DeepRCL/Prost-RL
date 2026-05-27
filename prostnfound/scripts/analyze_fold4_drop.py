import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


FOLD_RE = re.compile(r"(?:fold|prompt)[_-]?(\d+)", flags=re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze why fold4 underperforms by comparing it against other folds "
            "for both supervised and PNF+ runs."
        )
    )
    parser.add_argument(
        "--supervised-root",
        type=str,
        default="prostnfound/outputs_v2/cv_eval",
        help="Root directory containing supervised fold outputs.",
    )
    parser.add_argument(
        "--pnf-root",
        type=str,
        default="checkpoints_pnf+",
        help="Root directory containing PNF+ fold outputs.",
    )
    parser.add_argument(
        "--target-fold",
        type=int,
        default=4,
        help="Fold index to explain (default: 4).",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Folds to include in analysis.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="prostnfound/outputs_v2/fold4_analysis",
        help="Directory where analysis tables are saved.",
    )
    return parser.parse_args()


def extract_fold_index(path: Path) -> int | None:
    match = FOLD_RE.search(str(path))
    if not match:
        return None
    return int(match.group(1))


def mann_whitney_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    valid = np.isfinite(y_score)
    y_true = y_true[valid]
    y_score = y_score[valid]

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]
    ranks = np.empty_like(sorted_scores, dtype=float)
    ranks[:] = np.arange(1, len(sorted_scores) + 1, dtype=float)

    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            mean_rank = 0.5 * (i + 1 + j + 1)
            ranks[i : j + 1] = mean_rank
        i = j + 1

    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(len(order))
    full_ranks = ranks[inv_order]

    rank_sum_pos = float(full_ranks[y_true == 1].sum())
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def safe_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def load_metric_table(root: Path) -> pd.DataFrame:
    table_path = root / "cv_aggregates" / "metrics_per_fold.csv"
    if not table_path.exists():
        raise FileNotFoundError(f"Could not find metrics table: {table_path}")
    return pd.read_csv(table_path)


def metric_drop_table(
    metrics_df: pd.DataFrame,
    target_fold: int,
    ignore_tokens: tuple[str, ...] = ("err", "entropy", "pvalue", "infer_time"),
) -> pd.DataFrame:
    df = metrics_df.copy()
    if "__fold__" not in df.columns:
        raise ValueError("metrics_per_fold.csv must contain __fold__ column.")

    df["fold_idx"] = (
        df["__fold__"]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)
        .astype(float)
        .astype("Int64")
    )
    df = df[df["fold_idx"].notna()].copy()
    df["fold_idx"] = df["fold_idx"].astype(int)

    metric_cols = [c for c in df.columns if c not in {"__fold__", "__metrics_path__", "fold_idx"}]
    rows = []
    for m in metric_cols:
        if any(token in m.lower() for token in ignore_tokens):
            continue
        target_vals = pd.to_numeric(df.loc[df["fold_idx"] == target_fold, m], errors="coerce").dropna()
        other_vals = pd.to_numeric(df.loc[df["fold_idx"] != target_fold, m], errors="coerce").dropna()
        if len(target_vals) == 0 or len(other_vals) == 0:
            continue

        target_val = float(target_vals.mean())
        other_mean = float(other_vals.mean())
        other_std = float(other_vals.std(ddof=1)) if len(other_vals) > 1 else float("nan")
        delta = target_val - other_mean
        zscore = delta / other_std if np.isfinite(other_std) and other_std > 0 else float("nan")
        rows.append(
            {
                "metric": m,
                "target_fold_value": target_val,
                "other_folds_mean": other_mean,
                "other_folds_std": other_std,
                "delta_target_minus_others": delta,
                "zscore_vs_others": zscore,
            }
        )
    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    return out.sort_values("delta_target_minus_others").reset_index(drop=True)


def discover_fold_files(root: Path, filename: str, allowed_folds: set[int]) -> dict[int, Path]:
    discovered: dict[int, Path] = {}
    for path in root.rglob(filename):
        fold = extract_fold_index(path)
        if fold is None or fold not in allowed_folds:
            continue
        discovered[fold] = path
    return discovered


def per_fold_case_stats(per_fold_csv: dict[int, Path]) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for fold, csv_path in sorted(per_fold_csv.items()):
        df = pd.read_csv(csv_path)
        label = safe_numeric(df, "label")
        involvement = safe_numeric(df, "involvement")
        topk = safe_numeric(df, "topk_score")
        logits = safe_numeric(df, "image_level_cancer_logits")
        cs = safe_numeric(df, "clinically_significant")

        pos_mask = label == 1
        neg_mask = label == 0
        pos_inv = involvement[pos_mask & involvement.notna()]

        center_tvd = float("nan")
        rows.append(
            {
                "fold": fold,
                "n_cores": int(len(df)),
                "n_patients": int(df["patient_id"].nunique()) if "patient_id" in df.columns else np.nan,
                "positive_rate": float(label.mean()) if len(label) else float("nan"),
                "clin_sig_rate": float(cs.mean()) if len(cs) else float("nan"),
                "pos_involvement_mean": float(pos_inv.mean()) if len(pos_inv) else float("nan"),
                "pos_involvement_median": float(pos_inv.median()) if len(pos_inv) else float("nan"),
                "pos_involvement_le_0p10": (
                    float((pos_inv <= 0.10).mean()) if len(pos_inv) else float("nan")
                ),
                "pos_involvement_le_0p20": (
                    float((pos_inv <= 0.20).mean()) if len(pos_inv) else float("nan")
                ),
                "topk_auc_recomputed": mann_whitney_auc(label.values, topk.values),
                "topk_pos_mean": float(topk[pos_mask].mean()) if pos_mask.any() else float("nan"),
                "topk_neg_mean": float(topk[neg_mask].mean()) if neg_mask.any() else float("nan"),
                "topk_gap_pos_minus_neg": (
                    float(topk[pos_mask].mean() - topk[neg_mask].mean())
                    if pos_mask.any() and neg_mask.any()
                    else float("nan")
                ),
                "logit_auc_recomputed": mann_whitney_auc(label.values, logits.values),
                "logit_pos_mean": float(logits[pos_mask].mean()) if pos_mask.any() else float("nan"),
                "logit_neg_mean": float(logits[neg_mask].mean()) if neg_mask.any() else float("nan"),
                "logit_gap_pos_minus_neg": (
                    float(logits[pos_mask].mean() - logits[neg_mask].mean())
                    if pos_mask.any() and neg_mask.any()
                    else float("nan")
                ),
                "csv_path": str(csv_path),
                "center_tvd_vs_other_folds": center_tvd,
            }
        )

    out = pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)
    out["center_tvd_vs_other_folds"] = compute_center_tvd(per_fold_csv)
    return out


def compute_center_tvd(per_fold_csv: dict[int, Path]) -> pd.Series:
    center_hist_by_fold: dict[int, pd.Series] = {}
    for fold, csv_path in per_fold_csv.items():
        df = pd.read_csv(csv_path)
        if "center" not in df.columns:
            center_hist_by_fold[fold] = pd.Series(dtype=float)
            continue
        hist = df["center"].astype(str).value_counts(normalize=True)
        center_hist_by_fold[fold] = hist

    tvd_values = {}
    for fold, hist in center_hist_by_fold.items():
        other_folds = [f for f in center_hist_by_fold.keys() if f != fold]
        if not other_folds or len(hist) == 0:
            tvd_values[fold] = np.nan
            continue
        pooled_counts = pd.concat([center_hist_by_fold[f] for f in other_folds], axis=1).fillna(0.0).mean(axis=1)
        all_centers = sorted(set(hist.index).union(set(pooled_counts.index)))
        p = hist.reindex(all_centers, fill_value=0.0).values
        q = pooled_counts.reindex(all_centers, fill_value=0.0).values
        tvd_values[fold] = 0.5 * np.abs(p - q).sum()

    return pd.Series([tvd_values.get(fold, np.nan) for fold in sorted(per_fold_csv.keys())])


def fold_vs_others_delta(stats_df: pd.DataFrame, target_fold: int) -> pd.DataFrame:
    numeric_cols = [
        c
        for c in stats_df.columns
        if c not in {"fold", "csv_path"} and pd.api.types.is_numeric_dtype(stats_df[c])
    ]
    target_row = stats_df[stats_df["fold"] == target_fold]
    other_rows = stats_df[stats_df["fold"] != target_fold]
    if len(target_row) == 0 or len(other_rows) == 0:
        return pd.DataFrame()

    rows = []
    for col in numeric_cols:
        t = float(target_row[col].iloc[0])
        o = pd.to_numeric(other_rows[col], errors="coerce").dropna()
        if len(o) == 0:
            continue
        o_mean = float(o.mean())
        o_std = float(o.std(ddof=1)) if len(o) > 1 else float("nan")
        delta = t - o_mean
        z = delta / o_std if np.isfinite(o_std) and o_std > 0 else float("nan")
        rows.append(
            {
                "feature": col,
                "target_fold_value": t,
                "other_folds_mean": o_mean,
                "other_folds_std": o_std,
                "delta_target_minus_others": delta,
                "zscore_vs_others": z,
            }
        )
    return pd.DataFrame(rows).sort_values("delta_target_minus_others").reset_index(drop=True)


def load_all_fold_cases(per_fold_csv: dict[int, Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for fold, csv_path in sorted(per_fold_csv.items()):
        df = pd.read_csv(csv_path).copy()
        df["fold"] = int(fold)
        df["csv_path"] = str(csv_path)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def add_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["label", "involvement", "topk_score", "image_level_cancer_logits"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "clinically_significant" in out.columns:
        cs = out["clinically_significant"]
        if cs.dtype == bool:
            out["clinically_significant"] = cs.astype(float)
        else:
            out["clinically_significant"] = (
                cs.astype(str)
                .str.lower()
                .map({"true": 1.0, "false": 0.0})
                .fillna(pd.to_numeric(cs, errors="coerce"))
            )
    return out


def subgroup_shift_table(
    all_cases_df: pd.DataFrame,
    target_fold: int,
    group_col: str,
    min_group_count: int = 20,
) -> pd.DataFrame:
    if group_col not in all_cases_df.columns:
        return pd.DataFrame()

    df = add_numeric_columns(all_cases_df)
    df[group_col] = df[group_col].astype(str)
    metric_cols = ["topk_score", "image_level_cancer_logits"]
    rows: list[dict[str, float | str]] = []

    for group_value, gdf in df.groupby(group_col):
        t = gdf[gdf["fold"] == target_fold]
        o = gdf[gdf["fold"] != target_fold]
        if len(t) < min_group_count or len(o) < min_group_count:
            continue

        t_label = pd.to_numeric(t["label"], errors="coerce")
        o_label = pd.to_numeric(o["label"], errors="coerce")
        t_pos = t_label == 1
        t_neg = t_label == 0
        o_pos = o_label == 1
        o_neg = o_label == 0

        row: dict[str, float | str] = {
            "group_col": group_col,
            "group_value": group_value,
            "target_n": int(len(t)),
            "other_n": int(len(o)),
            "target_positive_rate": float(t_label.mean()),
            "other_positive_rate": float(o_label.mean()),
            "delta_positive_rate": float(t_label.mean() - o_label.mean()),
        }
        if "involvement" in t.columns:
            t_pos_inv = pd.to_numeric(t.loc[t_pos, "involvement"], errors="coerce")
            o_pos_inv = pd.to_numeric(o.loc[o_pos, "involvement"], errors="coerce")
            row["target_pos_inv_mean"] = float(t_pos_inv.mean()) if len(t_pos_inv) else float("nan")
            row["other_pos_inv_mean"] = float(o_pos_inv.mean()) if len(o_pos_inv) else float("nan")
            row["delta_pos_inv_mean"] = row["target_pos_inv_mean"] - row["other_pos_inv_mean"]

        for score_col in metric_cols:
            if score_col not in t.columns:
                continue
            t_score = pd.to_numeric(t[score_col], errors="coerce")
            o_score = pd.to_numeric(o[score_col], errors="coerce")

            row[f"target_{score_col}_auc"] = mann_whitney_auc(t_label.values, t_score.values)
            row[f"other_{score_col}_auc"] = mann_whitney_auc(o_label.values, o_score.values)
            row[f"delta_{score_col}_auc"] = (
                row[f"target_{score_col}_auc"] - row[f"other_{score_col}_auc"]
                if np.isfinite(row[f"target_{score_col}_auc"]) and np.isfinite(row[f"other_{score_col}_auc"])
                else float("nan")
            )
            row[f"target_{score_col}_neg_mean"] = float(t_score[t_neg].mean()) if t_neg.any() else float("nan")
            row[f"other_{score_col}_neg_mean"] = float(o_score[o_neg].mean()) if o_neg.any() else float("nan")
            row[f"delta_{score_col}_neg_mean"] = (
                row[f"target_{score_col}_neg_mean"] - row[f"other_{score_col}_neg_mean"]
            )
            row[f"target_{score_col}_pos_mean"] = float(t_score[t_pos].mean()) if t_pos.any() else float("nan")
            row[f"other_{score_col}_pos_mean"] = float(o_score[o_pos].mean()) if o_pos.any() else float("nan")
            row[f"delta_{score_col}_pos_mean"] = (
                row[f"target_{score_col}_pos_mean"] - row[f"other_{score_col}_pos_mean"]
            )
        rows.append(row)

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    sort_col = "delta_image_level_cancer_logits_auc"
    if sort_col not in out.columns:
        sort_col = "delta_topk_score_auc"
    return out.sort_values(sort_col).reset_index(drop=True)


def threshold_for_specificity(y_true: pd.Series, scores: pd.Series, specificity: float) -> float:
    y = pd.to_numeric(y_true, errors="coerce")
    s = pd.to_numeric(scores, errors="coerce")
    neg_scores = s[(y == 0) & s.notna()]
    if len(neg_scores) == 0:
        return float("nan")
    q = min(max(specificity, 0.0), 1.0)
    return float(neg_scores.quantile(q))


def threshold_failure_table(
    all_cases_df: pd.DataFrame,
    target_fold: int,
    score_col: str,
    target_specificity: float = 0.8,
) -> pd.DataFrame:
    if score_col not in all_cases_df.columns:
        return pd.DataFrame()

    df = add_numeric_columns(all_cases_df)
    other = df[df["fold"] != target_fold]
    target = df[df["fold"] == target_fold]

    thr = threshold_for_specificity(other["label"], other[score_col], specificity=target_specificity)
    if not np.isfinite(thr):
        return pd.DataFrame()

    target_y = pd.to_numeric(target["label"], errors="coerce")
    target_s = pd.to_numeric(target[score_col], errors="coerce")
    pred_pos = target_s >= thr
    pred_neg = target_s < thr
    is_pos = target_y == 1
    is_neg = target_y == 0

    rows: list[dict[str, float | str]] = []
    rows.append(
        {
            "score_col": score_col,
            "threshold_from_other_folds": thr,
            "target_specificity_goal_on_other_folds": target_specificity,
            "fold": target_fold,
            "target_specificity_achieved": float((pred_neg[is_neg]).mean()) if is_neg.any() else float("nan"),
            "target_sensitivity_achieved": float((pred_pos[is_pos]).mean()) if is_pos.any() else float("nan"),
            "target_fpr": float((pred_pos[is_neg]).mean()) if is_neg.any() else float("nan"),
            "target_fnr": float((pred_neg[is_pos]).mean()) if is_pos.any() else float("nan"),
            "n_target_neg": int(is_neg.sum()),
            "n_target_pos": int(is_pos.sum()),
        }
    )
    return pd.DataFrame(rows)


def worst_cases_table(
    all_cases_df: pd.DataFrame,
    target_fold: int,
    score_col: str,
    mode: str,
    top_n: int = 50,
) -> pd.DataFrame:
    df = add_numeric_columns(all_cases_df)
    t = df[df["fold"] == target_fold].copy()
    if score_col not in t.columns or "label" not in t.columns:
        return pd.DataFrame()
    y = pd.to_numeric(t["label"], errors="coerce")
    score = pd.to_numeric(t[score_col], errors="coerce")

    if mode == "false_positive_risk":
        cand = t[y == 0].copy()
        cand["risk_score"] = pd.to_numeric(cand[score_col], errors="coerce")
        cand = cand.sort_values("risk_score", ascending=False)
    elif mode == "false_negative_risk":
        cand = t[y == 1].copy()
        cand["risk_score"] = pd.to_numeric(cand[score_col], errors="coerce")
        cand = cand.sort_values("risk_score", ascending=True)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    keep_cols = [
        "fold",
        "center",
        "core_id",
        "patient_id",
        "loc",
        "grade",
        "grade_group",
        "label",
        "involvement",
        "clinically_significant",
        "topk_score",
        "image_level_cancer_logits",
        "risk_score",
    ]
    keep_cols = [c for c in keep_cols if c in cand.columns]
    return cand[keep_cols].head(top_n).reset_index(drop=True)


def print_brief_report(
    run_name: str,
    metric_drop_df: pd.DataFrame,
    case_delta_df: pd.DataFrame,
    subgroup_center_df: pd.DataFrame,
    subgroup_loc_df: pd.DataFrame,
    subgroup_grade_df: pd.DataFrame,
    threshold_topk_df: pd.DataFrame,
    threshold_logit_df: pd.DataFrame,
    target_fold: int,
) -> None:
    print(f"\n=== {run_name} | fold{target_fold} analysis ===")
    if len(metric_drop_df) > 0:
        print("\nLargest metric drops (fold vs other-fold mean):")
        cols = ["metric", "target_fold_value", "other_folds_mean", "delta_target_minus_others", "zscore_vs_others"]
        print(metric_drop_df.head(10)[cols].to_string(index=False))
    else:
        print("\nNo metric drop table available.")

    if len(case_delta_df) > 0:
        print("\nPotential explanatory shifts in fold composition/separation:")
        cols = [
            "feature",
            "target_fold_value",
            "other_folds_mean",
            "delta_target_minus_others",
            "zscore_vs_others",
        ]
        print(case_delta_df.head(10)[cols].to_string(index=False))
    else:
        print("\nNo case-level shift table available.")

    for title, sdf in [
        ("center", subgroup_center_df),
        ("location", subgroup_loc_df),
        ("grade_group", subgroup_grade_df),
    ]:
        if len(sdf) > 0:
            keep = [
                "group_value",
                "target_n",
                "other_n",
                "delta_positive_rate",
                "delta_pos_inv_mean",
                "delta_topk_score_auc",
                "delta_image_level_cancer_logits_auc",
                "delta_topk_score_neg_mean",
                "delta_image_level_cancer_logits_neg_mean",
            ]
            keep = [c for c in keep if c in sdf.columns]
            print(f"\nMost degraded subgroups by {title} (target fold vs other folds):")
            print(sdf.head(8)[keep].to_string(index=False))

    for score_name, tdf in [("topk_score", threshold_topk_df), ("image_level_cancer_logits", threshold_logit_df)]:
        if len(tdf) > 0:
            print(f"\nThreshold transfer check ({score_name}, threshold from other folds):")
            print(tdf.to_string(index=False))


def main() -> None:
    args = parse_args()
    target_fold = int(args.target_fold)
    allowed_folds = set(int(f) for f in args.folds)
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    supervised_root = Path(args.supervised_root).expanduser().resolve()
    pnf_root = Path(args.pnf_root).expanduser().resolve()

    supervised_metrics = load_metric_table(supervised_root)
    pnf_metrics = load_metric_table(pnf_root)

    supervised_metric_drop = metric_drop_table(supervised_metrics, target_fold=target_fold)
    pnf_metric_drop = metric_drop_table(pnf_metrics, target_fold=target_fold)

    supervised_case_files = discover_fold_files(supervised_root, "metrics_by_core.csv", allowed_folds)
    pnf_case_files = discover_fold_files(pnf_root, "metrics_by_core.csv", allowed_folds)

    if target_fold not in supervised_case_files:
        raise FileNotFoundError(
            f"Target fold{target_fold} metrics_by_core.csv not found under {supervised_root}"
        )
    if target_fold not in pnf_case_files:
        raise FileNotFoundError(
            f"Target fold{target_fold} metrics_by_core.csv not found under {pnf_root}"
        )

    supervised_case_stats = per_fold_case_stats(supervised_case_files)
    pnf_case_stats = per_fold_case_stats(pnf_case_files)
    supervised_case_delta = fold_vs_others_delta(supervised_case_stats, target_fold)
    pnf_case_delta = fold_vs_others_delta(pnf_case_stats, target_fold)
    supervised_all_cases = load_all_fold_cases(supervised_case_files)
    pnf_all_cases = load_all_fold_cases(pnf_case_files)

    supervised_subgroup_center = subgroup_shift_table(supervised_all_cases, target_fold, "center")
    supervised_subgroup_loc = subgroup_shift_table(supervised_all_cases, target_fold, "loc")
    supervised_subgroup_grade = subgroup_shift_table(supervised_all_cases, target_fold, "grade_group")
    pnf_subgroup_center = subgroup_shift_table(pnf_all_cases, target_fold, "center")
    pnf_subgroup_loc = subgroup_shift_table(pnf_all_cases, target_fold, "loc")
    pnf_subgroup_grade = subgroup_shift_table(pnf_all_cases, target_fold, "grade_group")

    supervised_thr_topk = threshold_failure_table(
        supervised_all_cases, target_fold, "topk_score", target_specificity=0.8
    )
    supervised_thr_logit = threshold_failure_table(
        supervised_all_cases, target_fold, "image_level_cancer_logits", target_specificity=0.8
    )
    pnf_thr_topk = threshold_failure_table(
        pnf_all_cases, target_fold, "topk_score", target_specificity=0.8
    )
    pnf_thr_logit = threshold_failure_table(
        pnf_all_cases, target_fold, "image_level_cancer_logits", target_specificity=0.8
    )

    supervised_fp_topk = worst_cases_table(
        supervised_all_cases, target_fold, "topk_score", mode="false_positive_risk", top_n=80
    )
    supervised_fn_topk = worst_cases_table(
        supervised_all_cases, target_fold, "topk_score", mode="false_negative_risk", top_n=80
    )
    pnf_fp_topk = worst_cases_table(
        pnf_all_cases, target_fold, "topk_score", mode="false_positive_risk", top_n=80
    )
    pnf_fn_topk = worst_cases_table(
        pnf_all_cases, target_fold, "topk_score", mode="false_negative_risk", top_n=80
    )

    supervised_metric_drop.to_csv(out_dir / "supervised_metric_drop.csv", index=False)
    pnf_metric_drop.to_csv(out_dir / "pnf_metric_drop.csv", index=False)
    supervised_case_stats.to_csv(out_dir / "supervised_fold_case_stats.csv", index=False)
    pnf_case_stats.to_csv(out_dir / "pnf_fold_case_stats.csv", index=False)
    supervised_case_delta.to_csv(out_dir / "supervised_fold_case_delta.csv", index=False)
    pnf_case_delta.to_csv(out_dir / "pnf_fold_case_delta.csv", index=False)
    supervised_subgroup_center.to_csv(out_dir / "supervised_subgroup_center_shift.csv", index=False)
    supervised_subgroup_loc.to_csv(out_dir / "supervised_subgroup_loc_shift.csv", index=False)
    supervised_subgroup_grade.to_csv(out_dir / "supervised_subgroup_grade_group_shift.csv", index=False)
    pnf_subgroup_center.to_csv(out_dir / "pnf_subgroup_center_shift.csv", index=False)
    pnf_subgroup_loc.to_csv(out_dir / "pnf_subgroup_loc_shift.csv", index=False)
    pnf_subgroup_grade.to_csv(out_dir / "pnf_subgroup_grade_group_shift.csv", index=False)
    supervised_thr_topk.to_csv(out_dir / "supervised_threshold_transfer_topk.csv", index=False)
    supervised_thr_logit.to_csv(out_dir / "supervised_threshold_transfer_logit.csv", index=False)
    pnf_thr_topk.to_csv(out_dir / "pnf_threshold_transfer_topk.csv", index=False)
    pnf_thr_logit.to_csv(out_dir / "pnf_threshold_transfer_logit.csv", index=False)
    supervised_fp_topk.to_csv(out_dir / "supervised_fold4_top_false_positive_risk_cases.csv", index=False)
    supervised_fn_topk.to_csv(out_dir / "supervised_fold4_top_false_negative_risk_cases.csv", index=False)
    pnf_fp_topk.to_csv(out_dir / "pnf_fold4_top_false_positive_risk_cases.csv", index=False)
    pnf_fn_topk.to_csv(out_dir / "pnf_fold4_top_false_negative_risk_cases.csv", index=False)

    print_brief_report(
        "Supervised",
        supervised_metric_drop,
        supervised_case_delta,
        supervised_subgroup_center,
        supervised_subgroup_loc,
        supervised_subgroup_grade,
        supervised_thr_topk,
        supervised_thr_logit,
        target_fold,
    )
    print_brief_report(
        "PNF+",
        pnf_metric_drop,
        pnf_case_delta,
        pnf_subgroup_center,
        pnf_subgroup_loc,
        pnf_subgroup_grade,
        pnf_thr_topk,
        pnf_thr_logit,
        target_fold,
    )
    print(f"\nSaved analysis tables to: {out_dir}")


if __name__ == "__main__":
    main()
