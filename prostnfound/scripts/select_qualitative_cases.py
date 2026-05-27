import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_SCORE_CANDIDATES = [
    "thresholded_needle_involvement",
    "average_needle_heatmap_value",
    "image_level_cancer_logits",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two metrics_by_core.csv files and select qualitative cases "
            "where model A is clinically better than model B."
        )
    )
    parser.add_argument("--model-a-name", required=True, help="Display name for model A")
    parser.add_argument("--model-b-name", required=True, help="Display name for model B")
    parser.add_argument("--csv-a", required=True, help="Path to model A metrics_by_core.csv")
    parser.add_argument("--csv-b", required=True, help="Path to model B metrics_by_core.csv")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for ranked CSVs and summary JSON",
    )
    parser.add_argument(
        "--score-column",
        default=None,
        help=(
            "Score column for comparison (if not provided, auto-picks first available "
            "from thresholded_needle_involvement, average_needle_heatmap_value, image_level_cancer_logits)."
        ),
    )
    parser.add_argument(
        "--min-cancer-involvement",
        type=float,
        default=0.4,
        help="Minimum involvement to prioritize difficult clinically relevant cancer cases",
    )
    parser.add_argument(
        "--cancer-threshold",
        type=float,
        default=0.5,
        help="Threshold above which prediction is treated as cancer-positive",
    )
    parser.add_argument(
        "--benign-threshold",
        type=float,
        default=0.2,
        help="Threshold below which prediction is treated as benign",
    )
    parser.add_argument(
        "--top-k-cancer-win",
        type=int,
        default=60,
        help="Number of top cancer-win cases to export",
    )
    parser.add_argument(
        "--top-k-benign-win",
        type=int,
        default=40,
        help="Number of top benign-win cases to export",
    )
    parser.add_argument(
        "--top-k-ispca-win",
        type=int,
        default=40,
        help="Number of top isPCa (non-csPCa) wins to export",
    )
    parser.add_argument(
        "--heatmap-root-a",
        default=None,
        help="Optional output root of model A run (contains heatmaps/...)",
    )
    parser.add_argument(
        "--heatmap-root-b",
        default=None,
        help="Optional output root of model B run (contains heatmaps/...)",
    )
    return parser.parse_args()


def _to_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    lowered = series.astype(str).str.lower()
    return lowered.isin({"true", "1", "yes", "y", "t"})


def _coerce_numeric(df: pd.DataFrame, columns: List[str]) -> None:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def _pick_score_column(df_a: pd.DataFrame, df_b: pd.DataFrame, requested: Optional[str]) -> str:
    if requested is not None:
        if requested not in df_a.columns or requested not in df_b.columns:
            raise ValueError(f"Requested score column '{requested}' not present in both CSVs.")
        return requested
    for c in DEFAULT_SCORE_CANDIDATES:
        if c in df_a.columns and c in df_b.columns:
            return c
    raise ValueError(
        "Could not auto-select score column. Tried: "
        + ", ".join(DEFAULT_SCORE_CANDIDATES)
        + ". Pass --score-column explicitly."
    )


def _safe_metadata_pick(row: pd.Series, key: str) -> object:
    a = row.get(f"{key}_a", np.nan)
    b = row.get(f"{key}_b", np.nan)
    if pd.notna(a):
        return a
    return b


def _resolve_heatmap_path(
    heatmap_root: Optional[Path],
    patient_id: str,
    core_id: str,
) -> Tuple[str, bool]:
    if heatmap_root is None:
        return "", False
    base = heatmap_root / "heatmaps" / str(patient_id)
    for ext in ("png", "pdf", "jpg", "jpeg", "webp"):
        p = base / f"{core_id}.{ext}"
        if p.exists():
            return str(p), True
    # Provide expected default path even when missing.
    expected = base / f"{core_id}.png"
    return str(expected), False


def main() -> None:
    args = parse_args()
    csv_a = Path(args.csv_a)
    csv_b = Path(args.csv_b)
    if args.output_dir is None:
        out_dir = csv_a.parent / f"qualitative_vs_{args.model_b_name.replace(' ', '_')}"
    else:
        out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_a = pd.read_csv(csv_a)
    df_b = pd.read_csv(csv_b)

    if "core_id" not in df_a.columns or "core_id" not in df_b.columns:
        raise ValueError("Both CSVs must contain a 'core_id' column.")

    df_a = df_a.drop_duplicates(subset=["core_id"], keep="first").copy()
    df_b = df_b.drop_duplicates(subset=["core_id"], keep="first").copy()

    score_col = _pick_score_column(df_a, df_b, args.score_column)

    needed_numeric = [score_col, "label", "involvement", "grade_group"]
    _coerce_numeric(df_a, needed_numeric)
    _coerce_numeric(df_b, needed_numeric)

    if "clinically_significant" in df_a.columns:
        df_a["clinically_significant"] = _to_bool_series(df_a["clinically_significant"])
    else:
        df_a["clinically_significant"] = False

    if "clinically_significant" in df_b.columns:
        df_b["clinically_significant"] = _to_bool_series(df_b["clinically_significant"])
    else:
        df_b["clinically_significant"] = False

    merged = df_a.merge(df_b, on="core_id", suffixes=("_a", "_b"), how="inner")
    if merged.empty:
        raise ValueError("No overlapping cores found between the two CSV files.")

    merged["score_a"] = merged[f"{score_col}_a"]
    merged["score_b"] = merged[f"{score_col}_b"]
    merged["score_gap"] = merged["score_a"] - merged["score_b"]

    merged["label"] = merged["label_a"].combine_first(merged["label_b"]).fillna(0).astype(int)
    merged["involvement"] = merged["involvement_a"].combine_first(merged["involvement_b"]).fillna(0.0)
    merged["clinically_significant"] = (
        merged["clinically_significant_a"].fillna(False) | merged["clinically_significant_b"].fillna(False)
    )
    merged["grade_group"] = merged["grade_group_a"].combine_first(merged["grade_group_b"])

    # Extra diagnostics requested:
    # 1) For benign cores, does model A produce lower heatmap score than model B?
    merged["benign_a_lower_than_b"] = (merged["label"] == 0) & (merged["score_a"] < merged["score_b"])
    merged["benign_a_lower_than_b_margin"] = np.where(
        merged["label"] == 0,
        merged["score_b"] - merged["score_a"],
        np.nan,
    )

    # isPCa = cancer but not clinically significant (commonly GG1/GG2).
    merged["is_ispca"] = (
        (merged["label"] == 1)
        & (
            (~merged["clinically_significant"])
            | (merged["grade_group"].isin([1, 2]))
        )
    )

    merged["pred_a_pos"] = merged["score_a"] >= args.cancer_threshold
    merged["pred_b_pos"] = merged["score_b"] >= args.cancer_threshold
    merged["pred_a_benign"] = merged["score_a"] <= args.benign_threshold
    merged["pred_b_benign"] = merged["score_b"] <= args.benign_threshold

    # Bucket 1: Cancer (prefer clinically significant + high involvement),
    # model A detects while model B misses.
    cancer_win = merged[
        (merged["label"] == 1)
        & (merged["involvement"] >= args.min_cancer_involvement)
        & (merged["pred_a_pos"])
        & (~merged["pred_b_pos"])
    ].copy()
    cancer_win["rank_score"] = (
        cancer_win["clinically_significant"].astype(int) * 100.0
        + cancer_win["involvement"] * 10.0
        + cancer_win["score_gap"]
    )
    cancer_win = cancer_win.sort_values(
        by=["clinically_significant", "involvement", "score_gap", "score_a"],
        ascending=[False, False, False, False],
    ).head(args.top_k_cancer_win)
    cancer_win["bucket"] = "cancer_high_involvement_modelA_hit_modelB_miss"

    # Bucket 2: Benign, model A keeps benign while model B false-positives.
    benign_win = merged[
        (merged["label"] == 0)
        & (merged["pred_a_benign"])
        & (merged["pred_b_pos"])
    ].copy()
    benign_win["rank_score"] = (
        (benign_win["score_b"] - benign_win["score_a"])
        + benign_win["score_b"]
    )
    benign_win = benign_win.sort_values(
        by=["score_b", "score_gap", "score_a"],
        ascending=[False, True, True],
    ).head(args.top_k_benign_win)
    benign_win["bucket"] = "benign_modelA_correct_modelB_false_positive"

    # Bucket 3: isPCa wins (non-csPCa positives where model A catches and model B misses)
    ispca_win = merged[
        merged["is_ispca"]
        & (merged["pred_a_pos"])
        & (~merged["pred_b_pos"])
    ].copy()
    ispca_win["rank_score"] = (
        ispca_win["involvement"] * 10.0
        + ispca_win["score_gap"]
    )
    ispca_win = ispca_win.sort_values(
        by=["involvement", "score_gap", "score_a"],
        ascending=[False, False, False],
    ).head(args.top_k_ispca_win)
    ispca_win["bucket"] = "ispca_modelA_hit_modelB_miss"

    selected = pd.concat([cancer_win, benign_win, ispca_win], ignore_index=True)

    if selected.empty:
        print("No qualifying cases found with current thresholds.")
        return

    for key in ("patient_id", "loc", "grade"):
        selected[key] = selected.apply(lambda r: _safe_metadata_pick(r, key), axis=1)

    heatmap_root_a = Path(args.heatmap_root_a) if args.heatmap_root_a else None
    heatmap_root_b = Path(args.heatmap_root_b) if args.heatmap_root_b else None

    resolved_a = selected.apply(
        lambda r: _resolve_heatmap_path(heatmap_root_a, str(r["patient_id"]), str(r["core_id"])),
        axis=1,
    )
    selected["heatmap_a_path"] = [x[0] for x in resolved_a]
    selected["heatmap_a_exists"] = [x[1] for x in resolved_a]

    resolved_b = selected.apply(
        lambda r: _resolve_heatmap_path(heatmap_root_b, str(r["patient_id"]), str(r["core_id"])),
        axis=1,
    )
    selected["heatmap_b_path"] = [x[0] for x in resolved_b]
    selected["heatmap_b_exists"] = [x[1] for x in resolved_b]

    keep_cols = [
        "bucket",
        "rank_score",
        "core_id",
        "patient_id",
        "loc",
        "grade",
        "label",
        "involvement",
        "clinically_significant",
        "score_a",
        "score_b",
        "score_gap",
        "benign_a_lower_than_b",
        "benign_a_lower_than_b_margin",
        "is_ispca",
        "pred_a_pos",
        "pred_b_pos",
        "pred_a_benign",
        "pred_b_benign",
        "heatmap_a_path",
        "heatmap_b_path",
        "heatmap_a_exists",
        "heatmap_b_exists",
    ]
    selected_out = selected[keep_cols].copy()

    all_path = out_dir / "qualitative_selected_cases.csv"
    cancer_path = out_dir / "qualitative_cancer_wins.csv"
    benign_path = out_dir / "qualitative_benign_wins.csv"
    ispca_path = out_dir / "qualitative_ispca_wins.csv"
    selected_out.to_csv(all_path, index=False)
    selected_out[selected_out["bucket"] == "cancer_high_involvement_modelA_hit_modelB_miss"].to_csv(
        cancer_path, index=False
    )
    selected_out[selected_out["bucket"] == "benign_modelA_correct_modelB_false_positive"].to_csv(
        benign_path, index=False
    )
    selected_out[selected_out["bucket"] == "ispca_modelA_hit_modelB_miss"].to_csv(
        ispca_path, index=False
    )

    benign_all = merged[merged["label"] == 0].copy()
    benign_lower_count = int(benign_all["benign_a_lower_than_b"].sum())
    benign_total_count = int(len(benign_all))
    benign_lower_pct = float(benign_lower_count / benign_total_count) if benign_total_count > 0 else 0.0

    summary = {
        "model_a_name": args.model_a_name,
        "model_b_name": args.model_b_name,
        "csv_a": str(csv_a),
        "csv_b": str(csv_b),
        "score_column": score_col,
        "min_cancer_involvement": args.min_cancer_involvement,
        "cancer_threshold": args.cancer_threshold,
        "benign_threshold": args.benign_threshold,
        "num_overlap_cores": int(len(merged)),
        "num_cancer_wins": int(len(cancer_win)),
        "num_benign_wins": int(len(benign_win)),
        "num_ispca_wins": int(len(ispca_win)),
        "benign_comparison": {
            "num_benign_total": benign_total_count,
            "num_benign_a_lower_than_b": benign_lower_count,
            "pct_benign_a_lower_than_b": benign_lower_pct,
        },
        "outputs": {
            "all": str(all_path),
            "cancer_wins": str(cancer_path),
            "benign_wins": str(benign_path),
            "ispca_wins": str(ispca_path),
        },
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] Score column: {score_col}")
    print(f"[OK] Overlap cores: {len(merged)}")
    print(f"[OK] Cancer wins selected: {len(cancer_win)}")
    print(f"[OK] Benign wins selected: {len(benign_win)}")
    print(f"[OK] isPCa wins selected: {len(ispca_win)}")
    print(
        "[OK] Benign with lower score in model A: "
        f"{benign_lower_count}/{benign_total_count} ({benign_lower_pct:.1%})"
    )
    print(f"[OK] Wrote: {all_path}")
    print(f"[OK] Wrote: {summary_path}")


if __name__ == "__main__":
    main()
