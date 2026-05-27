import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate k-fold metrics.json files and compute mean/std."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="checkpoints_supervised_cv",
        help="Root directory to recursively search for metrics.json files.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for aggregated tables. Defaults to <root>/cv_aggregates.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="metrics.json",
        help="Filename pattern to search recursively (default: metrics.json).",
    )
    parser.add_argument(
        "--ddof",
        type=int,
        default=1,
        help="Std degrees of freedom (1 = sample std, 0 = population std).",
    )
    return parser.parse_args()


def _extract_fold_from_path(path: Path) -> str:
    joined = str(path)
    # Require fold/prompt+digit to be at end of a path component (no trailing suffix like -v4)
    match = re.search(r"(?:fold|prompt)[_-]?(\d+)(?=[/\\]|$)", joined, flags=re.IGNORECASE)
    if match:
        return f"fold{match.group(1)}"
    return path.parent.name


def _extract_fold_index_from_path(path: Path) -> int | None:
    joined = str(path)
    # Require fold/prompt+digit to be at end of a path component (no trailing suffix like -v4)
    match = re.search(r"(?:fold|prompt)[_-]?(\d+)(?=[/\\]|$)", joined, flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1))


def _load_metrics_file(path: Path) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}
    numeric_only = {}
    for k, v in data.items():
        if isinstance(v, (int, float)) and np.isfinite(v):
            numeric_only[k] = float(v)
    return numeric_only


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root path does not exist: {root}")

    metrics_paths = sorted(root.rglob(args.pattern))
    allowed_fold_indices = set(range(5))
    metrics_paths = [
        path
        for path in metrics_paths
        if _extract_fold_index_from_path(path) in allowed_fold_indices
        and "broken" not in str(path).lower()
    ]
    if len(metrics_paths) == 0:
        print(
            f"[WARN] No '{args.pattern}' files found under: {root} "
            "for folds fold0..fold4."
        )
        return

    rows: List[Dict[str, float]] = []
    for path in metrics_paths:
        metrics = _load_metrics_file(path)
        if len(metrics) == 0:
            continue
        row: Dict[str, float] = dict(metrics)
        row["__fold__"] = _extract_fold_from_path(path)
        row["__metrics_path__"] = str(path)
        rows.append(row)

    if len(rows) == 0:
        print(f"[WARN] Found files but no numeric metrics in: {root}")
        return

    per_fold_df = pd.DataFrame(rows)
    metric_cols = [
        c for c in per_fold_df.columns if not c.startswith("__")
    ]
    per_fold_df = per_fold_df[["__fold__", "__metrics_path__", *metric_cols]]

    summary_rows = []
    for metric in metric_cols:
        values = per_fold_df[metric].dropna().astype(float).values
        if values.size == 0:
            continue
        std = float(np.std(values, ddof=args.ddof)) if values.size > args.ddof else float("nan")
        summary_rows.append(
            {
                "metric": metric,
                "n_folds": int(values.size),
                "mean": float(np.mean(values)),
                "std": std,
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("metric").reset_index(drop=True)

    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir is not None
        else root / "cv_aggregates"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    per_fold_csv = out_dir / "metrics_per_fold.csv"
    summary_csv = out_dir / "metrics_mean_std.csv"
    summary_json = out_dir / "metrics_mean_std.json"

    per_fold_df.to_csv(per_fold_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)

    print(f"[OK] Found {len(per_fold_df)} metrics files.")
    print(f"[OK] Per-fold metrics: {per_fold_csv}")
    print(f"[OK] Mean/std metrics: {summary_csv}")
    print(f"[OK] Mean/std JSON:   {summary_json}")


if __name__ == "__main__":
    main()
