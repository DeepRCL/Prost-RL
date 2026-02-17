#!/usr/bin/env python3
"""
Inspect NCT2013 metadata for core/needle length fields.
"""
from __future__ import annotations

import argparse
import os
from typing import Iterable, Optional

import pandas as pd

LENGTH_KEYWORDS = [
    "length",
    "core_length",
    "needle_length",
    "tissue_length",
    "len",
    "mm",
    "cm",
]


def _matches_length_key(column: str) -> bool:
    key = column.lower()
    return any(k in key for k in LENGTH_KEYWORDS)


def _summarize_column(series: pd.Series) -> dict:
    numeric = pd.to_numeric(series, errors="coerce")
    summary = {
        "non_null": int(series.notna().sum()),
        "numeric_non_null": int(numeric.notna().sum()),
        "min": float(numeric.min()) if numeric.notna().any() else None,
        "max": float(numeric.max()) if numeric.notna().any() else None,
        "mean": float(numeric.mean()) if numeric.notna().any() else None,
    }
    return summary


def _print_column_summaries(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for col in columns:
        summary = _summarize_column(df[col])
        print(f"- {col}")
        print(
            "  non_null={non_null} numeric_non_null={numeric_non_null} "
            "min={min} max={max} mean={mean}".format(**summary)
        )


def _inspect_metadata_csv(metadata_path: str, list_all: bool) -> None:
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_path}")

    df = pd.read_csv(metadata_path)
    print(f"Loaded metadata CSV: {metadata_path}")
    print(f"Rows: {len(df)}  Columns: {len(df.columns)}")

    print("All columns:")
    for col in df.columns:
        print(f"- {col}")

    sample_row = df.iloc[0].to_dict() if len(df) else {}
    print("Sample row (first):")
    print(sample_row)

    length_cols = [c for c in df.columns if _matches_length_key(c)]
    if not length_cols:
        print("No columns matching length-like keywords.")
        return

    print("Columns matching length-like keywords:")
    _print_column_summaries(df, length_cols)


def _inspect_h5_attrs(h5_path: str, max_examples: int) -> None:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required to inspect HDF5 files.") from exc

    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    print(f"Inspecting HDF5 attrs: {h5_path}")
    keys = {}
    examples = {}
    with h5py.File(h5_path, "r") as f:
        for idx, core_id in enumerate(f.keys()):
            attrs = dict(f[core_id].attrs)
            for key, value in attrs.items():
                if _matches_length_key(str(key)):
                    keys[key] = keys.get(key, 0) + 1
                    if len(examples.get(key, [])) < max_examples:
                        examples.setdefault(key, []).append(value)
            if idx > 5000 and len(keys) == 0:
                # Avoid scanning huge files if nothing matches early.
                break

    if not keys:
        print("No HDF5 attribute keys matching length-like keywords.")
        return

    print("HDF5 attribute keys matching length-like keywords:")
    for key, count in keys.items():
        sample_vals = examples.get(key, [])
        print(f"- {key} (found in {count} cores) sample={sample_vals}")


def _resolve_metadata_path(root: Optional[str], metadata_csv: Optional[str]) -> str:
    if metadata_csv:
        return metadata_csv

    data_root = root or os.environ.get("EXACTVU_PCA_DATA_ROOT")
    if not data_root:
        raise ValueError(
            "No data root provided. Set EXACTVU_PCA_DATA_ROOT or pass --root."
        )
    return os.path.join(data_root, "nct2013", "metadata_with_approx_psa_density.csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect NCT2013 metadata for core/needle length fields."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Root data directory containing nct2013/ (uses EXACTVU_PCA_DATA_ROOT if omitted).",
    )
    parser.add_argument(
        "--metadata-csv",
        type=str,
        default=None,
        help="Direct path to metadata_with_approx_psa_density.csv",
    )
    parser.add_argument(
        "--h5",
        type=str,
        default=None,
        help="Optional HDF5 file to scan for length-like attrs.",
    )
    parser.add_argument(
        "--list-all",
        action="store_true",
        help="List all metadata columns.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=5,
        help="Max number of sample values to print per HDF5 attr key.",
    )
    args = parser.parse_args()

    metadata_path = _resolve_metadata_path(args.root, args.metadata_csv)
    _inspect_metadata_csv(metadata_path, args.list_all)

    if args.h5:
        _inspect_h5_attrs(args.h5, args.max_examples)


if __name__ == "__main__":
    main()
