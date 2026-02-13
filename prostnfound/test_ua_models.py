"""
Batch zero-shot evaluator on UA (Optimum) for ProstNFound(+/RL) checkpoints.

This script calls `test_rl.py` for each configured checkpoint and aggregates the
resulting `metrics.json` files into one JSON + CSV summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


def _to_hydra_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, dict)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def _flatten_overrides(prefix: str, value: Any) -> list[str]:
    overrides: list[str] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            new_prefix = f"{prefix}.{key}" if prefix else str(key)
            overrides.extend(_flatten_overrides(new_prefix, nested))
        return overrides
    overrides.append(f"{prefix}={_to_hydra_value(value)}")
    return overrides


def _apply_hydra_prefix(overrides: list[str], mode: str = "set") -> list[str]:
    """
    Add Hydra override prefix to each `key=value` string.

    mode:
      - "set": use `key=value`
      - "set_or_add": use `++key=value` (works for both existing and missing keys)
    """
    if mode == "set_or_add":
        return [f"++{ov}" for ov in overrides]
    return overrides


def _load_config(path: Path) -> DictConfig:
    cfg = OmegaConf.load(path)
    if not isinstance(cfg, DictConfig):
        raise ValueError(f"Config at {path} is not a DictConfig.")
    return cfg


def _default_optimum_root() -> str | None:
    data_root = os.getenv("EXACTVU_PCA_DATA_ROOT")
    if not data_root:
        return None
    return str(Path(data_root) / "OPTIMUM" / "processed" / "UA_annotated_needles")


def _flatten_metrics(metrics: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in metrics.items():
        new_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_metrics(value, new_key))
        else:
            flat[new_key] = value
    return flat


def _validate_optimum_root(root_dir: str) -> tuple[bool, str]:
    root = Path(root_dir)
    if not root.exists():
        return False, f"Optimum root does not exist: {root}"
    if not root.is_dir():
        return False, f"Optimum root is not a directory: {root}"
    case_dirs = [p for p in root.iterdir() if p.is_dir()]
    if len(case_dirs) == 0:
        return (
            False,
            f"Optimum root has no case directories: {root}. "
            "Expected per-case subfolders (e.g., UA-xxx).",
        )

    # Validate annotated-needle layout expected by NeedleTraceImageFramesDataset:
    #   <root>/<case>/<cine>/image.png + info.json + needle_mask*.png
    valid_cine_dirs = 0
    inspected_cases = 0
    direct_png_cases = 0
    direct_png_total = 0

    for case_dir in case_dirs[:20]:
        inspected_cases += 1
        entries = list(case_dir.iterdir())
        direct_png = [p for p in entries if p.is_file() and p.suffix.lower() == ".png"]
        if direct_png:
            direct_png_cases += 1
            direct_png_total += len(direct_png)

        for cine_dir in entries:
            if not cine_dir.is_dir():
                continue
            image_ok = (cine_dir / "image.png").exists()
            info_ok = (cine_dir / "info.json").exists()
            mask_ok = (cine_dir / "needle_mask.png").exists() or (
                cine_dir / "needle_mask_full.png"
            ).exists()
            if image_ok and info_ok and mask_ok:
                valid_cine_dirs += 1
                if valid_cine_dirs >= 1:
                    break
        if valid_cine_dirs >= 1:
            break

    if valid_cine_dirs == 0:
        if direct_png_cases > 0:
            return (
                False,
                "Optimum root does not look like annotated-needle export expected by "
                "NeedleTraceImageFramesDataset. Detected direct PNG frames under case "
                f"folders ({direct_png_total} PNGs across {direct_png_cases}/{inspected_cases} "
                "inspected cases), but no <case>/<cine>/image.png+info.json+needle_mask*.png "
                f"layout under {root}. Point data.root_dir to UA_annotated_needles root.",
            )
        return (
            False,
            "Could not find any valid annotated cine folders under "
            f"{root}. Expected <case>/<cine>/image.png + info.json + needle_mask*.png.",
        )

    return True, f"Found {len(case_dirs)} case directories under {root} (annotated layout OK)"


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    keys: set[str] = set()
    for row in rows:
        keys.update(row.keys())
    fieldnames = sorted(keys)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run(config_path: Path) -> int:
    cfg = _load_config(config_path)

    script_dir = Path(__file__).resolve().parent
    test_script = script_dir / cfg.get("test_script", "test_rl.py")
    if not test_script.exists():
        raise FileNotFoundError(f"Could not find test script: {test_script}")

    base_output_dir = Path(cfg.get("base_output_dir", "outputs/ua_zeroshot_eval"))
    base_output_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = OmegaConf.to_container(cfg.get("data", {}), resolve=True)
    if not isinstance(data_cfg, dict):
        raise ValueError("`data` in config must be a dictionary.")

    if not data_cfg.get("root_dir"):
        default_root = _default_optimum_root()
        if default_root is None:
            raise ValueError(
                "UA root_dir is missing and EXACTVU_PCA_DATA_ROOT is not set.\n"
                "Set EXACTVU_PCA_DATA_ROOT or provide data.root_dir in config."
            )
        data_cfg["root_dir"] = default_root

    # Preflight check to avoid launching long eval jobs with a wrong root.
    require_nonempty_optimum_root = bool(cfg.get("require_nonempty_optimum_root", True))
    if str(data_cfg.get("dataset", "")).lower() == "optimum":
        ok, msg = _validate_optimum_root(str(data_cfg["root_dir"]))
        if not ok and require_nonempty_optimum_root:
            raise ValueError(msg)
        print(f"[data-check] {msg}")

    # Use set_or_add for nested data overrides because cfg/test_rl.yaml uses
    # struct mode and may not declare all Optimum-specific keys (e.g. root_dir).
    data_overrides = _apply_hydra_prefix(
        _flatten_overrides("data", data_cfg),
        mode="set_or_add",
    )
    common_overrides = list(cfg.get("common_overrides", []))
    split = cfg.get("split", "test")
    fail_fast = bool(cfg.get("fail_fast", False))
    python_exec = str(cfg.get("python", sys.executable))

    models = cfg.get("models", [])
    if not models:
        raise ValueError("No models configured. Add entries under `models`.")

    summary_rows: list[dict[str, Any]] = []

    for idx, model_cfg in enumerate(models):
        model_cfg = OmegaConf.to_container(model_cfg, resolve=True)
        if not isinstance(model_cfg, dict):
            continue

        name = str(model_cfg.get("name", f"model_{idx}"))
        enabled = bool(model_cfg.get("enabled", True))
        checkpoint = model_cfg.get("checkpoint")
        extra_overrides = model_cfg.get("overrides", []) or []
        model_name_override = model_cfg.get("model", None)
        model_kw_override = model_cfg.get("model_kw", None)

        if not enabled:
            print(f"[skip] {name}: disabled")
            continue

        if not checkpoint:
            print(f"[skip] {name}: missing checkpoint")
            continue

        checkpoint_path = Path(str(checkpoint))
        if not checkpoint_path.exists():
            msg = f"[fail] {name}: checkpoint not found -> {checkpoint_path}"
            print(msg)
            summary_rows.append(
                {
                    "model_name": name,
                    "checkpoint": str(checkpoint_path),
                    "status": "checkpoint_missing",
                }
            )
            if fail_fast:
                break
            continue

        run_output_dir = base_output_dir / str(model_cfg.get("output_subdir", name))
        run_output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            python_exec,
            str(test_script),
            f"checkpoint={checkpoint_path}",
            f"output_dir={run_output_dir}",
            f"split={split}",
            *data_overrides,
            *common_overrides,
        ]

        # Optional per-model architecture fallback for legacy checkpoints that
        # may not store model/model_kw in checkpoint args.
        if model_name_override is not None:
            cmd.append(f"model={_to_hydra_value(model_name_override)}")
        if model_kw_override is not None:
            cmd.append(f"model_kw={_to_hydra_value(model_kw_override)}")

        cmd.extend(extra_overrides)

        print(f"[run] {name}")
        print("      " + " ".join(cmd))
        proc = subprocess.run(cmd, cwd=script_dir, check=False)

        metrics_file = run_output_dir / "metrics.json"
        if proc.returncode != 0:
            summary_rows.append(
                {
                    "model_name": name,
                    "checkpoint": str(checkpoint_path),
                    "status": f"failed_rc_{proc.returncode}",
                    "output_dir": str(run_output_dir),
                }
            )
            if fail_fast:
                break
            continue

        if not metrics_file.exists():
            summary_rows.append(
                {
                    "model_name": name,
                    "checkpoint": str(checkpoint_path),
                    "status": "missing_metrics",
                    "output_dir": str(run_output_dir),
                }
            )
            if fail_fast:
                break
            continue

        with metrics_file.open("r") as f:
            metrics = json.load(f)

        row = {
            "model_name": name,
            "checkpoint": str(checkpoint_path),
            "status": "ok",
            "output_dir": str(run_output_dir),
        }
        row.update(_flatten_metrics(metrics))
        summary_rows.append(row)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_json = base_output_dir / f"summary_{timestamp}.json"
    summary_csv = base_output_dir / f"summary_{timestamp}.csv"

    with summary_json.open("w") as f:
        json.dump(summary_rows, f, indent=2)
    _write_csv(summary_rows, summary_csv)

    latest_json = base_output_dir / "summary_latest.json"
    latest_csv = base_output_dir / "summary_latest.csv"
    with latest_json.open("w") as f:
        json.dump(summary_rows, f, indent=2)
    _write_csv(summary_rows, latest_csv)

    print(f"\nWrote summary JSON: {summary_json}")
    print(f"Wrote summary CSV:  {summary_csv}")
    print(f"Updated latest JSON: {latest_json}")
    print(f"Updated latest CSV:  {latest_csv}")

    ok_count = sum(1 for r in summary_rows if r.get("status") == "ok")
    total_count = len(summary_rows)
    print(f"Completed: {ok_count}/{total_count} successful runs")

    return 0 if ok_count > 0 else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch test ProstNFound models on UA (Optimum) with test_rl.py"
    )
    parser.add_argument(
        "--config",
        default="cfg/test_ua_models.yaml",
        help="Path to evaluation config yaml.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent / config_path

    raise SystemExit(run(config_path))


if __name__ == "__main__":
    main()
