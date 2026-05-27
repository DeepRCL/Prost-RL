import argparse
import json
from pathlib import Path
from collections import OrderedDict

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a model soup by averaging fold checkpoints."
    )
    parser.add_argument(
        "--checkpoints",
        nargs="*",
        default=None,
        help="Explicit checkpoint paths (.pth). If omitted, auto-discover under --root.",
    )
    parser.add_argument(
        "--root",
        default="/home/mahdi.abootorabi/prostnfound/checkpoints_supervised_cv",
        help="Root directory for auto-discovery of fold checkpoints.",
    )
    parser.add_argument(
        "--preferred-name",
        default="best_rl.pth",
        choices=["best_rl.pth", "experiment_state_rl.pth"],
        help="Preferred checkpoint filename when auto-discovering.",
    )
    parser.add_argument(
        "--weights",
        nargs="*",
        type=float,
        default=None,
        help="Optional weights for checkpoints. Defaults to uniform.",
    )
    parser.add_argument(
        "--output",
        default="/home/mahdi.abootorabi/prostnfound/checkpoints_supervised_cv/PPO-supervised-baseline-foldsoup/best_rl_soup.pth",
        help="Output path for souped checkpoint.",
    )
    return parser.parse_args()


def discover_checkpoints(root: Path, preferred_name: str):
    discovered = []
    for d in sorted(root.glob("PPO-supervised-baseline-fold*")):
        if not d.is_dir():
            continue
        preferred = d / preferred_name
        fallback = d / ("experiment_state_rl.pth" if preferred_name == "best_rl.pth" else "best_rl.pth")
        if preferred.exists():
            discovered.append(preferred)
        elif fallback.exists():
            discovered.append(fallback)
    return discovered


def get_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict) and "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
        return ckpt_obj["model"]
    if isinstance(ckpt_obj, dict):
        return ckpt_obj
    raise ValueError("Unsupported checkpoint format: expected dict with model state dict.")


def normalize_weights(weights, n):
    if weights is None or len(weights) == 0:
        return [1.0 / n] * n
    if len(weights) != n:
        raise ValueError(f"Number of weights ({len(weights)}) must match checkpoints ({n}).")
    s = float(sum(weights))
    if s <= 0:
        raise ValueError("Weights sum must be > 0.")
    return [w / s for w in weights]


def average_state_dicts(state_dicts, weights):
    base = state_dicts[0]
    averaged = OrderedDict()
    skipped = []

    for key, tensor0 in base.items():
        # Keep missing or shape-mismatch keys from first checkpoint.
        compatible = True
        tensors = [tensor0]
        for sd in state_dicts[1:]:
            t = sd.get(key, None)
            if t is None or t.shape != tensor0.shape or t.dtype != tensor0.dtype:
                compatible = False
                break
            tensors.append(t)

        if not compatible:
            averaged[key] = tensor0
            skipped.append(key)
            continue

        # Average only floating-point tensors; keep integer buffers from first.
        if torch.is_floating_point(tensor0):
            avg = torch.zeros_like(tensor0, dtype=tensor0.dtype)
            for w, t in zip(weights, tensors):
                avg += w * t
            averaged[key] = avg
        else:
            averaged[key] = tensor0

    return averaged, skipped


def main():
    args = parse_args()

    if args.checkpoints:
        checkpoint_paths = [Path(p).expanduser().resolve() for p in args.checkpoints]
    else:
        checkpoint_paths = discover_checkpoints(Path(args.root).expanduser().resolve(), args.preferred_name)

    if len(checkpoint_paths) < 2:
        raise ValueError(
            f"Need at least 2 checkpoints for soup; found {len(checkpoint_paths)}.\n"
            f"Root searched: {args.root}"
        )

    weights = normalize_weights(args.weights, len(checkpoint_paths))

    ckpts = []
    state_dicts = []
    for p in checkpoint_paths:
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        obj = torch.load(p, map_location="cpu", weights_only=False)
        ckpts.append(obj)
        state_dicts.append(get_state_dict(obj))

    averaged_state, skipped = average_state_dicts(state_dicts, weights)

    # Keep first checkpoint metadata and replace model with souped weights.
    out_obj = ckpts[0].copy() if isinstance(ckpts[0], dict) else {"model": averaged_state}
    out_obj["model"] = averaged_state
    out_obj["soup_info"] = {
        "type": "uniform" if args.weights is None else "weighted",
        "weights": weights,
        "sources": [str(p) for p in checkpoint_paths],
        "num_checkpoints": len(checkpoint_paths),
        "num_skipped_keys": len(skipped),
        "skipped_keys_preview": skipped[:30],
    }

    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_obj, out_path)

    # Also save a small json manifest for tracking.
    manifest = out_path.with_suffix(".json")
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump(out_obj["soup_info"], f, indent=2)

    print(f"[OK] Soup checkpoint saved to: {out_path}")
    print(f"[OK] Soup manifest saved to:   {manifest}")
    print(f"[OK] Checkpoints used: {len(checkpoint_paths)}")
    print(f"[OK] Skipped keys (kept from first): {len(skipped)}")


if __name__ == "__main__":
    main()
