"""Standalone single-image inference for the Prost-RL EXP4 checkpoint.

Given one B-mode micro-ultrasound image (optionally with a prostate
segmentation mask and clinical metadata), this loads a released
`best_rl.pth` checkpoint end-to-end and produces:
  - a per-pixel cancer-likelihood heatmap
  - a prostate-level cancer-likelihood score
  - an image-level csPCa probability (classification head)
  - a heatmap-over-bmode overlay PNG

Setup (from the repository root):
    pip install -r requirements.txt
    pip install -e ./medAI -e ./external_libs
    export MEDSAM_CHECKPOINT_DIR=/path/to/checkpoints   # dir with medsam_vit_b_cpu.pth

Usage:
    cd prostnfound
    export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
    python inference.py \
        --checkpoint /path/to/EXP4-pairwise-ranking-rl-fold0/best_rl.pth \
        --image /path/to/bmode.png \
        --prostate-mask /path/to/prostate_mask.png \
        --age 65 --psa 6.5 --psa-density 0.00015 --loc LBM \
        --output heatmap.png

`--prostate-mask`, `--age`, `--psa`, `--psa-density`, and `--loc` are all
optional: omitted clinical values fall back to training-set averages, and an
omitted prostate mask falls back to treating the whole image as prostate
tissue (this will make the RL attention policy and the cancer heatmap look at
the full field of view instead of just the gland, so supplying a real
prostate mask is strongly recommended for meaningful results).
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

IMAGE_SIZE = 256
MASK_SIZE = 64

# Training-set min/max/avg used to normalize clinical prompts, copied from
# prostnfound/src/transform.py. Must match what EXP4 was trained on.
PSA_MIN, PSA_MAX, PSA_AVG = 0.2, 32.95, 6.821426488456866
AGE_MIN, AGE_MAX, AGE_AVG = 0, 79, 62.5816
PSAD_MIN, PSAD_MAX, PSAD_AVG = (
    4.615739672282483e-06,
    0.000837278201784,
    0.000175347951594383,
)


def encode_core_location(loc):
    """Core location code (e.g. 'LBM', 'RAL') -> (base_apex, mid_lateral) in [-1, 1].

    Index 1 of the code is base(B)/mid(M)/apex(A), index 2 is mid(M)/lateral(L).
    See medAI/medAI/datasets/nct2013/data_access.py::get_metadata_table.
    """
    if not loc or len(loc) < 3:
        return 0.0, 0.0  # unspecified -> neutral
    code = loc.upper()
    base_apex = {"B": -1.0, "M": 0.0, "A": 1.0}.get(code[1], 0.0)
    mid_lateral = 1.0 if code[2] == "M" else -1.0
    return base_apex, mid_lateral


class InferenceWrapper(torch.nn.Module):
    """Deterministic eval-time wrapper, trimmed from `ProstNFoundMeta` in
    prostnfound/train_rl.py to just what's needed for a single forward pass."""

    def __init__(self, model, boundary_tolerance_patches=1):
        super().__init__()
        self.model = model
        self.boundary_tolerance_patches = boundary_tolerance_patches
        self.register_buffer("temperature", torch.tensor([1.0]))
        self.register_buffer("bias", torch.tensor([0.0]))

    @torch.inference_mode()
    def forward(self, bmode, prostate_mask, prompts):
        outputs = self.model(
            bmode,
            None,
            prostate_mask,
            None,
            output_mode="all",
            deterministic=True,
            return_rl_info=True,
            **prompts,
        )
        cancer_logits = outputs["mask_logits"]
        cancer_logits = (
            cancer_logits / self.temperature[None, None, None, :]
            + self.bias[None, None, None, :]
        )

        # Hard-mask the decoder output outside the prostate, matching the
        # published evaluation protocol (test_rl.py, apply_prostate_mask_to_decoder=True).
        H, W = cancer_logits.shape[-2:]
        mask = F.interpolate(prostate_mask.float(), size=(H, W), mode="nearest")
        if self.boundary_tolerance_patches > 0:
            scale = H // 16
            tol_px = self.boundary_tolerance_patches * scale
            kernel = torch.ones(1, 1, 2 * tol_px + 1, 2 * tol_px + 1, device=mask.device)
            mask = (F.conv2d(mask, kernel, padding=tol_px) > 0).float()
        cancer_logits = torch.where(
            mask < 0.5, torch.full_like(cancer_logits, -100.0), cancer_logits
        )
        return cancer_logits, outputs.get("cls_outputs"), outputs.get("rl_attention_map")


def load_model(checkpoint_path, device):
    if not os.environ.get("MEDSAM_CHECKPOINT_DIR"):
        raise RuntimeError(
            "MEDSAM_CHECKPOINT_DIR is not set. It must point to a directory "
            "containing medsam_vit_b_cpu.pth (used to build the model skeleton "
            "before the fine-tuned EXP4 weights are loaded on top of it)."
        )
    # medAI.modeling registers all model architectures (incl. the RL ones) as
    # an import side effect, so this import must happen after the env check.
    from medAI.modeling import create_model

    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    train_args = state["args"]
    model_kw = dict(train_args["model_kw"])
    base_model = create_model(train_args["model"], **model_kw)
    wrapper = InferenceWrapper(
        base_model, boundary_tolerance_patches=model_kw.get("boundary_tolerance_patches", 1)
    )
    missing, unexpected = wrapper.load_state_dict(state["model"], strict=False)
    if missing or unexpected:
        print(f"[load_model] missing keys: {missing}")
        print(f"[load_model] unexpected keys: {unexpected}")
    wrapper.to(device).eval()
    return wrapper, base_model.prompts


def load_image_as_tensor(path, size, device):
    """Grayscale -> 3-channel -> min-max normalize to [0, 1] -> resize.
    Mirrors prostnfound/src/transform.py::ProstNFoundTransform (mean=[0,0,0],
    std=[1,1,1], i.e. no additional normalization beyond min-max scaling)."""
    img = Image.open(path).convert("L")
    arr = np.array(img).astype(np.float32)
    t = torch.from_numpy(arr)[None, None].repeat(1, 3, 1, 1)
    t = (t - t.min()) / (t.max() - t.min() + 1e-8)
    t = F.interpolate(t, size=(1024, 1024), mode="bilinear", align_corners=False, antialias=True)
    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return t.to(device)


def load_mask_as_tensor(path, size, device):
    if path is None:
        return torch.ones(1, 1, size, size, device=device)
    m = Image.open(path).convert("L")
    arr = np.array(m)
    # Mask PNGs in this dataset are encoded as 0/1, not 0/255 — threshold at
    # the midpoint of the file's own value range so both encodings work.
    threshold = arr.max() / 2 if arr.max() > 0 else 0
    arr = (arr > threshold).astype(np.float32)
    t = torch.from_numpy(arr)[None, None]
    t = F.interpolate(t, size=(size, size), mode="nearest")
    return t.to(device)


def build_prompts(prompt_names, age, psa, psa_density, loc, device):
    base_apex, mid_lateral = encode_core_location(loc)
    norm_age = (age - AGE_MIN) / (AGE_MAX - AGE_MIN) if age is not None else (
        AGE_AVG - AGE_MIN
    ) / (AGE_MAX - AGE_MIN)
    norm_psa = (psa - PSA_MIN) / (PSA_MAX - PSA_MIN) if psa is not None else (
        PSA_AVG - PSA_MIN
    ) / (PSA_MAX - PSA_MIN)
    norm_psad = (psa_density - PSAD_MIN) / (PSAD_MAX - PSAD_MIN) if psa_density is not None else (
        PSAD_AVG - PSAD_MIN
    ) / (PSAD_MAX - PSAD_MIN)

    values = {
        "age": norm_age,
        "psa": norm_psa,
        "approx_psa_density": norm_psad,
        "base_apex_encoding": base_apex,
        "mid_lateral_encoding": mid_lateral,
    }
    return {
        name: torch.tensor([[values[name]]], dtype=torch.float32, device=device)
        for name in prompt_names
    }


def render_overlay(bmode, heatmap, out_path):
    import matplotlib.pyplot as plt

    img = bmode[0, 0].detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img, cmap="gray")
    ax.imshow(heatmap, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    ax.axis("off")
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True, help="Path to a best_rl.pth checkpoint")
    parser.add_argument("--image", required=True, help="Path to a B-mode micro-ultrasound image")
    parser.add_argument(
        "--prostate-mask", default=None, help="Optional binary prostate segmentation mask image"
    )
    parser.add_argument("--age", type=float, default=None, help="Patient age in years")
    parser.add_argument("--psa", type=float, default=None, help="Serum PSA in ng/mL")
    parser.add_argument(
        "--psa-density", type=float, default=None, help="PSA density (PSA / prostate volume)"
    )
    parser.add_argument(
        "--loc",
        default=None,
        help="Core/biopsy location code, e.g. 'LBM' (Left-Base-Medial), 'RAL' (Right-Apex-Lateral)",
    )
    parser.add_argument("--output", default="heatmap.png", help="Where to save the overlay PNG")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, prompt_names = load_model(args.checkpoint, device)

    bmode = load_image_as_tensor(args.image, IMAGE_SIZE, device)
    prostate_mask_hi = load_mask_as_tensor(args.prostate_mask, IMAGE_SIZE, device)
    prostate_mask = F.interpolate(prostate_mask_hi, size=(MASK_SIZE, MASK_SIZE), mode="nearest")
    prompts = build_prompts(prompt_names, args.age, args.psa, args.psa_density, args.loc, device)

    cancer_logits, cls_outputs, _ = model(bmode, prostate_mask, prompts)
    heatmap_lo = cancer_logits[0, 0].sigmoid().cpu().numpy()
    heatmap = np.array(
        Image.fromarray((heatmap_lo * 255).astype(np.uint8)).resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR
        )
    ) / 255.0

    roi = prostate_mask_hi[0, 0].cpu().numpy() > 0.5
    core_score = float(heatmap[roi].mean()) if roi.any() else float(heatmap.mean())
    print(f"Prostate-level cancer-likelihood score: {core_score:.4f}")

    if cls_outputs is not None:
        cspca_prob = torch.softmax(cls_outputs[0][0], dim=-1)[1].item()
        print(f"Image-level csPCa probability (classification head): {cspca_prob:.4f}")

    render_overlay(bmode, heatmap, args.output)
    print(f"Saved heatmap overlay to {args.output}")


if __name__ == "__main__":
    main()
