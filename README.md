# Prost-RL

[![arXiv](https://img.shields.io/badge/arXiv-2606.05531-b31b1b.svg)](https://arxiv.org/abs/2606.30951) 

Official implementation of **Prost-RL**: a reinforcement-learning framework for robust micro-ultrasound prostate cancer detection, built on the ProstNFound+ backbone (MedSAM encoder–decoder with clinical prompts).

> 🏆 **Accepted at MICCAI 2026 (Early Accept — top 9%)**

**Paper:** *Learning Where to Look: A Reinforcement Learning Framework for Robust Micro-Ultrasound Prostate Cancer Detection*

**Authors:** Mohammad Mahdi Abootorabi, Sina Namazi, Armin Saadat, Lyuyang Wang, Obed Dzikunu, Paul F. R. Wilson, Zhuoxin Guo, Brian Wodlinger, Parvin Mousavi, Purang Abolmaesumi

*The University of British Columbia · Queen's University · Vector Institute · Exact Imaging*

## Overview

Micro-ultrasound (µUS) is an emerging modality for prostate cancer (PCa) detection that operates at frequencies up to 29 MHz to resolve prostate micro-architecture with MRI-comparable accuracy. However, interpretation is highly experience-dependent, and supervision for training deep models is **sparse, noisy, and severely imbalanced** — typically limited to core-level histopathology (cancer grade and involvement percentage) without pixel-level lesion annotations.

**Prost-RL** reframes µUS PCa detection as a spatially aware, policy-driven inference problem by learning **where to look before decoding**. It integrates a lightweight reinforcement-learning policy into a foundation-model encoder–decoder to produce interpretable spatial attention maps that act as soft prompts for both cancer-likelihood heatmap prediction and image-level classification.


<img width="5545" height="2365" alt="pipeline (1) (1)" src="https://github.com/user-attachments/assets/955b3998-a97f-4034-9125-b772fb53ac8c" />



### Key contributions

1. **Spatial attention policy** — a lightweight policy network πθ that generates an attention map α to modulate encoder features before they reach the heatmap decoder and csPCa classifier.
2. **Noise-robust weakly supervised objective** — Symmetric Cross-Entropy (SCE) combined with pixel-level entropy regularization, mitigating noisy proportion labels and enforcing spatially sharp heatmaps.
3. **Adaptive Policy Optimization (APO)** — a DRPO-based RL fine-tuning stage with a pairwise ranking reward (csPCa bonus γ=2), applied after supervised warm-up. Gaussian noise is injected into the attention logits to enable exploration over otherwise deterministic continuous attention.

### Headline results

On a multi-center retrospective cohort of **6,607 biopsy cores from 693 patients across five clinical sites** (patient-level five-fold cross-validation, center-stratified):

| Task | Metric | ProstNFound+ | **Prost-RL (Ours)** |
|------|--------|--------------|----------------------|
| Core-level detection (all cores) | AUROC | 76.9 ± 3.5 | **79.0 ± 3.5** *** |
| Core-level detection (all cores) | Sens@80%Spec | 60.1 ± 5.6 | **64.6 ± 6.3** * |
| Core-level detection (high involvement) | AUROC | 83.6 ± 2.4 | **84.9 ± 2.5** |
| csPCa classification head | AUROC | 78.5 ± 5.3 | **79.3 ± 5.8** |
| csPCa classification head | Sens@80%Spec | 58.2 ± 10.6 | **62.8 ± 12.6** |

\* p<0.05, \*\*\* p<0.001 (two-sided paired t-test over five folds).

## Repository layout

```
medAI/              # Models, datasets, MedSAM adapters, DRPO
external_libs/      # Additional dependencies (editable install)
prostnfound/        # Training and evaluation scripts (Hydra configs)
environment.yml     # Conda environment
requirements.txt    # Pip dependencies
```

## Setup

### Environment

From the repository root:

```bash
conda env create -f environment.yml
conda activate prostnfound
```

Or install editable packages manually:

```bash
pip install -r requirements.txt
pip install -e ./medAI -e ./external_libs
```

### Data and checkpoints

The NCT2013 micro-ultrasound cohort is accessed via `EXACTVU_PCA_DATA_ROOT` (must contain an `nct2013/` subdirectory with images, masks, and metadata). See [ClinicalTrials.gov NCT02079025](https://clinicaltrials.gov/study/NCT02079025) for the prospective trial this cohort derives from.

The dataset contains B-mode sagittal-plane µUS images (depth 28 mm, width 46.06 mm) acquired with the ExactVu system, with core-level ISUP Grade Group and involvement labels. Images are resized to 256×256 and masks to 64×64 (matching attention-map resolution).

Download [MedSAM](https://github.com/bowang-lab/MedSAM) weights and set:

```bash
export EXACTVU_PCA_DATA_ROOT=/path/to/exactvu_pca_data
export MEDSAM_CHECKPOINT_DIR=/path/to/medsam_checkpoints
export CHECKPOINT_DIR=/path/to/checkpoints   # optional; used by some medAI utilities
```

## Training (EXP4 — best model)

Training is two-stage per fold: **supervised warm-up**, then **DRPO / pairwise-ranking RL** initialized from that checkpoint.

Run from `prostnfound/`:

```bash
cd prostnfound
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
```

### Stage 1 — supervised warm-up (35 epochs, batch size 8)

Optimizes the noise-robust supervised objective `L_sup = L_SCE + L_ent + L_clf` (symmetric cross-entropy on the heatmap proportions, pixel-entropy regularizer over the prostate–needle intersection, and balanced CE on the classification head).

| Fold | Config |
|------|--------|
| 0 | `cfg/train/experiments/ppo/supervised_baseline.yaml` |
| 1–4 | `cfg/train/experiments/ppo/cross_fold/supervised_baseline_fold{N}.yaml` |

```bash
# Example: fold 0
python train_rl.py --config cfg/train/experiments/ppo/supervised_baseline.yaml

# Example: fold 2
python train_rl.py --config cfg/train/experiments/ppo/cross_fold/supervised_baseline_fold2.yaml
```

Checkpoints are written under `checkpoints_supervised_cv/PPO-supervised-baseline-fold{N}/best_rl.pth`.

### Stage 2 — Prost-RL / EXP4 (35 epochs, batch size 16, DRPO)

Jointly optimizes the DRPO policy loss alongside the supervised objectives. K=4 stochastic rollouts per image (Gaussian noise σ=0.15 injected into attention logits) drive pairwise-ranking-reward exploration, with hierarchical (cancer-vs-benign) advantage scaling to upweight rare positives and hard borderline lesions.

| Fold | Config |
|------|--------|
| 0 | `cfg/train/experiments/v3/exp4_pairwise_ranking_rl.yaml` |
| 1–4 | `cfg/train/experiments/v3_cross_fold/exp4_pairwise_ranking_rl_fold{N}.yaml` |

```bash
# Example: fold 0 (loads ../checkpoints_supervised_cv/PPO-supervised-baseline-fold0/best_rl.pth)
python train_rl.py --config cfg/train/experiments/v3/exp4_pairwise_ranking_rl.yaml

# Example: fold 3
python train_rl.py --config cfg/train/experiments/v3_cross_fold/exp4_pairwise_ranking_rl_fold3.yaml
```

Outputs are saved to `checkpoints_supervised_cv/EXP4-pairwise-ranking-rl-fold{N}/` (fold 0 run name may include a `-v2` suffix in the config).

### Key hyperparameters (EXP4)

| Setting | Value |
|---------|-------|
| Loss | `symmetric_ce_entropy_reg` (α=β=1, ε=1e-4) |
| RL algorithm | DRPO (Domain-aware Group Relative Policy Optimization) |
| Reward | `pairwise_ranking` (csPCa bonus γ=2) |
| Rollouts K | 4 |
| Attention noise σ | 0.15 |
| RL loss weight | 0.8 |
| Optimizer | AdamW, lr=2e-5, encoder lr=1e-5, wd=1e-3, cosine annealing |
| Model selection | `val/core_auc_high_involvement` (≥40% involvement) |

## Evaluation

```bash
cd prostnfound

python test_rl.py \
  checkpoint=/path/to/checkpoints_supervised_cv/EXP4-pairwise-ranking-rl-fold0/best_rl.pth \
  output_dir=outputs/exp4_fold0 \
  data.fold=0 \
  split=val
```

Repeat for folds 1–4 with the matching checkpoint and `data.fold`. Optional: set `PNF_RL_CHECKPOINT` instead of the `checkpoint=` override.

Core-level scores are computed as the mean heatmap activation within the needle–prostate intersection; image-level csPCa scores come from the classification head. We report AUROC and sensitivity at fixed specificities (80% by default; 60% also reported for the classification head).

## Inference on a single image

We release the fold-0 EXP4 checkpoint (best on the tracked cross-validation metric, `val/core_auc_high_involvement`) for standalone use outside the training/eval pipeline: **[download link — add your hosted checkpoint URL here]**.

`prostnfound/inference.py` loads that checkpoint and runs it on one B-mode micro-ultrasound image, with an optional prostate mask and clinical metadata:

```bash
cd prostnfound
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
export MEDSAM_CHECKPOINT_DIR=/path/to/checkpoints   # dir with medsam_vit_b_cpu.pth (see Setup above)

python inference.py \
  --checkpoint /path/to/EXP4-pairwise-ranking-rl-fold0-best_rl.pth \
  --image /path/to/bmode.png \
  --prostate-mask /path/to/prostate_mask.png \
  --age 65 --psa 6.5 --psa-density 0.00015 --loc LBM \
  --output heatmap.png
```

This prints a prostate-level cancer-likelihood score and an image-level csPCa probability, and saves a heatmap-over-bmode overlay to `--output`.

Notes:
- `--prostate-mask`, `--age`, `--psa`, `--psa-density`, and `--loc` are all optional. Omitted clinical values fall back to training-set averages; an omitted prostate mask falls back to treating the whole image as prostate tissue. **Supplying a real prostate segmentation mask is strongly recommended** — the RL attention policy is trained to look only inside the prostate, so without a mask it attends over the full field of view and heatmap quality degrades.
- `--loc` is the biopsy/core location code (e.g. `LBM` = Left-Base-Medial, `RAL` = Right-Apex-Lateral); it encodes where in the gland the region of interest sits. Leave it out if unknown.
- `MEDSAM_CHECKPOINT_DIR` is required even for inference-only use: it's used to build the model architecture skeleton before the fine-tuned EXP4 weights are loaded on top of it.
- This is one fold of a 5-fold cross-validation study, not a single globally-trained model — see [Evaluation](#evaluation) for the other folds' checkpoints and per-fold metrics.

## Method at a glance

Given an input image `x`, the MedSAM encoder produces spatial features `F = Enc(x) ∈ R^{C×H×W}`. The attention policy πθ processes `F` together with a clinical-metadata embedding `c` (age, PSA, PSAD, POS), gates `F` channel-wise, and produces spatial attention logits that are masked outside the prostate region and normalized to obtain `α`. Features are modulated via residual injection:

```
F̃ = F_proc ⊙ α
E  = F + φ(F̃)
```

where `φ(·)` is a bias-free 1×1 projection with GELU. The shared modulated embeddings `E` feed both the heatmap decoder and the csPCa classifier; `α` is itself an interpretable spatial map of where the model attended.

For weak proportional labels `q ∈ [0,1]` over the needle–prostate intersection, SCE replaces vanilla BCE to bound the gradient under model–label disagreement, and the pixel-entropy regularizer prevents the trivial "predict `q` everywhere" degeneracy by encouraging sharp, decisive heatmaps. APO then refines the policy with DRPO + pairwise ranking, where Gaussian noise on attention logits breaks the determinism of continuous spatial attention to enable rollout-based exploration.

## Citation

If you use this code, please cite our paper and the ProstNFound+ baseline:

```bibtex
coming soon...
```

## License

Research code released for reproducibility. Dataset access is subject to the NCT2013 trial data use agreement; contact the authors for data sharing questions.

## Acknowledgments

This work was supported in part by the Canadian Institutes of Health Research (CIHR), the Natural Sciences and Engineering Research Council of Canada (NSERC), the Vector Institute, and through computational resources and services provided by Advanced Research Computing at the University of British Columbia. P. Mousavi is supported in part by a Canada CIFAR AI Chair and a Canada Research Chair.
