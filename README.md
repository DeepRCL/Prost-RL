# Prost-RL

Official implementation of **Prost-RL**: a reinforcement-learning framework for robust micro-ultrasound prostate cancer detection, built on the [ProstNFound+](https://github.com/DeepRCL/ProstNFound) backbone (MedSAM encoder–decoder with clinical prompts).

**Paper:** *Learning Where to Look: A Reinforcement Learning Framework for Robust Micro-Ultrasound Prostate Cancer Detection*

## Overview

Prost-RL adds three components on top of ProstNFound+:

1. **Spatial attention policy** — learns where to attend before decoding.
2. **Noise-robust supervision** — symmetric cross-entropy plus pixel entropy regularization for weak core-level labels.
3. **Adaptive Policy Optimization (APO)** — DRPO fine-tuning with a pairwise ranking reward after supervised warm-up.

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
| RL algorithm | DRPO |
| Reward | `pairwise_ranking` (csPCa bonus γ=2) |
| Rollouts K | 4 |
| Attention noise σ | 0.15 |
| RL loss weight | 0.8 |
| Optimizer | AdamW, lr=2e-5, encoder lr=1e-5, wd=1e-3 |
| Model selection | `val/core_auc_high_involvement` |

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

## Citation

If you use this code, please cite our paper (MICCAI 2025 submission) and the ProstNFound+ baseline:

```bibtex
@inproceedings{abootorabi2025prostrl,
  title={Learning Where to Look: A Reinforcement Learning Framework for Robust Micro-Ultrasound Prostate Cancer Detection},
  author={Abootorabi, Mohammad Mahdi and Namazi, Sina and Saadat, Armin and Wang, Lyuyang and Dzikunu, Obed and Wilson, Paul F. R. and Guo, Zhuoxin and Wodlinger, Brian and Mousavi, Parvin and Abolmaesumi, Purang},
  year={2025}
}
```

## License

Research code released for reproducibility. Dataset access is subject to the NCT2013 trial data use agreement; contact the authors for data sharing questions.

## Acknowledgments

Built on [ProstNFound+](https://github.com/DeepRCL/ProstNFound), [MedSAM](https://github.com/bowang-lab/MedSAM), and the `medAI` training stack used in the ProstNFound line of work.
