# Prost-RL training code

Hydra configs and scripts for Prost-RL live here. See the [repository README](../README.md) for setup, EXP4 training commands, and evaluation.

**Entry points:** `train_rl.py`, `test_rl.py`

**Best model configs:**
- Supervised warm-up: `cfg/train/experiments/ppo/supervised_baseline.yaml` (and `cross_fold/` for folds 1–4)
- Prost-RL (EXP4): `cfg/train/exp4_prost_rl.yaml` or `cfg/train/experiments/v3/exp4_pairwise_ranking_rl.yaml`
