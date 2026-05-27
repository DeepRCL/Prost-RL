#!/bin/bash
#SBATCH --job-name=pnf_sup_cv
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=34:00:00
#SBATCH --partition=mig
#SBATCH --qos=m
#SBATCH --output=/home/mahdi.abootorabi/prostnfound/prostnfound/logs/supervised_cv_fold_%x_%j.log
#SBATCH --error=/home/mahdi.abootorabi/prostnfound/prostnfound/logs/supervised_cv_fold_%x_%j.err

set -euo pipefail

# Do not infer repo root from BASH_SOURCE in Slurm jobs, because Slurm may run
# a copied script from /var/lib/slurm/... . Use the real project path directly.
REPO_ROOT="/home/mahdi.abootorabi/prostnfound/prostnfound"
cd "${REPO_ROOT}"

# Usage:
#   sbatch scripts/sbatch_supervised_baseline_cv.sh 0
#   sbatch scripts/sbatch_supervised_baseline_cv.sh 1
#   ...
if [ "$#" -ne 1 ]; then
  echo "Usage: sbatch scripts/sbatch_supervised_baseline_cv.sh <fold_number>"
  echo "Example: sbatch scripts/sbatch_supervised_baseline_cv.sh 3"
  exit 1
fi

FOLD="$1"

case "${FOLD}" in
  0) CONFIG="cfg/train/experiments/ppo/supervised_baseline.yaml" ;;
  1) CONFIG="cfg/train/experiments/ppo/cross_fold/supervised_baseline_fold1.yaml" ;;
  2) CONFIG="cfg/train/experiments/ppo/cross_fold/supervised_baseline_fold2.yaml" ;;
  3) CONFIG="cfg/train/experiments/ppo/cross_fold/supervised_baseline_fold3.yaml" ;;
  4) CONFIG="cfg/train/experiments/ppo/cross_fold/supervised_baseline_fold4.yaml" ;;
  *)
    echo "Unsupported fold: ${FOLD}. Expected 0..4."
    exit 1
    ;;
esac

# Required conda activation.
source /home/mahdi.abootorabi/miniconda3/etc/profile.d/conda.sh
conda activate prostnfound

export PYTHONPATH="${PYTHONPATH:-}:${REPO_ROOT}"
export EXACTVU_PCA_DATA_ROOT=/data/project/users/mahdi
export EXACTVU_PCA_DATA_ROOT=/data/project/prostate-us
export MEDSAM_CHECKPOINT_DIR=/home/mahdi.abootorabi/prostnfound/checkpoints
export CHECKPOINT_DIR=/home/mahdi.abootorabi/prostnfound/checkpoints

echo "=============================================="
echo "Starting supervised CV training at $(date)"
echo "SLURM_JOB_ID        : ${SLURM_JOB_ID:-N/A}"
echo "SLURM_JOB_NAME      : ${SLURM_JOB_NAME:-N/A}"
echo "Running fold        : ${FOLD}"
echo "Config              : ${CONFIG}"
echo "Working dir         : ${PWD}"
echo "=============================================="

srun python train_rl.py --config "${CONFIG}"

echo "Finished supervised CV training at $(date)"
