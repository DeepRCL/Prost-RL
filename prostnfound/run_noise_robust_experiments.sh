#!/bin/bash
# =============================================================================
# NOISE-ROBUST LOSS FUNCTION EXPERIMENTS
# =============================================================================
# 
# This script runs experiments with different noise-robust loss functions
# to find the best calibration between activation and true involvement.
#
# The problem: Current model predicts "uniform uncertainty" (~0.55 everywhere)
# instead of identifying specific pixels that are cancerous.
#
# Solutions being tested:
# 1. Top-K MIL: Only supervise top k% pixels as cancer (k = involvement)
# 2. Thresholded Involvement: Compare fraction of pixels > 0.5 to involvement  
# 3. Symmetric Cross Entropy: Noise-robust loss from Manifold DivideMix
#
# Usage:
#   ./run_noise_robust_experiments.sh [optional: specific experiment name]
#
# Examples:
#   ./run_noise_robust_experiments.sh                    # Run all experiments
#   ./run_noise_robust_experiments.sh topk_mil           # Run only topk_mil
#   ./run_noise_robust_experiments.sh symmetric_ce       # Run only symmetric_ce
#
# =============================================================================

set -e

# Directory containing experiment configs
EXPERIMENT_DIR="cfg/train/experiments"

# Available experiments
declare -A EXPERIMENTS=(
    ["topk_mil"]="noise_robust_topk_mil.yaml"
    ["topk_mil_soft"]="noise_robust_topk_mil_soft.yaml"
    ["thresholded"]="noise_robust_thresholded.yaml"
    ["symmetric_ce"]="noise_robust_symmetric_ce.yaml"
    ["symmetric_ce_strong"]="noise_robust_symmetric_ce_strong.yaml"
)

# Function to run a single experiment
run_experiment() {
    local name=$1
    local config=$2
    
    echo "=============================================="
    echo "Running experiment: $name"
    echo "Config: $EXPERIMENT_DIR/$config"
    echo "=============================================="
    
    python train_rl.py --config "$EXPERIMENT_DIR/$config"
    
    echo ""
    echo "Completed: $name"
    echo ""
}

# Parse command line arguments
if [ $# -eq 0 ]; then
    # Run all experiments
    echo "Running ALL noise-robust loss experiments..."
    echo ""
    
    for name in "${!EXPERIMENTS[@]}"; do
        config="${EXPERIMENTS[$name]}"
        run_experiment "$name" "$config"
    done
    
    echo "=============================================="
    echo "ALL EXPERIMENTS COMPLETED"
    echo "=============================================="
    echo ""
    echo "Next steps:"
    echo "1. Run test_rl.py on each model checkpoint"
    echo "2. Generate comparison plots to evaluate calibration"
    echo "3. Compare 'Activation vs True Involvement' plots"
    echo ""
    
else
    # Run specific experiment
    exp_name=$1
    
    if [ "${EXPERIMENTS[$exp_name]+isset}" ]; then
        config="${EXPERIMENTS[$exp_name]}"
        run_experiment "$exp_name" "$config"
    else
        echo "Unknown experiment: $exp_name"
        echo ""
        echo "Available experiments:"
        for name in "${!EXPERIMENTS[@]}"; do
            echo "  - $name"
        done
        exit 1
    fi
fi
