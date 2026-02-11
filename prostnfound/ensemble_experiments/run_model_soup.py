#!/usr/bin/env python
"""
Quick script to run model soup experiments.

Usage:
    # Create uniform soup (all models, equal weights)
    python run_model_soup.py --mode uniform --output_dir outputs/soup_uniform

    # Create uniform soup with specific models
    python run_model_soup.py --mode uniform \
        --models v3_ppo_symmetric v4_gdpo_normalmil supervised \
        --output_dir outputs/soup_3models

    # Dry run (just load and create soup, don't evaluate)
    python run_model_soup.py --mode uniform --dry_run
"""

import argparse
import os
import sys

# Set CHECKPOINT_DIR before importing medAI modules
if 'CHECKPOINT_DIR' not in os.environ:
    # Default to common location
    default_checkpoint_dir = os.path.expanduser("~/prostnfound/checkpoints")
    if os.path.exists(default_checkpoint_dir):
        os.environ['CHECKPOINT_DIR'] = default_checkpoint_dir

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
from tqdm import tqdm

from ensemble_experiments.models.model_soup import (
    create_soup_from_config,
    create_uniform_soup,
    ModelSoup,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run model soup experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--mode", 
        type=str, 
        default="uniform",
        choices=["uniform", "custom"],
        help="Soup mode: uniform (equal weights) or custom (specify weights)"
    )
    
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model keys to include (from configs/models.yaml). Default: all models"
    )
    
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="Custom weights for each model (must sum to 1)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/models.yaml",
        help="Path to models config file"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ensemble_experiments/outputs/model_soup",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Data split to evaluate on"
    )
    
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Just create the soup, don't evaluate"
    )
    
    parser.add_argument(
        "--save_soup",
        action="store_true",
        help="Save the souped model checkpoint"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Create output directory
    output_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load config to get checkpoint paths
    config_path = os.path.join(project_root, "ensemble_experiments", args.config)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Determine which models to use
    if args.models is None:
        # Use all models
        model_keys = list(config['models'].keys())
    else:
        model_keys = args.models
    
    print(f"\n{'='*60}")
    print(f"Creating model soup with {len(model_keys)} models:")
    for key in model_keys:
        print(f"  - {key}: {config['models'][key]['name']}")
    print(f"{'='*60}\n")
    
    # Get checkpoint paths
    checkpoint_paths = [config['models'][key]['checkpoint'] for key in model_keys]
    
    # Create soup
    if args.mode == "uniform":
        soup = create_uniform_soup(checkpoint_paths, device=device)
    else:
        if args.weights is None:
            raise ValueError("Custom mode requires --weights")
        soup = ModelSoup.from_checkpoints(checkpoint_paths, weights=args.weights, device=device)
    
    print(f"\nModel soup created successfully!")
    
    # Save soup if requested
    if args.save_soup:
        soup_path = os.path.join(output_dir, "model_soup.pth")
        torch.save({
            'model': soup.model.state_dict(),
            'model_keys': model_keys,
            'mode': args.mode,
            'weights': args.weights,
        }, soup_path)
        print(f"Saved soup to: {soup_path}")
    
    if args.dry_run:
        print("\nDry run complete. Skipping evaluation.")
        return
    
    # Evaluate on specified split
    print(f"\nEvaluating on {args.split} split...")
    
    # Import evaluation utilities
    from src.loaders import get_dataloaders
    from src.evaluator import CancerLogitsHeatmapsEvaluator as Evaluator
    
    # Load data
    # Use one of the base model configs to get data settings
    first_checkpoint = os.path.join(project_root, checkpoint_paths[0])
    state = torch.load(first_checkpoint, map_location='cpu', weights_only=False)
    from argparse import Namespace
    train_args = Namespace(**state["args"])
    
    # Convert data config (dict) to Namespace for compatibility with get_dataloaders
    def dict_to_namespace(d):
        """Recursively convert dict to Namespace."""
        if isinstance(d, dict):
            for k, v in d.items():
                d[k] = dict_to_namespace(v)
            return Namespace(**d)
        return d
    
    if hasattr(train_args, 'data') and isinstance(train_args.data, dict):
        data_args = dict_to_namespace(train_args.data)
        loaders = get_dataloaders(data_args, mode="test")
    elif hasattr(train_args, 'data'):
        loaders = get_dataloaders(train_args.data, mode="test")
    else:
        loaders = get_dataloaders(train_args, mode="test")
    
    loader = loaders[args.split]
    
    # Evaluate
    evaluator = Evaluator(log_images=False)
    
    for data in tqdm(loader, desc=f"Evaluating on {args.split}"):
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=True):
                out = soup(data, deterministic=True)
        
        # Merge output into data for evaluator
        data.update(out)
        evaluator(data)
    
    # Get metrics
    metrics = evaluator.aggregate_metrics()
    metrics = {k: float(v) for k, v in metrics.items()}
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Model Soup Results ({args.split} split)")
    print(f"{'='*60}")
    for key, value in sorted(metrics.items()):
        print(f"  {key}: {value:.4f}")
    
    # Save metrics
    import json
    metrics_path = os.path.join(output_dir, f"metrics_{args.split}.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to: {metrics_path}")
    
    # Save per-core metrics
    table = evaluator.accumulator.compute()
    table_path = os.path.join(output_dir, f"metrics_by_core_{args.split}.csv")
    table.to_csv(table_path)
    print(f"Saved per-core metrics to: {table_path}")


if __name__ == "__main__":
    main()
