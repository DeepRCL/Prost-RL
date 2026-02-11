#!/usr/bin/env python
"""
Unified evaluation script for all ensemble methods.

Usage:
    # Test output ensemble (simple averaging)
    python test_ensemble.py --method output_average --split test

    # Test model soup
    python test_ensemble.py --method model_soup --split test

    # Test trained transformer aggregator
    python test_ensemble.py --method transformer \
        --checkpoint checkpoints/transformer_agg/best.pth \
        --split test

    # Test trained MoE
    python test_ensemble.py --method moe \
        --checkpoint checkpoints/moe/best.pth \
        --split test

    # Quick test with limited samples
    python test_ensemble.py --method output_average --limit 50
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

# Set CHECKPOINT_DIR before importing medAI modules
if 'CHECKPOINT_DIR' not in os.environ:
    default_checkpoint_dir = os.path.expanduser("~/prostnfound/checkpoints")
    if os.path.exists(default_checkpoint_dir):
        os.environ['CHECKPOINT_DIR'] = default_checkpoint_dir

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import yaml
from tqdm import tqdm

# Import ensemble methods
from ensemble_experiments.models.base_loader import load_models_from_config
from ensemble_experiments.models.output_ensemble import create_output_ensemble
from ensemble_experiments.models.model_soup import create_soup_from_config
from ensemble_experiments.models.transformer_aggregator import create_transformer_ensemble
from ensemble_experiments.models.moe_ensemble import create_moe_ensemble


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate ensemble methods",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["output_average", "model_soup", "transformer", "moe"],
        help="Ensemble method to evaluate"
    )
    
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model keys to include (default: all models from config)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/models.yaml",
        help="Path to models config file"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained ensemble checkpoint (for transformer/moe)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: outputs/<method>)"
    )
    
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Data split to evaluate on"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to evaluate (for quick testing)"
    )
    
    parser.add_argument(
        "--save_heatmaps",
        action="store_true",
        help="Save heatmap visualizations"
    )
    
    parser.add_argument(
        "--use_amp",
        action="store_true",
        default=True,
        help="Use automatic mixed precision"
    )
    
    return parser.parse_args()


def create_ensemble(args, device):
    """Create the appropriate ensemble model."""
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, "ensemble_experiments", args.config)
    
    print(f"\nCreating {args.method} ensemble...")
    
    if args.method == "output_average":
        ensemble = create_output_ensemble(
            config_path=config_path,
            model_keys=args.models,
            learn_weights=False,
            device=device,
        )
        
    elif args.method == "model_soup":
        ensemble = create_soup_from_config(
            config_path=config_path,
            model_keys=args.models,
            device=device,
        )
        
    elif args.method == "transformer":
        ensemble = create_transformer_ensemble(
            config_path=config_path,
            model_keys=args.models,
            use_attention_maps=True,
            device=device,
        )
        # Load trained weights if provided
        if args.checkpoint:
            checkpoint_path = os.path.join(project_root, args.checkpoint)
            state = torch.load(checkpoint_path, map_location=device)
            ensemble.aggregator.load_state_dict(state['aggregator'])
            print(f"Loaded aggregator weights from: {args.checkpoint}")
        
    elif args.method == "moe":
        ensemble = create_moe_ensemble(
            config_path=config_path,
            model_keys=args.models,
            routing_type='involvement',
            device=device,
        )
        # Load trained weights if provided
        if args.checkpoint:
            checkpoint_path = os.path.join(project_root, args.checkpoint)
            state = torch.load(checkpoint_path, map_location=device)
            ensemble.router.load_state_dict(state['router'])
            print(f"Loaded router weights from: {args.checkpoint}")
    
    else:
        raise ValueError(f"Unknown method: {args.method}")
    
    ensemble.eval()
    return ensemble


def get_data_loader(args, device):
    """Get data loader for evaluation."""
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, "ensemble_experiments", args.config)
    
    # Load model config to get data settings
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get first model's checkpoint to extract data config
    first_model_key = list(config['models'].keys())[0]
    checkpoint_path = os.path.join(project_root, config['models'][first_model_key]['checkpoint'])
    
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    from argparse import Namespace
    train_args = Namespace(**state["args"])
    
    # Import data loader
    from src.loaders import get_dataloaders
    
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
    
    return loaders[args.split]


def main():
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = f"ensemble_experiments/outputs/{args.method}"
    output_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create ensemble
    ensemble = create_ensemble(args, device)
    
    # Get data loader
    loader = get_data_loader(args, device)
    
    # Import evaluator
    from src.evaluator import CancerLogitsHeatmapsEvaluator as Evaluator
    
    evaluator = Evaluator(log_images=False)
    accumulator = defaultdict(list)
    
    # Track routing/weighting info for analysis
    if args.method in ["transformer", "moe"]:
        weight_records = []
    
    print(f"\nEvaluating {args.method} on {args.split} split...")
    
    num_samples = 0
    for i, data in enumerate(tqdm(loader, desc="Evaluating")):
        if args.limit and num_samples >= args.limit:
            break
        
        # Inference
        t0 = time.perf_counter()
        
        with torch.amp.autocast('cuda', enabled=args.use_amp):
            with torch.inference_mode():
                out = ensemble(data, deterministic=True)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        infer_time = time.perf_counter() - t0
        accumulator['infer_time'].append(infer_time)
        
        # Merge outputs into data for evaluator
        # Handle different output key names
        if 'cancer_logits' in out and out['cancer_logits'] is not None:
            data['cancer_logits'] = out['cancer_logits']
        
        # Handle classification outputs carefully - evaluator expects specific format
        if 'image_level_classification_outputs' in out:
            cls_out = out['image_level_classification_outputs']
            # Only add if it's a valid tensor with proper shape
            if cls_out is not None and isinstance(cls_out, torch.Tensor) and cls_out.numel() > 0:
                # Evaluator expects (B, C) tensor with softmax-able output
                # Convert to proper format if needed
                if cls_out.dim() == 1:
                    # Convert logits to pseudo-2-class format for softmax compatibility
                    cls_out = torch.stack([-cls_out, cls_out], dim=-1)
                elif cls_out.shape[-1] == 1:
                    # Single logit, convert to 2-class format
                    cls_out = torch.cat([-cls_out, cls_out], dim=-1)
                data['image_level_classification_outputs'] = cls_out
        
        # Track weights for analysis
        if args.method == "transformer" and 'ensemble_weights' in out:
            weights = out['ensemble_weights']
            for b in range(weights.shape[0]):
                weight_records.append({
                    'core_id': data['core_id'][b] if 'core_id' in data else str(num_samples + b),
                    **{f'weight_{k}': weights[b, j].item() 
                       for j, k in enumerate(ensemble.model_keys)},
                })
        
        if args.method == "moe" and 'routing_weights' in out:
            weights = out['routing_weights']
            for b in range(weights.shape[0]):
                weight_records.append({
                    'core_id': data['core_id'][b] if 'core_id' in data else str(num_samples + b),
                    'involvement': data['involvement'][b].item() if 'involvement' in data else None,
                    **{f'weight_{k}': weights[b, j].item() 
                       for j, k in enumerate(ensemble.model_keys)},
                })
        
        # Evaluate
        evaluator(data)
        num_samples += data['bmode'].shape[0]
    
    # Aggregate metrics
    metrics = evaluator.aggregate_metrics()
    metrics['infer_time'] = np.array(accumulator['infer_time']).mean()
    metrics = {k: float(v) for k, v in metrics.items()}
    
    # Print results
    print(f"\n{'='*60}")
    print(f"{args.method.upper()} Ensemble Results ({args.split} split)")
    print(f"{'='*60}")
    
    # Print key metrics
    key_metrics = ['auroc', 'auprc', 'sensitivity', 'specificity', 'balanced_accuracy', 'infer_time']
    for key in key_metrics:
        if key in metrics:
            if key == 'infer_time':
                print(f"  {key}: {metrics[key]*1000:.1f} ms")
            else:
                print(f"  {key}: {metrics[key]:.4f}")
    
    print(f"\nAll metrics:")
    for key, value in sorted(metrics.items()):
        print(f"  {key}: {value:.4f}")
    
    # Save metrics
    metrics_path = os.path.join(output_dir, f"metrics_{args.split}.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to: {metrics_path}")
    
    # Save per-core metrics
    table = evaluator.accumulator.compute()
    table_path = os.path.join(output_dir, f"metrics_by_core_{args.split}.csv")
    table.to_csv(table_path)
    print(f"Saved per-core metrics to: {table_path}")
    
    # Save weight analysis for transformer/moe
    if args.method in ["transformer", "moe"] and weight_records:
        import pandas as pd
        weights_df = pd.DataFrame(weight_records)
        weights_path = os.path.join(output_dir, f"routing_weights_{args.split}.csv")
        weights_df.to_csv(weights_path, index=False)
        print(f"Saved routing weights to: {weights_path}")
        
        # Print weight statistics
        print(f"\nRouting Weight Statistics:")
        for col in weights_df.columns:
            if col.startswith('weight_'):
                model_name = col.replace('weight_', '')
                print(f"  {model_name}: {weights_df[col].mean():.4f} ± {weights_df[col].std():.4f}")
    
    # Save args
    args_path = os.path.join(output_dir, f"args_{args.split}.json")
    with open(args_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Evaluation complete! Results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
