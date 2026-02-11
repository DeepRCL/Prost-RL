#!/usr/bin/env python
"""
Training script for learnable ensemble methods (transformer aggregator, MoE).

Usage:
    # Train transformer aggregator
    python train_ensemble.py --method transformer \
        --epochs 20 \
        --lr 1e-4 \
        --output_dir checkpoints/transformer_agg

    # Train MoE router
    python train_ensemble.py --method moe \
        --epochs 10 \
        --lr 1e-3 \
        --output_dir checkpoints/moe

    # Train with weighted output ensemble
    python train_ensemble.py --method weighted_ensemble \
        --epochs 5 \
        --lr 0.1 \
        --output_dir checkpoints/weighted_ensemble
"""

import argparse
import json
import os
import sys
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
import torch.nn as nn
import torch.nn.functional as F
import yaml
from tqdm import tqdm

# Import ensemble methods
from ensemble_experiments.models.output_ensemble import create_output_ensemble, optimize_ensemble_weights
from ensemble_experiments.models.transformer_aggregator import create_transformer_ensemble
from ensemble_experiments.models.moe_ensemble import create_moe_ensemble


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train learnable ensemble methods",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["weighted_ensemble", "transformer", "moe"],
        help="Ensemble method to train"
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
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for checkpoints"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size (default: use from data config)"
    )
    
    parser.add_argument(
        "--use_amp",
        action="store_true",
        default=True,
        help="Use automatic mixed precision"
    )
    
    parser.add_argument(
        "--val_every",
        type=int,
        default=1,
        help="Validate every N epochs"
    )
    
    parser.add_argument(
        "--save_every",
        type=int,
        default=5,
        help="Save checkpoint every N epochs"
    )
    
    parser.add_argument(
        "--heatmap_weight",
        type=float,
        default=10.0,
        help="Weight for heatmap loss (default: 10.0 to balance with cls_loss)"
    )
    
    parser.add_argument(
        "--cls_weight",
        type=float,
        default=1.0,
        help="Weight for classification loss (default: 1.0)"
    )
    
    parser.add_argument(
        "--heatmap_only",
        action="store_true",
        default=False,
        help="Train only on heatmap loss (ignore classification)"
    )
    
    return parser.parse_args()


def get_data_loaders(args):
    """Get train and validation data loaders."""
    
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
        loaders = get_dataloaders(data_args, mode="train")
    elif hasattr(train_args, 'data'):
        loaders = get_dataloaders(train_args.data, mode="train")
    else:
        loaders = get_dataloaders(train_args, mode="train")
    
    return loaders['train'], loaders['val']


def create_ensemble(args, device):
    """Create the appropriate ensemble model."""
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, "ensemble_experiments", args.config)
    
    print(f"\nCreating {args.method} ensemble...")
    
    if args.method == "weighted_ensemble":
        ensemble = create_output_ensemble(
            config_path=config_path,
            model_keys=args.models,
            learn_weights=True,
            device=device,
        )
        
    elif args.method == "transformer":
        ensemble = create_transformer_ensemble(
            config_path=config_path,
            model_keys=args.models,
            use_attention_maps=True,
            hidden_dim=128,
            device=device,
        )
        
    elif args.method == "moe":
        ensemble = create_moe_ensemble(
            config_path=config_path,
            model_keys=args.models,
            routing_type='involvement',
            hidden_dim=64,
            device=device,
        )
    
    else:
        raise ValueError(f"Unknown method: {args.method}")
    
    return ensemble


def compute_loss(ensemble, data, out, method, heatmap_weight=10.0, cls_weight=1.0, heatmap_only=False):
    """Compute loss for training the ensemble.
    
    Args:
        ensemble: The ensemble model
        data: Input data batch
        out: Model outputs
        method: Ensemble method name
        heatmap_weight: Weight for heatmap loss (default 10.0 to balance with cls)
        cls_weight: Weight for classification loss (default 1.0)
        heatmap_only: If True, ignore classification loss
    """
    
    loss = 0.0
    loss_dict = {}
    
    # Heatmap loss (BCE)
    if 'cancer_logits' in out and out['cancer_logits'] is not None:
        device = out['cancer_logits'].device
        
        # Target: needle mask weighted by involvement OR cancer_mask if available
        if 'cancer_mask' in data:
            target = data['cancer_mask'].float().to(device)
        else:
            # Use needle mask weighted by involvement as soft target
            needle_mask = data.get('needle_mask')
            if needle_mask is not None:
                target = needle_mask.float().to(device)
            else:
                target = torch.zeros_like(out['cancer_logits'])
            involvement = data.get('involvement', torch.ones(target.shape[0], 1, 1, 1))
            if involvement.dim() == 1:
                involvement = involvement.view(-1, 1, 1, 1)
            involvement = involvement.to(device)
            target = target * involvement
        
        # Resize if needed
        if target.shape[-2:] != out['cancer_logits'].shape[-2:]:
            target = F.interpolate(target, size=out['cancer_logits'].shape[-2:], mode='nearest')
        
        heatmap_loss = F.binary_cross_entropy_with_logits(out['cancer_logits'], target)
        loss = loss + heatmap_weight * heatmap_loss
        loss_dict['heatmap_loss'] = heatmap_loss.item()
        loss_dict['weighted_heatmap_loss'] = (heatmap_weight * heatmap_loss).item()
    
    # Classification loss (BCE) - skip if heatmap_only
    if not heatmap_only and 'image_level_classification_outputs' in out and out['image_level_classification_outputs'] is not None:
        cls_out = out['image_level_classification_outputs']
        batch_size = data['label'].shape[0]
        cls_target = data['label'].float().to(cls_out.device)
        
        # Handle different cls_out shapes
        if cls_out.dim() == 3:
            # Shape is (num_models, B, C) - need to reduce first
            # Average across models if not already done
            cls_out = cls_out.mean(dim=0)
        
        if cls_out.dim() == 2 and cls_out.shape[-1] == 2:
            # 2-class output, extract cancer logit (class 1 - class 0)
            cls_out = cls_out[:, 1] - cls_out[:, 0]
            cls_out = cls_out.view(-1)
        elif cls_out.dim() == 2 and cls_out.shape[-1] == 1:
            cls_out = cls_out.view(-1)
        
        cls_target = cls_target.view(-1)
        
        cls_loss = F.binary_cross_entropy_with_logits(cls_out, cls_target)
        loss = loss + cls_weight * cls_loss
        loss_dict['cls_loss'] = cls_loss.item()
        loss_dict['weighted_cls_loss'] = (cls_weight * cls_loss).item()
    
    loss_dict['total_loss'] = loss.item()
    
    return loss, loss_dict


def train_epoch(ensemble, train_loader, optimizer, scaler, args, device):
    """Train for one epoch."""
    
    ensemble.train()
    
    # Only aggregator/router should be in train mode
    # Base models should stay in eval mode
    if hasattr(ensemble, 'models'):
        for model in ensemble.models.values():
            model.eval()
    
    total_loss = 0.0
    loss_counts = defaultdict(float)
    num_batches = 0
    weight_sum = None
    
    for data in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda', enabled=args.use_amp):
            out = ensemble(data, deterministic=True)
            loss, loss_dict = compute_loss(ensemble, data, out, args.method, args.heatmap_weight, args.cls_weight, args.heatmap_only)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        for k, v in loss_dict.items():
            loss_counts[k] += v
        num_batches += 1
        
        # Track weights and heatmap diversity for the first few batches
        if num_batches <= 3 and 'ensemble_weights' in out:
            weights = out['ensemble_weights']
            if weight_sum is None:
                weight_sum = weights.mean(dim=0).detach().cpu()
            else:
                weight_sum = weight_sum + weights.mean(dim=0).detach().cpu()
        
        # Measure heatmap diversity on first batch
        if num_batches == 1 and 'individual_outputs' in out:
            heatmaps_list = []
            print(f"  [DEBUG] individual_outputs has {len(out['individual_outputs'])} models")
            for key, ind_out in out['individual_outputs'].items():
                if 'cancer_logits' in ind_out:
                    h = ind_out['cancer_logits'].detach()
                    # Print stats for each model
                    print(f"  [DEBUG] {key}: shape={h.shape}, mean={h.mean().item():.4f}, std={h.std().item():.4f}, id={id(h)}")
                    heatmaps_list.append(h.clone())  # Clone to avoid reference issues
            if len(heatmaps_list) >= 2:
                stacked = torch.stack(heatmaps_list, dim=0)  # (num_models, B, 1, H, W)
                # Compute variance across models for each pixel
                var_across_models = stacked.var(dim=0).mean().item()
                mean_val = stacked.mean().item()
                # Also compute pairwise differences
                max_diff = 0
                for i in range(len(heatmaps_list)):
                    for j in range(i+1, len(heatmaps_list)):
                        diff = (heatmaps_list[i] - heatmaps_list[j]).abs().max().item()
                        max_diff = max(max_diff, diff)
                print(f"  [DIVERSITY] Variance: {var_across_models:.6f}, Max pairwise diff: {max_diff:.6f}")
    
    avg_losses = {k: v / num_batches for k, v in loss_counts.items()}
    
    # Print average weights from first few batches
    if weight_sum is not None:
        avg_weights = weight_sum / min(3, num_batches)
        print(f"  Avg ensemble weights (first 3 batches): {avg_weights.numpy()}")
    
    # Check gradients on WEIGHT_HEAD specifically (not just encoder)
    if hasattr(ensemble, 'aggregator'):
        # Check weight_head gradients
        if hasattr(ensemble.aggregator, 'weight_head'):
            for name, param in ensemble.aggregator.weight_head.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    print(f"  Grad norm for weight_head.{name}: {grad_norm:.6f}")
                break
        # Also check heatmap_encoder
        for name, param in list(ensemble.aggregator.named_parameters())[:2]:
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                print(f"  Grad norm for {name}: {grad_norm:.6f}")
            break
    
    return avg_losses


def validate(ensemble, val_loader, args, device):
    """Validate the ensemble."""
    
    ensemble.eval()
    
    from src.evaluator import CancerLogitsHeatmapsEvaluator as Evaluator
    evaluator = Evaluator(log_images=False)
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for data in tqdm(val_loader, desc="Validating"):
            with torch.amp.autocast('cuda', enabled=args.use_amp):
                out = ensemble(data, deterministic=True)
                loss, _ = compute_loss(ensemble, data, out, args.method, args.heatmap_weight, args.cls_weight, args.heatmap_only)
            
            # Update data for evaluator
            if 'cancer_logits' in out and out['cancer_logits'] is not None:
                data['cancer_logits'] = out['cancer_logits']
                
                # CRITICAL: Compute average_needle_heatmap_value from combined output!
                # This is what the evaluator uses for AUC calculation
                cancer_probs = out['cancer_logits'].sigmoid()
                needle_mask = data['needle_mask'].to(cancer_probs.device)
                
                # Resize if needed
                if cancer_probs.shape[-2:] != needle_mask.shape[-2:]:
                    needle_mask = F.interpolate(needle_mask.float(), size=cancer_probs.shape[-2:], mode='nearest')
                
                # Compute mean prediction in needle region for each sample
                B = cancer_probs.shape[0]
                mean_preds = []
                for b in range(B):
                    mask = needle_mask[b] > 0.5
                    if mask.sum() > 0:
                        mean_pred = cancer_probs[b][mask].mean()
                    else:
                        mean_pred = cancer_probs[b].mean()
                    mean_preds.append(mean_pred)
                data['average_needle_heatmap_value'] = torch.stack(mean_preds)
            
            if 'image_level_classification_outputs' in out and out['image_level_classification_outputs'] is not None:
                cls_out = out['image_level_classification_outputs']
                # Evaluator expects: list containing (B, 2) tensor for softmax
                # We have: (B, 1) logit tensor
                # Convert: stack with negative to create 2-class format
                if cls_out.dim() == 1:
                    cls_out = cls_out.unsqueeze(-1)  # (B,) -> (B, 1)
                # Create 2-class format: [-logit, logit] so softmax gives [1-p, p]
                cls_out_2class = torch.cat([-cls_out, cls_out], dim=-1)  # (B, 2)
                data['image_level_classification_outputs'] = [cls_out_2class]
            
            evaluator(data)
            total_loss += loss.item()
            num_batches += 1
    
    metrics = evaluator.aggregate_metrics()
    metrics['val_loss'] = total_loss / num_batches
    
    return {k: float(v) for k, v in metrics.items()}


def main():
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = f"ensemble_experiments/checkpoints/{args.method}"
    output_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Get data loaders
    train_loader, val_loader = get_data_loaders(args)
    print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")
    
    # Create ensemble
    ensemble = create_ensemble(args, device)
    
    # Get trainable parameters
    if args.method == "weighted_ensemble":
        trainable_params = [ensemble.weight_logits]
    elif args.method == "transformer":
        trainable_params = ensemble.get_trainable_params()
    elif args.method == "moe":
        trainable_params = ensemble.get_trainable_params()
    
    # Count parameters
    num_trainable = sum(p.numel() for p in trainable_params)
    print(f"Trainable parameters: {num_trainable:,}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=args.use_amp)
    
    # Training loop
    best_auroc = 0.0
    best_epoch = 0
    history = []
    
    print(f"\n{'='*60}")
    print(f"Training {args.method} for {args.epochs} epochs")
    print(f"{'='*60}\n")
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 40)
        
        # Train
        train_losses = train_epoch(ensemble, train_loader, optimizer, scaler, args, device)
        scheduler.step()
        
        print(f"Train losses: {train_losses}")
        
        # Validate
        if epoch % args.val_every == 0:
            val_metrics = validate(ensemble, val_loader, args, device)
            print(f"Val metrics: core_auc={val_metrics.get('core_auc', 0):.4f}, "
                  f"sens@80spe={val_metrics.get('sens_at_80_spe', 0):.4f}, "
                  f"loss={val_metrics.get('val_loss', 0):.4f}")
            
            # Track best
            current_auroc = val_metrics.get('core_auc', 0)
            if current_auroc > best_auroc:
                best_auroc = current_auroc
                best_epoch = epoch
                
                # Save best checkpoint
                save_checkpoint(ensemble, optimizer, epoch, val_metrics, 
                               os.path.join(output_dir, "best.pth"), args)
                print(f"  -> New best! Saved to best.pth")
            
            history.append({
                'epoch': epoch,
                'train_losses': train_losses,
                'val_metrics': val_metrics,
            })
        
        # Save periodic checkpoint
        if epoch % args.save_every == 0:
            save_checkpoint(ensemble, optimizer, epoch, val_metrics,
                           os.path.join(output_dir, f"epoch_{epoch}.pth"), args)
    
    # Save final checkpoint
    save_checkpoint(ensemble, optimizer, args.epochs, val_metrics,
                   os.path.join(output_dir, "final.pth"), args)
    
    # Save training history
    history_path = os.path.join(output_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save config
    with open(os.path.join(output_dir, "train_args.json"), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best AUROC: {best_auroc:.4f} at epoch {best_epoch}")
    print(f"Checkpoints saved to: {output_dir}")
    
    # ---------------------------------------------------------
    # Final Evaluation with Best Model
    # ---------------------------------------------------------
    print(f"\nRunning final evaluation with best model...")
    best_path = os.path.join(output_dir, "best.pth")
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path)
        
        # Load best weights
        if args.method == "weighted_ensemble":
            ensemble.weight_logits.data = checkpoint['weight_logits'].to(device)
        elif args.method == "transformer":
            ensemble.aggregator.load_state_dict(checkpoint['aggregator'])
        elif args.method == "moe":
            ensemble.router.load_state_dict(checkpoint['router'])
            
        ensemble.eval()
        
        from src.evaluator import CancerLogitsHeatmapsEvaluator as Evaluator
        evaluator = Evaluator(log_images=False)
        
        with torch.no_grad():
            for data in tqdm(val_loader, desc="Final Evaluation"):
                with torch.amp.autocast('cuda', enabled=args.use_amp):
                    out = ensemble(data, deterministic=True)
                
                # Update data for evaluator
                if 'cancer_logits' in out:
                    data['cancer_logits'] = out['cancer_logits']
                    
                    # CRITICAL: Compute average_needle_heatmap_value from combined output!
                    cancer_probs = out['cancer_logits'].sigmoid()
                    needle_mask = data['needle_mask'].to(cancer_probs.device)
                    
                    if cancer_probs.shape[-2:] != needle_mask.shape[-2:]:
                        needle_mask = F.interpolate(needle_mask.float(), size=cancer_probs.shape[-2:], mode='nearest')
                    
                    B = cancer_probs.shape[0]
                    mean_preds = []
                    for b in range(B):
                        mask = needle_mask[b] > 0.5
                        if mask.sum() > 0:
                            mean_pred = cancer_probs[b][mask].mean()
                        else:
                            mean_pred = cancer_probs[b].mean()
                        mean_preds.append(mean_pred)
                    data['average_needle_heatmap_value'] = torch.stack(mean_preds)
                
                if 'image_level_classification_outputs' in out and out['image_level_classification_outputs'] is not None:
                    cls_out = out['image_level_classification_outputs']
                    # Evaluator expects: list containing (B, 2) tensor for softmax
                    if cls_out.dim() == 1:
                        cls_out = cls_out.unsqueeze(-1)
                    cls_out_2class = torch.cat([-cls_out, cls_out], dim=-1)
                    data['image_level_classification_outputs'] = [cls_out_2class]
                
                evaluator(data)
        
        # Save metrics.json
        metrics = evaluator.aggregate_metrics()
        metrics = {k: float(v) for k, v in metrics.items()}
        
        metrics_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics.json to {metrics_path}")
        
        # Save per-core metrics
        if evaluator.results_table is not None:
            csv_path = os.path.join(output_dir, "metrics_by_core.csv")
            evaluator.results_table.to_csv(csv_path, index=False)
            print(f"Saved metrics_by_core.csv to {csv_path}")
            
    print(f"{'='*60}")


def save_checkpoint(ensemble, optimizer, epoch, metrics, path, args):
    """Save a checkpoint."""
    
    checkpoint = {
        'epoch': epoch,
        'metrics': metrics,
        'args': vars(args),
    }
    
    if args.method == "weighted_ensemble":
        checkpoint['weight_logits'] = ensemble.weight_logits.detach().cpu()
    elif args.method == "transformer":
        checkpoint['aggregator'] = ensemble.aggregator.state_dict()
    elif args.method == "moe":
        checkpoint['router'] = ensemble.router.state_dict()
    
    checkpoint['optimizer'] = optimizer.state_dict()
    
    torch.save(checkpoint, path)


if __name__ == "__main__":
    main()
