"""
Output Ensemble: Averaging model outputs (not weights).

This ensemble method runs multiple models in parallel and combines their
predictions (heatmaps, classification logits) using averaging.

Supports:
- Simple averaging: (p1 + p2 + ... + pn) / n
- Weighted averaging: w1*p1 + w2*p2 + ... + wn*pn
- Learned weights: Optimize weights on validation set
"""

import os
from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .base_loader import load_models_from_config, extract_model_outputs, get_project_root


class OutputEnsemble(nn.Module):
    """
    Ensemble that combines model outputs with learnable weights.
    
    All base models are frozen. Only the combination weights are trainable.
    """
    
    def __init__(
        self,
        models: Dict[str, nn.Module],
        init_weights: Optional[torch.Tensor] = None,
        learn_weights: bool = True,
    ):
        """
        Initialize output ensemble.
        
        Args:
            models: Dictionary of models (key -> model)
            init_weights: Initial ensemble weights (default: uniform)
            learn_weights: Whether weights should be learnable
        """
        super().__init__()
        
        self.model_keys = list(models.keys())
        self.num_models = len(models)
        
        # Store models as a ModuleDict
        self.models = nn.ModuleDict(models)
        
        # Freeze all base models
        for model in self.models.values():
            for param in model.parameters():
                param.requires_grad = False
        
        # Initialize weights
        if init_weights is None:
            init_weights = torch.ones(self.num_models) / self.num_models
        
        if learn_weights:
            # Learnable weights (logits, will be softmaxed)
            self.weight_logits = nn.Parameter(torch.zeros(self.num_models))
        else:
            self.register_buffer('weight_logits', torch.log(init_weights + 1e-8))
    
    def get_weights(self) -> torch.Tensor:
        """Get normalized ensemble weights."""
        return F.softmax(self.weight_logits, dim=0)
    
    def forward(self, data: Dict[str, Any], deterministic: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass through all models and combine outputs.
        
        Args:
            data: Input data batch
            deterministic: Whether to use deterministic policy for RL models
            
        Returns:
            Dictionary with combined outputs:
            - 'cancer_logits': Combined heatmap logits (B, 1, H, W)
            - 'cancer_probs': Combined heatmap probabilities (B, 1, H, W)
            - 'cls_logits': Combined classification logits (B, C)
            - 'ensemble_weights': Current ensemble weights
            - 'individual_outputs': Dict of individual model outputs
        """
        weights = self.get_weights()
        
        # Collect outputs from all models
        all_heatmap_probs = []
        all_cls_logits = []
        individual_outputs = {}
        
        for i, key in enumerate(self.model_keys):
            model = self.models[key]
            out = model(data, deterministic=deterministic)
            individual_outputs[key] = out
            
            # Collect heatmap probs
            if 'cancer_logits' in out:
                all_heatmap_probs.append(out['cancer_logits'].sigmoid())
            
            # Collect classification logits
            if 'image_level_classification_outputs' in out and out['image_level_classification_outputs'] is not None:
                cls_out = out['image_level_classification_outputs']
                # Handle case where cls_out is a list of tensors
                if isinstance(cls_out, list):
                    # Stack list of tensors 
                    cls_out = torch.stack(cls_out, dim=0) if len(cls_out) > 0 else None
                if cls_out is not None:
                    # Ensure it's a 2D tensor (B, C)
                    if cls_out.dim() == 1:
                        cls_out = cls_out.unsqueeze(-1)
                    all_cls_logits.append(cls_out)
        
        # Combine heatmaps
        if all_heatmap_probs:
            # Stack: (num_models, B, 1, H, W)
            heatmap_stack = torch.stack(all_heatmap_probs, dim=0)
            # Weighted average
            weights_view = weights.view(-1, 1, 1, 1, 1)
            combined_probs = (heatmap_stack * weights_view).sum(dim=0)
            # Convert back to logits for compatibility
            combined_logits = torch.logit(combined_probs.clamp(1e-7, 1-1e-7))
        else:
            combined_probs = None
            combined_logits = None
        
        # Combine classification logits
        if all_cls_logits and len(all_cls_logits) == self.num_models:
            # Ensure all have same shape - use the min shape
            try:
                cls_stack = torch.stack(all_cls_logits, dim=0)
                # Weighted average of logits (or could average probs)
                weights_view = weights.view(-1, 1, 1)
                combined_cls = (cls_stack * weights_view).sum(dim=0)
            except Exception as e:
                # Fall back to simple average if shapes don't match
                print(f"Warning: Could not stack cls logits, using simple average: {e}")
                combined_cls = sum(all_cls_logits) / len(all_cls_logits)
        else:
            combined_cls = None
        
        return {
            'cancer_logits': combined_logits,
            'cancer_probs': combined_probs,
            'image_level_classification_outputs': combined_cls,
            'ensemble_weights': weights.detach(),
            'individual_outputs': individual_outputs,
        }
    
    def print_weights(self):
        """Print current ensemble weights."""
        weights = self.get_weights()
        print("\nEnsemble Weights:")
        for key, w in zip(self.model_keys, weights):
            print(f"  {key}: {w.item():.4f}")


class SimpleAverageEnsemble(OutputEnsemble):
    """
    Simple output ensemble with fixed uniform weights.
    
    This is the simplest ensemble method - just average all model outputs.
    """
    
    def __init__(self, models: Dict[str, nn.Module]):
        super().__init__(models, learn_weights=False)


def create_output_ensemble(
    config_path: str = "configs/models.yaml",
    model_keys: Optional[List[str]] = None,
    learn_weights: bool = False,
    device: Optional[torch.device] = None,
) -> OutputEnsemble:
    """
    Create an output ensemble from config.
    
    Args:
        config_path: Path to models.yaml
        model_keys: Which models to include (None = default_ensemble)
        learn_weights: Whether to make weights learnable
        device: Target device
        
    Returns:
        OutputEnsemble instance
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load models
    models = load_models_from_config(config_path, model_keys=model_keys, device=device, freeze=True)
    
    if learn_weights:
        ensemble = OutputEnsemble(models, learn_weights=True)
    else:
        ensemble = SimpleAverageEnsemble(models)
    
    return ensemble.to(device)


def optimize_ensemble_weights(
    ensemble: OutputEnsemble,
    val_loader,
    criterion: Optional[nn.Module] = None,
    num_epochs: int = 10,
    lr: float = 0.1,
) -> OutputEnsemble:
    """
    Optimize ensemble weights on validation set.
    
    Args:
        ensemble: OutputEnsemble with learnable weights
        val_loader: Validation data loader
        criterion: Loss function (default: BCE for heatmaps)
        num_epochs: Number of optimization epochs
        lr: Learning rate for weights
        
    Returns:
        Ensemble with optimized weights
    """
    if criterion is None:
        criterion = nn.BCEWithLogitsLoss()
    
    # Only optimize weights
    optimizer = torch.optim.Adam([ensemble.weight_logits], lr=lr)
    
    print(f"\nOptimizing ensemble weights for {num_epochs} epochs...")
    ensemble.print_weights()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0
        
        for data in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
            
            out = ensemble(data, deterministic=True)
            
            # Compute loss on heatmap predictions
            if out['cancer_logits'] is not None and 'cancer_mask' in data:
                target = data['cancer_mask'].float()
                if target.shape[-2:] != out['cancer_logits'].shape[-2:]:
                    target = F.interpolate(target, size=out['cancer_logits'].shape[-2:], mode='nearest')
                
                loss = criterion(out['cancer_logits'], target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        print(f"  Epoch {epoch+1}: avg loss = {avg_loss:.4f}")
    
    print("\nOptimized weights:")
    ensemble.print_weights()
    
    return ensemble


if __name__ == "__main__":
    print("Testing Output Ensemble creation...")
    
    # Test creating ensemble
    ensemble = create_output_ensemble(learn_weights=False)
    print(f"\nEnsemble created with {ensemble.num_models} models")
    ensemble.print_weights()
    
    print("Output ensemble test passed!")
