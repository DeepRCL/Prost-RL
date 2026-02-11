"""
Model Soup: Weight averaging across multiple models.

Model soups combine multiple fine-tuned models by averaging their weights.
This assumes all models share the same architecture.

Reference: "Model soups: averaging weights of multiple fine-tuned models improves accuracy 
without increasing inference time" (Wortsman et al., 2022)
"""

import os
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict

import torch
import torch.nn as nn
from tqdm import tqdm

from .base_loader import load_model_from_checkpoint, load_models_from_config, get_project_root


class ModelSoup(nn.Module):
    """
    Model Soup ensemble that averages model weights.
    
    Unlike output ensembles, model soups create a single model with averaged weights,
    meaning inference cost is the same as a single model.
    
    Supports:
    - Uniform averaging: (w1 + w2 + ... + wn) / n
    - Weighted averaging: alpha1*w1 + alpha2*w2 + ... + alphan*wn
    """
    
    def __init__(self, base_model: nn.Module):
        """
        Initialize with a souped model.
        
        Args:
            base_model: Model with averaged weights
        """
        super().__init__()
        self.model = base_model
    
    def forward(self, data, deterministic=True):
        """Forward pass through the souped model."""
        return self.model(data, deterministic=deterministic)
    
    @classmethod
    def from_checkpoints(
        cls,
        checkpoint_paths: List[str],
        weights: Optional[List[float]] = None,
        device: Optional[torch.device] = None,
    ) -> 'ModelSoup':
        """
        Create a model soup from multiple checkpoints.
        
        Args:
            checkpoint_paths: List of paths to checkpoint files
            weights: Optional interpolation weights (must sum to 1). Default is uniform.
            device: Device to load model onto
            
        Returns:
            ModelSoup instance with averaged weights
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if len(checkpoint_paths) < 2:
            raise ValueError("Model soup requires at least 2 checkpoints")
        
        # Set default uniform weights
        if weights is None:
            weights = [1.0 / len(checkpoint_paths)] * len(checkpoint_paths)
        
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1, got {sum(weights)}")
        
        if len(weights) != len(checkpoint_paths):
            raise ValueError(f"Number of weights ({len(weights)}) must match number of checkpoints ({len(checkpoint_paths)})")
        
        print(f"\nCreating model soup from {len(checkpoint_paths)} checkpoints...")
        print(f"Weights: {weights}")
        
        # Load all state dicts
        state_dicts = []
        base_model = None
        
        for i, path in enumerate(checkpoint_paths):
            if not os.path.isabs(path):
                path = os.path.join(get_project_root(), path)
            
            print(f"  Loading checkpoint {i+1}/{len(checkpoint_paths)}: {os.path.basename(path)}")
            state = torch.load(path, map_location=device, weights_only=False)
            state_dicts.append(state["model"])
            
            # Keep track of args from first model for reconstruction
            if i == 0:
                from argparse import Namespace
                train_args = Namespace(**state["args"])
        
        # Average the state dicts
        print("  Averaging weights...")
        averaged_state_dict = average_state_dicts(state_dicts, weights)
        
        # Reconstruct model with averaged weights
        print("  Reconstructing model...")
        from medAI.modeling import create_model
        from train_rl import ProstNFoundMeta
        
        model_name = train_args.model
        model_kw = dict(train_args.model_kw)
        
        # Handle V2 models
        is_v1_model = 'rl_v2' not in model_name
        if not is_v1_model and 'policy_arch_version' in model_kw:
            del model_kw['policy_arch_version']
        
        base_model = create_model(model_name, **model_kw)
        
        from medAI.modeling.prostnfound_rl import ProstNFoundRL
        try:
            from medAI.modeling.prostnfound_rl_v2 import ProstNFoundRLV2
        except ImportError:
            ProstNFoundRLV2 = None
        
        is_rl_model = isinstance(base_model, ProstNFoundRL)
        if ProstNFoundRLV2 is not None:
            is_rl_model = is_rl_model or isinstance(base_model, ProstNFoundRLV2)
        
        model = ProstNFoundMeta(
            base_model,
            is_rl=is_rl_model,
            apply_prostate_mask_to_decoder=True,
            boundary_tolerance_patches=0
        )
        
        # Load averaged weights
        result = model.load_state_dict(averaged_state_dict, strict=False)
        if result.missing_keys:
            print(f"  Missing keys: {len(result.missing_keys)}")
        if result.unexpected_keys:
            print(f"  Unexpected keys: {len(result.unexpected_keys)}")
        
        model.to(device)
        model.eval()
        
        # Freeze
        for param in model.parameters():
            param.requires_grad = False
        
        print(f"  Model soup created successfully!")
        return cls(model)


def average_state_dicts(
    state_dicts: List[Dict[str, torch.Tensor]],
    weights: List[float],
) -> Dict[str, torch.Tensor]:
    """
    Average multiple state dictionaries with given weights.
    
    Args:
        state_dicts: List of state dictionaries
        weights: Interpolation weights for each state dict
        
    Returns:
        Averaged state dictionary
    """
    averaged = OrderedDict()
    
    # Get all keys from first state dict
    keys = list(state_dicts[0].keys())
    
    for key in keys:
        # Check if parameter exists in all state dicts
        if not all(key in sd for sd in state_dicts):
            print(f"  Warning: Key '{key}' not in all state dicts, skipping")
            continue
        
        # Check if shapes match
        shapes = [sd[key].shape for sd in state_dicts]
        if not all(s == shapes[0] for s in shapes):
            print(f"  Warning: Shape mismatch for '{key}': {shapes}, skipping")
            continue
        
        # Average the parameters
        averaged[key] = sum(w * sd[key] for w, sd in zip(weights, state_dicts))
    
    return averaged


def create_uniform_soup(
    checkpoint_paths: List[str],
    device: Optional[torch.device] = None,
) -> ModelSoup:
    """
    Create a uniform model soup with equal weights.
    
    Args:
        checkpoint_paths: List of checkpoint paths
        device: Target device
        
    Returns:
        ModelSoup with uniformly averaged weights
    """
    return ModelSoup.from_checkpoints(checkpoint_paths, weights=None, device=device)


def create_soup_from_config(
    config_path: str = "configs/models.yaml",
    model_keys: Optional[List[str]] = None,
    weights: Optional[List[float]] = None,
    device: Optional[torch.device] = None,
) -> ModelSoup:
    """
    Create a model soup from models defined in config.
    
    Args:
        config_path: Path to models.yaml
        model_keys: Which models to include (None = default_ensemble)
        weights: Optional interpolation weights
        device: Target device
        
    Returns:
        ModelSoup instance
    """
    import yaml
    
    if not os.path.isabs(config_path):
        config_path = os.path.join(get_project_root(), "ensemble_experiments", config_path)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if model_keys is None:
        model_keys = config.get('default_ensemble', list(config['models'].keys()))
    
    # Get checkpoint paths
    checkpoint_paths = [config['models'][key]['checkpoint'] for key in model_keys]
    
    return ModelSoup.from_checkpoints(checkpoint_paths, weights=weights, device=device)


if __name__ == "__main__":
    print("Testing Model Soup creation...")
    
    # Test with default ensemble (3 models)
    soup = create_soup_from_config()
    print(f"\nModel soup created: {type(soup.model)}")
    
    # Test forward pass would require data - skip for now
    print("Model soup test passed!")
