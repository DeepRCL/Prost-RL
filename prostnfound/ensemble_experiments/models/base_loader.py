"""
Base model loading utilities for ensemble experiments.

This module provides functions to load pre-trained ProstNFound models 
from checkpoints and prepare them for ensembling.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from argparse import Namespace

# Set MEDSAM_CHECKPOINT_DIR before importing medAI (which reads it at import time in sam.py)
# Note: sam.py reads MEDSAM_CHECKPOINT_DIR but confusingly warns about CHECKPOINT_DIR
if 'MEDSAM_CHECKPOINT_DIR' not in os.environ:
    # Try common locations
    for path in [
        os.path.expanduser("~/prostnfound/checkpoints"),  # checkpoints subfolder
        os.path.expanduser("~/prostnfound"),  # prostnfound root
    ]:
        medsam_path = os.path.join(path, "medsam_vit_b_cpu.pth")
        if os.path.exists(medsam_path):
            os.environ['MEDSAM_CHECKPOINT_DIR'] = path
            break

import torch
import torch.nn as nn
import yaml

# Import from existing codebase
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from medAI.modeling import create_model
from medAI.modeling.prostnfound_rl import ProstNFoundRL
try:
    from medAI.modeling.prostnfound_rl_v2 import ProstNFoundRLV2
except ImportError:
    ProstNFoundRLV2 = None

from train_rl import ProstNFoundMeta


@dataclass
class ModelLoaderConfig:
    """Configuration for loading a model."""
    name: str
    checkpoint: str
    type: str  # 'ppo', 'gdpo', 'supervised'
    description: str = ""
    
    # Runtime options
    apply_prostate_mask_to_decoder: bool = True
    boundary_tolerance_patches: int = 0


def get_project_root() -> str:
    """Get the prostnfound project root directory."""
    # This file is in prostnfound/ensemble_experiments/models/
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
    apply_prostate_mask_to_decoder: bool = True,
    boundary_tolerance_patches: int = 0,
    freeze: bool = True,
) -> nn.Module:
    """
    Load a single model from a checkpoint file.
    
    Args:
        checkpoint_path: Path to the checkpoint file (absolute or relative to project root)
        device: Device to load model onto
        apply_prostate_mask_to_decoder: Whether to mask decoder output with prostate mask
        boundary_tolerance_patches: Boundary tolerance in patches
        freeze: Whether to freeze model parameters
        
    Returns:
        Loaded model wrapped in ProstNFoundMeta
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Handle relative paths
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(get_project_root(), checkpoint_path)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading model from: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract training args from checkpoint
    train_args = Namespace(**state["args"])
    
    # Get model config
    model_name = train_args.model
    model_kw = dict(train_args.model_kw)
    
    # Handle V2 models - remove policy_arch_version if present
    is_v1_model = 'rl_v2' not in model_name
    if not is_v1_model and 'policy_arch_version' in model_kw:
        del model_kw['policy_arch_version']
    
    # Create base model
    base_model = create_model(model_name, **model_kw)
    
    # Check if RL model
    is_rl_model = isinstance(base_model, ProstNFoundRL)
    if ProstNFoundRLV2 is not None:
        is_rl_model = is_rl_model or isinstance(base_model, ProstNFoundRLV2)
    
    # Wrap in ProstNFoundMeta
    model = ProstNFoundMeta(
        base_model,
        is_rl=is_rl_model,
        apply_prostate_mask_to_decoder=apply_prostate_mask_to_decoder,
        boundary_tolerance_patches=boundary_tolerance_patches
    )
    
    # Load state dict
    result = model.load_state_dict(state["model"], strict=False)
    if result.missing_keys:
        print(f"  Missing keys: {result.missing_keys}")
    if result.unexpected_keys:
        print(f"  Unexpected keys: {result.unexpected_keys}")
    
    model.to(device)
    model.eval()
    
    # Freeze if requested
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
    
    print(f"  Model type: {type(base_model).__name__}, RL: {is_rl_model}, Frozen: {freeze}")
    return model


def load_models_from_config(
    config_path: str,
    model_keys: Optional[List[str]] = None,
    device: Optional[torch.device] = None,
    freeze: bool = True,
) -> Dict[str, nn.Module]:
    """
    Load multiple models from a YAML config file.
    
    Args:
        config_path: Path to models.yaml config file
        model_keys: List of model keys to load (None = load default_ensemble)
        device: Device to load models onto
        freeze: Whether to freeze model parameters
        
    Returns:
        Dictionary mapping model key to loaded model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Handle relative paths
    if not os.path.isabs(config_path):
        config_path = os.path.join(get_project_root(), "ensemble_experiments", config_path)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Determine which models to load
    if model_keys is None:
        model_keys = config.get('default_ensemble', list(config['models'].keys()))
    
    print(f"\n{'='*60}")
    print(f"Loading {len(model_keys)} models for ensemble:")
    print(f"{'='*60}")
    
    models = {}
    for key in model_keys:
        if key not in config['models']:
            raise ValueError(f"Unknown model key: {key}. Available: {list(config['models'].keys())}")
        
        model_cfg = config['models'][key]
        print(f"\n[{key}] {model_cfg['name']} ({model_cfg['type']})")
        
        models[key] = load_model_from_checkpoint(
            checkpoint_path=model_cfg['checkpoint'],
            device=device,
            freeze=freeze,
        )
    
    print(f"\n{'='*60}")
    print(f"Successfully loaded {len(models)} models")
    print(f"{'='*60}\n")
    
    return models


def load_all_models(
    config_path: str = "configs/models.yaml",
    device: Optional[torch.device] = None,
    freeze: bool = True,
) -> Dict[str, nn.Module]:
    """
    Load all models defined in the config.
    
    Args:
        config_path: Path to models.yaml (relative to ensemble_experiments/)
        device: Device to load models onto
        freeze: Whether to freeze model parameters
        
    Returns:
        Dictionary mapping model key to loaded model
    """
    # Handle relative paths
    if not os.path.isabs(config_path):
        config_path = os.path.join(
            get_project_root(), 
            "ensemble_experiments", 
            config_path
        )
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    all_keys = list(config['models'].keys())
    return load_models_from_config(config_path, model_keys=all_keys, device=device, freeze=freeze)


def extract_model_outputs(
    models: Dict[str, nn.Module],
    data: Dict[str, Any],
    deterministic: bool = True,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Run inference on all models and collect their outputs.
    
    Args:
        models: Dictionary of loaded models
        data: Input data batch
        deterministic: Whether to use deterministic policy for RL models
        
    Returns:
        Dictionary mapping model key to output dict containing:
        - 'cancer_logits': (B, 1, H, W)
        - 'cancer_probs': (B, 1, H, W) 
        - 'cls_logits': (B, num_classes) or None
        - 'rl_attention_map': (B, 1, h, w) or None
    """
    outputs = {}
    
    with torch.no_grad():
        for key, model in models.items():
            out = model(data, deterministic=deterministic)
            
            # Extract relevant outputs
            model_out = {}
            
            # Heatmap logits and probs
            if 'cancer_logits' in out:
                model_out['cancer_logits'] = out['cancer_logits']
                model_out['cancer_probs'] = out['cancer_logits'].sigmoid()
            
            # Classification outputs
            if 'image_level_classification_outputs' in out and out['image_level_classification_outputs'] is not None:
                model_out['cls_logits'] = out['image_level_classification_outputs']
            
            # RL attention map
            if 'rl_attention_map' in out and out['rl_attention_map'] is not None:
                model_out['rl_attention_map'] = out['rl_attention_map']
            
            outputs[key] = model_out
    
    return outputs


if __name__ == "__main__":
    # Test loading
    print("Testing model loading...")
    
    # Test loading a single model
    model = load_model_from_checkpoint(
        "checkpoints_rl_v3/V3-PPO-symmetric_ce/best_rl.pth"
    )
    print(f"Single model loaded: {type(model)}")
    
    # Test loading from config
    models = load_models_from_config("configs/models.yaml")
    print(f"Loaded {len(models)} models from config")
