"""
Mixture of Experts (MoE) Ensemble: Learned routing based on input characteristics.

This ensemble uses a router network to dynamically select or weight experts
(models) based on input features. The routing can be conditioned on:
- Classification confidence from each model
- Heatmap statistics (mean, max, std)
- Estimated involvement level
- Input image features

Key insight from experiments:
- GDPO models may be better at low involvement cases
- PPO models may be better at high involvement cases
- Supervised models often have better calibration

The MoE learns to leverage these complementary strengths.
"""

import os
from typing import Dict, List, Optional, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_loader import load_models_from_config, get_project_root


class SimpleRouter(nn.Module):
    """
    Simple router based on aggregated model outputs.
    
    Takes classification logits or heatmap statistics from each model
    and outputs routing weights.
    """
    
    def __init__(
        self,
        num_models: int,
        input_type: str = 'cls',  # 'cls', 'heatmap', 'both'
        hidden_dim: int = 32,
    ):
        """
        Initialize router.
        
        Args:
            num_models: Number of expert models
            input_type: What features to use for routing
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.num_models = num_models
        self.input_type = input_type
        
        # Determine input dimension
        if input_type == 'cls':
            input_dim = num_models  # One cls logit per model
        elif input_type == 'heatmap':
            input_dim = num_models * 3  # mean, max, std per model
        else:  # both
            input_dim = num_models + num_models * 3
        
        self.router = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_models),
        )
    
    def forward(
        self,
        cls_logits: Optional[torch.Tensor] = None,
        heatmaps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute routing weights.
        
        Args:
            cls_logits: (B, num_models) classification logits
            heatmaps: (B, num_models, H, W) heatmap probabilities
            
        Returns:
            weights: (B, num_models) routing weights (sum to 1)
        """
        features = []
        
        if self.input_type in ['cls', 'both'] and cls_logits is not None:
            features.append(cls_logits)
        
        if self.input_type in ['heatmap', 'both'] and heatmaps is not None:
            # Compute statistics per model
            # heatmaps: (B, num_models, H, W)
            hm_mean = heatmaps.mean(dim=(-2, -1))  # (B, num_models)
            hm_max = heatmaps.amax(dim=(-2, -1))   # (B, num_models)
            hm_std = heatmaps.std(dim=(-2, -1))    # (B, num_models)
            features.extend([hm_mean, hm_max, hm_std])
        
        # Concatenate features
        x = torch.cat(features, dim=1)
        
        # Route
        logits = self.router(x)
        weights = F.softmax(logits, dim=1)
        
        return weights


class InvolvementAwareRouter(nn.Module):
    """
    Router that explicitly uses estimated involvement as a routing signal.
    
    The intuition is that different models may excel at different
    involvement levels, so we learn to route based on predicted involvement.
    """
    
    def __init__(
        self,
        num_models: int,
        hidden_dim: int = 64,
    ):
        """
        Initialize involvement-aware router.
        
        Args:
            num_models: Number of expert models
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.num_models = num_models
        
        # Input: cls logits + heatmap stats + estimated involvement
        # involvement is computed as mean(heatmap within needle mask)
        input_dim = num_models + num_models * 3 + num_models  # cls + stats + involvement
        
        self.router = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_models),
        )
    
    def forward(
        self,
        cls_logits: torch.Tensor,
        heatmaps: torch.Tensor,
        needle_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute routing weights based on involvement estimates.
        
        Args:
            cls_logits: (B, num_models) classification logits
            heatmaps: (B, num_models, H, W) heatmap probabilities
            needle_mask: (B, 1, H, W) needle region mask (optional)
            
        Returns:
            weights: (B, num_models) routing weights
        """
        B = heatmaps.shape[0]
        
        # Compute heatmap stats
        hm_mean = heatmaps.mean(dim=(-2, -1))  # (B, num_models)
        hm_max = heatmaps.amax(dim=(-2, -1))
        hm_std = heatmaps.std(dim=(-2, -1))
        
        # Compute per-model involvement estimate
        if needle_mask is not None:
            # Resize mask to match heatmap if needed
            if needle_mask.shape[-2:] != heatmaps.shape[-2:]:
                needle_mask = F.interpolate(
                    needle_mask.float(),
                    size=heatmaps.shape[-2:],
                    mode='nearest'
                )
            
            # Expand mask for all models: (B, 1, H, W) -> (B, num_models, H, W)
            mask_expanded = needle_mask.expand(-1, self.num_models, -1, -1)
            
            # Compute mean within needle region per model
            masked_heatmaps = heatmaps * mask_expanded
            involvement = masked_heatmaps.sum(dim=(-2, -1)) / (mask_expanded.sum(dim=(-2, -1)) + 1e-8)
        else:
            # Fallback: use global mean as involvement proxy
            involvement = hm_mean
        
        # Concatenate all features
        x = torch.cat([cls_logits, hm_mean, hm_max, hm_std, involvement], dim=1)
        
        # Route
        logits = self.router(x)
        weights = F.softmax(logits, dim=1)
        
        return weights


class InvolvementAwareMoE(nn.Module):
    """
    Mixture of Experts ensemble with involvement-aware routing.
    
    Combines multiple models with learned routing that considers
    predicted involvement level and model confidence.
    """
    
    def __init__(
        self,
        models: Dict[str, nn.Module],
        routing_type: str = 'involvement',  # 'simple', 'involvement'
        hidden_dim: int = 64,
    ):
        """
        Initialize MoE ensemble.
        
        Args:
            models: Dictionary of expert models
            routing_type: Type of router to use
            hidden_dim: Hidden dimension for router
        """
        super().__init__()
        
        self.model_keys = list(models.keys())
        self.num_models = len(models)
        
        # Store models (frozen)
        self.models = nn.ModuleDict(models)
        for model in self.models.values():
            for param in model.parameters():
                param.requires_grad = False
        
        # Router
        if routing_type == 'simple':
            self.router = SimpleRouter(
                num_models=self.num_models,
                input_type='both',
                hidden_dim=hidden_dim,
            )
        else:  # involvement
            self.router = InvolvementAwareRouter(
                num_models=self.num_models,
                hidden_dim=hidden_dim,
            )
        
        self.routing_type = routing_type
    
    def forward(self, data: Dict[str, Any], deterministic: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass through MoE ensemble.
        
        Args:
            data: Input data batch
            deterministic: Whether to use deterministic policy
            
        Returns:
            Dictionary with ensemble outputs
        """
        # Collect outputs from all models
        heatmaps = []
        cls_logits = []
        individual_outputs = {}
        
        with torch.no_grad():
            for key in self.model_keys:
                model = self.models[key]
                out = model(data, deterministic=deterministic)
                individual_outputs[key] = out
                
                # Collect heatmaps (as probabilities)
                if 'cancer_logits' in out:
                    heatmaps.append(out['cancer_logits'].sigmoid())
                
                # Collect classification logits
                if 'image_level_classification_outputs' in out and out['image_level_classification_outputs'] is not None:
                    cls = out['image_level_classification_outputs']
                    if cls.dim() > 1 and cls.shape[1] > 1:
                        cls = cls[:, 1:2]
                    cls_logits.append(cls.squeeze(-1) if cls.dim() > 1 else cls)
        
        # Stack tensors
        heatmaps = torch.stack(heatmaps, dim=1).squeeze(2)  # (B, num_models, H, W)
        cls_logits = torch.stack(cls_logits, dim=1)  # (B, num_models)
        
        # Get needle mask if available
        needle_mask = data.get('needle_mask', None)
        
        # Compute routing weights
        if self.routing_type == 'involvement' and hasattr(self.router, 'forward'):
            weights = self.router(cls_logits, heatmaps, needle_mask)
        else:
            weights = self.router(cls_logits, heatmaps)
        
        # Weighted combination
        # heatmaps: (B, num_models, H, W)
        weights_expanded = weights.unsqueeze(-1).unsqueeze(-1)  # (B, num_models, 1, 1)
        combined_heatmap = (heatmaps * weights_expanded).sum(dim=1, keepdim=True)  # (B, 1, H, W)
        
        # Weighted classification
        combined_cls = (cls_logits * weights).sum(dim=1, keepdim=True)  # (B, 1)
        
        return {
            'cancer_logits': torch.logit(combined_heatmap.clamp(1e-7, 1-1e-7)),
            'cancer_probs': combined_heatmap,
            'image_level_classification_outputs': combined_cls,
            'routing_weights': weights,
            'individual_outputs': individual_outputs,
        }
    
    def get_trainable_params(self):
        """Get only the trainable (router) parameters."""
        return list(self.router.parameters())
    
    def analyze_routing(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze routing decisions for a batch.
        
        Returns statistics about how the router is weighting models.
        """
        out = self.forward(data, deterministic=True)
        weights = out['routing_weights']  # (B, num_models)
        
        analysis = {
            'mean_weights': {k: w.item() for k, w in zip(self.model_keys, weights.mean(dim=0))},
            'std_weights': {k: w.item() for k, w in zip(self.model_keys, weights.std(dim=0))},
            'max_weights': {k: w.item() for k, w in zip(self.model_keys, weights.max(dim=0)[0])},
        }
        
        return analysis


def create_moe_ensemble(
    config_path: str = "configs/models.yaml",
    model_keys: Optional[List[str]] = None,
    routing_type: str = 'involvement',
    hidden_dim: int = 64,
    device: Optional[torch.device] = None,
) -> InvolvementAwareMoE:
    """
    Create a MoE ensemble from config.
    
    Args:
        config_path: Path to models.yaml
        model_keys: Which models to include
        routing_type: 'simple' or 'involvement'
        hidden_dim: Hidden dimension for router
        device: Target device
        
    Returns:
        InvolvementAwareMoE instance
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load frozen models
    models = load_models_from_config(config_path, model_keys=model_keys, device=device, freeze=True)
    
    ensemble = InvolvementAwareMoE(
        models,
        routing_type=routing_type,
        hidden_dim=hidden_dim,
    )
    
    return ensemble.to(device)


if __name__ == "__main__":
    print("Testing MoE Ensemble...")
    
    # Test simple router
    router = SimpleRouter(num_models=3, input_type='both')
    print(f"Simple router parameters: {sum(p.numel() for p in router.parameters()):,}")
    
    # Test involvement-aware router
    inv_router = InvolvementAwareRouter(num_models=3)
    print(f"Involvement router parameters: {sum(p.numel() for p in inv_router.parameters()):,}")
    
    # Test with dummy data
    B = 2
    cls_logits = torch.rand(B, 3)
    heatmaps = torch.rand(B, 3, 64, 64)
    needle_mask = torch.rand(B, 1, 64, 64) > 0.5
    
    weights_simple = router(cls_logits, heatmaps)
    weights_inv = inv_router(cls_logits, heatmaps, needle_mask.float())
    
    print(f"\nSimple router weights: {weights_simple}")
    print(f"Involvement router weights: {weights_inv}")
    print(f"Weights sum: {weights_inv.sum(dim=1)}")
    
    print("\nMoE ensemble test passed!")
