"""
Transformer Aggregator: Cross-model attention for combining outputs.

This aggregator uses a transformer to learn how to combine outputs from
multiple models. It can leverage:
- Heatmap predictions (spatial information)
- Classification logits (confidence)
- RL attention maps (where each model focused)

The transformer learns cross-model attention to produce optimal weighted
combinations that may vary per-sample.
"""

import os
from typing import Dict, List, Optional, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_loader import load_models_from_config, get_project_root


class HeatmapEncoder(nn.Module):
    """Encode heatmaps into feature vectors."""
    
    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1),  # 64x64 -> 32x32
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1),  # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),  # 16x16 -> 8x8
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, out_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Heatmap tensor (B, 1, H, W)
        Returns:
            Feature vector (B, out_dim)
        """
        return self.encoder(x)


class AttentionMapEncoder(nn.Module):
    """Encode RL attention maps into feature vectors."""
    
    def __init__(self, out_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),  # 16x16 -> 16x16
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, out_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Attention map tensor (B, 1, h, w)
        Returns:
            Feature vector (B, out_dim)
        """
        return self.encoder(x)


class TransformerAggregator(nn.Module):
    """
    Transformer-based aggregator for combining model outputs.
    
    Each model's output becomes a "token" with features extracted from:
    - Heatmap (spatial predictions)
    - Classification logit (confidence)
    - RL attention map (focus area) [optional]
    
    A transformer learns cross-model attention to produce:
    - Combined classification prediction
    - Per-sample weights for heatmap combination
    """
    
    def __init__(
        self,
        num_models: int = 5,
        use_attention_maps: bool = True,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """
        Initialize transformer aggregator.
        
        Args:
            num_models: Number of models in the ensemble
            use_attention_maps: Whether to use RL attention maps as features
            hidden_dim: Hidden dimension for transformer
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_models = num_models
        self.use_attention_maps = use_attention_maps
        self.hidden_dim = hidden_dim
        
        # Encoders
        self.heatmap_encoder = HeatmapEncoder(out_dim=64)
        
        if use_attention_maps:
            self.attn_encoder = AttentionMapEncoder(out_dim=32)
            token_dim = 64 + 32 + 1  # heatmap + attention + cls
        else:
            self.attn_encoder = None
            token_dim = 64 + 1  # heatmap + cls
        
        # Project to hidden dim
        self.input_proj = nn.Linear(token_dim, hidden_dim)
        
        # Learnable model position embeddings (like model IDs)
        self.model_embeddings = nn.Embedding(num_models, hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output heads
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Per-sample heatmap weighting
        self.weight_head = nn.Linear(hidden_dim, num_models)
    
    def forward(
        self,
        heatmaps: torch.Tensor,
        cls_logits: torch.Tensor,
        attention_maps: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through transformer aggregator.
        
        Args:
            heatmaps: Stacked heatmaps (B, num_models, H, W)
            cls_logits: Stacked classification logits (B, num_models)
            attention_maps: Stacked attention maps (B, num_models, h, w) or None
            
        Returns:
            combined_cls: Combined classification logit (B, 1)
            combined_heatmap: Combined heatmap (B, 1, H, W)
            weights: Per-sample model weights (B, num_models)
        """
        B = heatmaps.shape[0]
        device = heatmaps.device
        
        # Debug: print shapes for troubleshooting (enabled for debugging)
        # print(f"[DEBUG] heatmaps shape: {heatmaps.shape}, cls_logits shape: {cls_logits.shape}")
        # if attention_maps is not None:
        #     print(f"[DEBUG] attention_maps shape: {attention_maps.shape}")
        # else:
        #     print(f"[DEBUG] attention_maps is None")
        
        # heatmaps are now LOGITS, convert to probs for feature extraction
        heatmap_probs = heatmaps.sigmoid()  # (B, num_models, H, W)
        
        # Encode each model's outputs into tokens
        tokens = []
        for i in range(self.num_models):
            # Get heatmap probs for model i (for feature extraction)
            # heatmap_probs is (B, num_models, H, W) -> need (B, 1, H, W) for Conv2d
            hm_i = heatmap_probs[:, i, :, :]  # (B, H, W)
            hm_i = hm_i.unsqueeze(1)  # (B, 1, H, W)
            hm_feat = self.heatmap_encoder(hm_i)  # (B, 64)
            
            # Classification logit for model i: (B,) -> (B, 1)
            cls_feat = cls_logits[:, i].unsqueeze(1)  # (B, 1)
            
            # Debug first iteration
            # if i == 0:
            #     print(f"[DEBUG] hm_feat shape: {hm_feat.shape}, cls_feat shape: {cls_feat.shape}")
            
            # Optionally encode attention map
            if self.use_attention_maps and attention_maps is not None:
                # attention_maps is (B, num_models, h, w) -> need (B, 1, h, w)
                attn_i = attention_maps[:, i, :, :]  # (B, h, w)
                attn_i = attn_i.unsqueeze(1)  # (B, 1, h, w)
                attn_feat = self.attn_encoder(attn_i)  # (B, 32)
                
                # if i == 0:
                #     print(f"[DEBUG] attn_feat shape: {attn_feat.shape}")
                
                # Verify all are 2D before concat
                assert hm_feat.dim() == 2, f"hm_feat should be 2D, got {hm_feat.shape}"
                assert attn_feat.dim() == 2, f"attn_feat should be 2D, got {attn_feat.shape}"
                assert cls_feat.dim() == 2, f"cls_feat should be 2D, got {cls_feat.shape}"
                
                token = torch.cat([hm_feat, attn_feat, cls_feat], dim=1)
            else:
                token = torch.cat([hm_feat, cls_feat], dim=1)
            
            tokens.append(token)
        
        # Stack tokens: (B, num_models, token_dim)
        tokens = torch.stack(tokens, dim=1)
        
        # Project to hidden dim
        tokens = self.input_proj(tokens)  # (B, num_models, hidden_dim)
        
        # Add model position embeddings
        model_ids = torch.arange(self.num_models, device=device)
        model_embs = self.model_embeddings(model_ids)  # (num_models, hidden_dim)
        tokens = tokens + model_embs.unsqueeze(0)  # (B, num_models, hidden_dim)
        
        # Cross-model attention via transformer
        refined = self.transformer(tokens)  # (B, num_models, hidden_dim)
        
        # Global aggregation (mean pooling over models)
        aggregated = refined.mean(dim=1)  # (B, hidden_dim)
        
        # Predictions
        combined_cls = self.cls_head(aggregated)  # (B, 1)
        
        # Per-sample model weights
        weights = F.softmax(self.weight_head(aggregated), dim=1)  # (B, num_models)
        
        # Compute weighted LOGIT combination (not probs!)
        # This provides better gradient flow
        # heatmaps: (B, num_models, H, W) -> add channel dim: (B, num_models, 1, H, W)
        heatmaps_expanded = heatmaps.unsqueeze(2)
        # weights: (B, num_models) -> (B, num_models, 1, 1, 1)
        weights_expanded = weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # Weighted sum of LOGITS: (B, 1, H, W)
        combined_heatmap_logits = (heatmaps_expanded * weights_expanded).sum(dim=1)
        
        return combined_cls, combined_heatmap_logits, weights


class TransformerEnsemble(nn.Module):
    """
    Full ensemble model that wraps base models with transformer aggregator.
    """
    
    def __init__(
        self,
        models: Dict[str, nn.Module],
        use_attention_maps: bool = True,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        """
        Initialize transformer ensemble.
        
        Args:
            models: Dictionary of base models
            use_attention_maps: Whether to use RL attention maps
            hidden_dim: Hidden dimension for aggregator
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
        """
        super().__init__()
        
        self.model_keys = list(models.keys())
        self.num_models = len(models)
        
        # Store models (frozen)
        self.models = nn.ModuleDict(models)
        for model in self.models.values():
            for param in model.parameters():
                param.requires_grad = False
        
        # Learnable aggregator
        self.aggregator = TransformerAggregator(
            num_models=self.num_models,
            use_attention_maps=use_attention_maps,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
        )
    
    def forward(self, data: Dict[str, Any], deterministic: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass through ensemble.
        
        Args:
            data: Input data batch
            deterministic: Whether to use deterministic policy
            
        Returns:
            Dictionary with ensemble outputs
        """
        # Collect outputs from all models
        heatmaps = []
        cls_logits = []
        attention_maps = []
        individual_outputs = {}
        
        with torch.no_grad():
            for key in self.model_keys:
                model = self.models[key]
                out = model(data, deterministic=deterministic)
                
                # IMPORTANT: Clone tensors to prevent reuse between models!
                # Some models may share internal buffers that get overwritten
                out_cloned = {}
                for k, v in out.items():
                    if isinstance(v, torch.Tensor):
                        out_cloned[k] = v.clone()
                    else:
                        out_cloned[k] = v
                individual_outputs[key] = out_cloned
                
                # Collect heatmaps - CLONE to prevent tensor reuse!
                if 'cancer_logits' in out:
                    heatmaps.append(out['cancer_logits'].clone())
                
                # Collect classification logits
                if 'image_level_classification_outputs' in out and out['image_level_classification_outputs'] is not None:
                    cls = out['image_level_classification_outputs']
                    
                    # Handle list of tensors - take first element if list
                    if isinstance(cls, list):
                        if len(cls) > 0:
                            cls = cls[0]  # Take first element, not stack
                        else:
                            cls = None
                    
                    if cls is not None:
                        # Ensure we have a (B,) tensor with one logit per sample
                        # cls might be (B,), (B, 1), (B, 2), etc.
                        if cls.dim() == 0:
                            # Scalar - expand to batch
                            cls = cls.unsqueeze(0)
                        elif cls.dim() >= 2:
                            # Multi-class: take positive class (index 1) or last column
                            if cls.shape[-1] == 2:
                                cls = cls[..., 1]  # Binary: take positive class
                            else:
                                cls = cls[..., 0]  # Single class
                        # Now cls should be (B,)
                        if cls.dim() > 1:
                            cls = cls.squeeze()  # Remove extra dims
                        cls_logits.append(cls)
                
                # Collect attention maps - ensure consistent shape and CLONE!
                if 'rl_attention_map' in out and out['rl_attention_map'] is not None:
                    attn = out['rl_attention_map'].clone()
                    # Ensure 3D shape (B, h, w) by removing channel dim if present
                    if attn.dim() == 4:
                        attn = attn.squeeze(1)  # (B, 1, h, w) -> (B, h, w)
                    attention_maps.append(attn)
        
        # Stack heatmaps - each is (B, 1, H, W)
        # Result: (B, num_models, 1, H, W) -> squeeze to (B, num_models, H, W)
        heatmaps_stacked = torch.stack(heatmaps, dim=1)  # (B, num_models, 1, H, W)
        if heatmaps_stacked.dim() == 5:
            heatmaps = heatmaps_stacked.squeeze(2)  # (B, num_models, H, W)
        else:
            heatmaps = heatmaps_stacked
        
        # Stack cls_logits - each should be (B,)
        cls_logits = torch.stack(cls_logits, dim=1)  # (B, num_models)
        
        # Stack attention maps if all models have them
        if attention_maps and len(attention_maps) == self.num_models:
            # Each attn is (B, h, w) -> stack to (B, num_models, h, w)
            attention_maps = torch.stack(attention_maps, dim=1)  # (B, num_models, h, w)
        else:
            attention_maps = None
        
        # Run aggregator - now returns LOGITS directly
        combined_cls, combined_heatmap_logits, weights = self.aggregator(
            heatmaps, cls_logits, attention_maps
        )
        
        return {
            'cancer_logits': combined_heatmap_logits,  # Already logits!
            'cancer_probs': combined_heatmap_logits.sigmoid(),
            'image_level_classification_outputs': combined_cls,
            'ensemble_weights': weights,
            'individual_outputs': individual_outputs,
        }
    
    def get_trainable_params(self):
        """Get only the trainable (aggregator) parameters."""
        return list(self.aggregator.parameters())


def create_transformer_ensemble(
    config_path: str = "configs/models.yaml",
    model_keys: Optional[List[str]] = None,
    use_attention_maps: bool = True,
    hidden_dim: int = 128,
    device: Optional[torch.device] = None,
) -> TransformerEnsemble:
    """
    Create a transformer ensemble from config.
    
    Args:
        config_path: Path to models.yaml
        model_keys: Which models to include
        use_attention_maps: Whether to use RL attention maps
        hidden_dim: Hidden dimension for aggregator
        device: Target device
        
    Returns:
        TransformerEnsemble instance
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load frozen models
    models = load_models_from_config(config_path, model_keys=model_keys, device=device, freeze=True)
    
    ensemble = TransformerEnsemble(
        models,
        use_attention_maps=use_attention_maps,
        hidden_dim=hidden_dim,
    )
    
    return ensemble.to(device)


if __name__ == "__main__":
    print("Testing Transformer Aggregator...")
    
    # Test aggregator alone
    agg = TransformerAggregator(num_models=3, use_attention_maps=True)
    print(f"Aggregator parameters: {sum(p.numel() for p in agg.parameters()):,}")
    
    # Test with dummy data
    B = 2
    heatmaps = torch.rand(B, 3, 64, 64)
    cls_logits = torch.rand(B, 3)
    attn_maps = torch.rand(B, 3, 16, 16)
    
    combined_cls, combined_heatmap, weights = agg(heatmaps, cls_logits, attn_maps)
    print(f"Combined cls shape: {combined_cls.shape}")
    print(f"Combined heatmap shape: {combined_heatmap.shape}")
    print(f"Weights shape: {weights.shape}")
    print(f"Weights sum: {weights.sum(dim=1)}")
    
    print("\nTransformer aggregator test passed!")
