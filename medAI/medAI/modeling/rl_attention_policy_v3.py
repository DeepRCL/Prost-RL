"""
RL Attention Policy V3 - Patch-Level Attention
Implements proper patch-level attention for MedSAM decoder conditioning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class PatchAttentionPolicy(nn.Module):
    """
    Policy that outputs patch-level attention and passes it correctly to decoder.
    
    Two modes:
    1. Discrete: Sample k patches from distribution
    2. Continuous: Use full attention map
    
    Args:
        feature_dim: Input feature dimension (256)
        hidden_dim: Hidden dimension (default: 256)
        num_patches: Number of patches to select in discrete mode (default: 4)
        image_size: Image size (default: 256)
        use_clinical_features: Use clinical features (default: True)
        use_prostate_mask_constraint: Constrain to prostate (default: True)
        discrete_mode: If True, sample k patches; if False, use continuous attention (default: True)
    """
    
    def __init__(
        self,
        feature_dim: int = 256,
        hidden_dim: int = 256,
        num_patches: int = 4,
        image_size: int = 256,
        use_clinical_features: bool = True,
        use_prostate_mask_constraint: bool = True,
        discrete_mode: bool = True,
        boundary_tolerance_patches: int = 1,  # Patches to dilate prostate mask (on 16x16 feature grid)
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_patches = num_patches
        self.image_size = image_size
        self.use_clinical_features = use_clinical_features
        self.use_prostate_mask_constraint = use_prostate_mask_constraint
        self.discrete_mode = discrete_mode
        self.boundary_tolerance_patches = boundary_tolerance_patches
        
        # Simplified feature processor (2 layers)
        self.feature_processor = nn.Sequential(
            nn.Conv2d(feature_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # Simplified attention head (2 layers)
        self.attention_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1),
        )
        
        # Clinical feature embedding (simplified)
        if use_clinical_features:
            self.clinical_embedder = nn.Sequential(
                nn.Linear(4, hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim // 2, hidden_dim),
            )
            self.clinical_gate = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid(),
            )
        
    def forward(
        self,
        image_features: torch.Tensor,
        clinical_features: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        prostate_mask: Optional[torch.Tensor] = None,
        action_indices: Optional[torch.Tensor] = None,
        return_action_indices: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Returns:
            attention_features: Patch-level features to condition decoder (B, C, H, W) or (B, k, C)
            log_probs: Log probabilities (B, k) or (B, H*W)
            attention_map: Attention heatmap (B, 1, H, W)
            value: None (value function is separate)
            action_indices: Selected patch indices if discrete mode (B, k) or None
        """
        B, C, H, W = image_features.shape
        
        # Process features
        features = self.feature_processor(image_features)  # B x hidden_dim x H x W
        
        # Clinical modulation
        if self.use_clinical_features and clinical_features is not None:
            clinical_emb = self.clinical_embedder(clinical_features)  # B x hidden_dim
            gate = self.clinical_gate(clinical_emb)  # B x hidden_dim
            gate = gate[:, :, None, None]  # B x hidden_dim x 1 x 1
            features = features * gate
        
        # Generate attention map
        attention_map = self.attention_head(features)  # B x 1 x H x W
        attention_logits = attention_map.view(B, -1)  # B x (H*W)
        
        # Apply prostate mask constraint with boundary tolerance
        prostate_mask_dilated = None
        if self.use_prostate_mask_constraint and prostate_mask is not None:
            prostate_mask_resized = F.interpolate(
                prostate_mask.float(),
                size=(H, W),
                mode='nearest'
            )
            
            # Apply dilation to allow some boundary tolerance (at patch level)
            # This accounts for imperfect prostate segmentation
            # Each patch on the 16x16 feature grid = 16x16 pixels in original image
            if self.boundary_tolerance_patches > 0:
                # Create dilation kernel (size in patches on feature grid)
                kernel_size = 2 * self.boundary_tolerance_patches + 1
                dilation_kernel = torch.ones(1, 1, kernel_size, kernel_size, device=prostate_mask.device)
                # Dilate the mask on the feature grid
                prostate_mask_dilated = F.conv2d(
                    prostate_mask_resized, 
                    dilation_kernel, 
                    padding=self.boundary_tolerance_patches
                )
                prostate_mask_dilated = (prostate_mask_dilated > 0).float()
            else:
                prostate_mask_dilated = prostate_mask_resized
            
            mask_flat = prostate_mask_dilated.view(B, -1)
            valid_mask_per_sample = (mask_flat > 0.5).any(dim=1, keepdim=True)
            mask_to_apply = (mask_flat < 0.5) & valid_mask_per_sample
            attention_logits = attention_logits.masked_fill(mask_to_apply, float('-inf'))
        
        # Clamp for numerical stability
        attention_logits = torch.clamp(attention_logits, min=-50.0, max=50.0)
        
        if self.discrete_mode:
            # Discrete mode: Sample k patches
            return self._discrete_forward(
                features, attention_logits, attention_map, B, H, W,
                deterministic, action_indices, return_action_indices
            )
        else:
            # Continuous mode: Use full attention map with mask applied
            return self._continuous_forward(
                features, attention_logits, attention_map, B, H, W,
                prostate_mask_dilated
            )
    
    def _discrete_forward(
        self,
        features: torch.Tensor,
        attention_logits: torch.Tensor,
        attention_map: torch.Tensor,
        B: int, H: int, W: int,
        deterministic: bool,
        action_indices: Optional[torch.Tensor],
        return_action_indices: bool,
    ):
        """Discrete patch selection mode."""
        
        if action_indices is not None:
            # Evaluating fixed actions
            coords_flat = action_indices.to(device=attention_logits.device, dtype=torch.long)
            selected_log_probs = self._compute_log_probs_for_indices(attention_logits, coords_flat)
        elif deterministic:
            # Select top-k patches
            coords_flat, selected_log_probs = self._select_topk_patches(attention_logits, B, H, W)
        else:
            # Sample k patches
            coords_flat, selected_log_probs = self._sample_patches(attention_logits, B)
        
        # Extract features from selected patches
        coords_y = coords_flat // W
        coords_x = coords_flat % W
        
        # Gather patch features (B, k, C)
        patch_features_list = []
        for i in range(B):
            patch_feats = features[i, :, coords_y[i], coords_x[i]].T  # k x C
            patch_features_list.append(patch_feats)
        patch_features = torch.stack(patch_features_list, dim=0)  # B x k x C
        
        action_indices_out = coords_flat if return_action_indices else None
        
        return patch_features, selected_log_probs, attention_map, None, action_indices_out
    
    def _continuous_forward(
        self,
        features: torch.Tensor,
        attention_logits: torch.Tensor,
        attention_map: torch.Tensor,
        B: int, H: int, W: int,
        prostate_mask_dilated: torch.Tensor = None,
    ):
        """Continuous attention mode."""
        
        # Compute attention weights
        attention_probs = F.softmax(attention_logits, dim=1).view(B, 1, H, W)
        
        # Weighted features (B, C, H, W)
        weighted_features = features * attention_probs
        
        # Log probs for policy gradient (entropy over distribution)
        log_probs = F.log_softmax(attention_logits, dim=1)  # B x (H*W)
        entropy = -(attention_probs.view(B, -1) * log_probs).sum(dim=1)  # B
        policy_log_prob = -entropy  # Use negative entropy as "log prob" for REINFORCE
        
        # For visualization: return the masked attention map (attention_probs) 
        # instead of raw attention_map, so we don't show misleading values outside prostate
        # The attention_probs already has zeros outside prostate due to softmax(-inf) = 0
        # We use sigmoid for visualization since it's bounded [0, 1]
        masked_attention_map = attention_probs  # This is already masked via softmax
        
        # Alternatively, create a visualization-friendly map by applying mask to sigmoid of raw logits
        viz_attention_map = torch.sigmoid(attention_map)  # B x 1 x H x W
        if prostate_mask_dilated is not None:
            # Zero out regions outside the dilated prostate mask
            viz_attention_map = viz_attention_map * prostate_mask_dilated
        
        return weighted_features, policy_log_prob.unsqueeze(1), viz_attention_map, None, None
    
    def _compute_log_probs_for_indices(self, logits: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Compute log probs for given indices (without replacement)."""
        B, num_locs = logits.shape
        k = indices.shape[1]
        
        logits_work = torch.where(
            logits == float("-inf"),
            torch.full_like(logits, -100.0),
            logits,
        )
        probs_work = F.log_softmax(logits_work, dim=1).exp()
        probs_work = torch.clamp(probs_work, min=1e-8)
        probs_work = probs_work / probs_work.sum(dim=1, keepdim=True)
        
        logps = []
        for t in range(k):
            dist = torch.distributions.Categorical(probs=probs_work)
            idx_t = indices[:, t]
            logp_t = dist.log_prob(idx_t)
            logps.append(logp_t)
            probs_work = probs_work.scatter(1, idx_t.unsqueeze(1), 1e-8)
            probs_work = probs_work / probs_work.sum(dim=1, keepdim=True)
        
        return torch.stack(logps, dim=1)
    
    def _select_topk_patches(self, logits: torch.Tensor, B: int, H: int, W: int):
        """Select top-k patches deterministically."""
        coords_flat_list = []
        log_probs_list = []
        
        for b in range(B):
            valid_positions = logits[b] != float('-inf')
            num_valid = valid_positions.sum().item()
            
            if num_valid == 0:
                selected = torch.randperm(logits.shape[1], device=logits.device)[:self.num_patches]
            elif num_valid < self.num_patches:
                valid_idx = torch.where(valid_positions)[0]
                extra = self.num_patches - num_valid
                extra_idx = valid_idx[torch.randint(0, num_valid, (extra,), device=valid_idx.device)]
                selected = torch.cat([valid_idx, extra_idx])
            else:
                masked_logits = logits[b].clone()
                masked_logits[~valid_positions] = -1e9
                selected = torch.topk(masked_logits, k=self.num_patches, dim=0).indices
            
            coords_flat_list.append(selected)
            
            safe_logits = torch.where(
                logits[b] == float('-inf'),
                torch.full_like(logits[b], -100.0),
                logits[b]
            )
            log_probs_b = F.log_softmax(safe_logits, dim=0)
            log_probs_list.append(log_probs_b[selected])
        
        return torch.stack(coords_flat_list, dim=0), torch.stack(log_probs_list, dim=0)
    
    def _sample_patches(self, logits: torch.Tensor, B: int):
        """Sample k patches without replacement."""
        log_probs_all = F.log_softmax(logits, dim=1)
        attention_probs = log_probs_all.exp().clone()
        
        invalid_rows = torch.isnan(attention_probs).any(dim=1) | (attention_probs.sum(dim=1) < 1e-6)
        if invalid_rows.any():
            uniform_probs = torch.ones_like(attention_probs[0]) / attention_probs.shape[1]
            attention_probs[invalid_rows] = uniform_probs
        
        sampled_indices_list = []
        log_probs_list = []
        
        for i in range(self.num_patches):
            attention_probs = torch.clamp(attention_probs, min=1e-8)
            attention_probs = attention_probs / attention_probs.sum(dim=1, keepdim=True)
            
            dist = torch.distributions.Categorical(probs=attention_probs)
            sampled_idx = dist.sample()
            sampled_log_prob = dist.log_prob(sampled_idx)
            
            sampled_indices_list.append(sampled_idx)
            log_probs_list.append(sampled_log_prob)
            
            attention_probs = attention_probs.scatter(1, sampled_idx.unsqueeze(1), 1e-8)
            attention_probs = attention_probs / attention_probs.sum(dim=1, keepdim=True)
        
        coords_flat = torch.stack(sampled_indices_list, dim=1)
        selected_log_probs = torch.stack(log_probs_list, dim=1)
        
        return coords_flat, selected_log_probs


class ValueNetwork(nn.Module):
    """
    Separate value network for PPO that takes both state and action.
    
    Stays lightweight even when policy is complex.
    
    Args:
        feature_dim: Input image feature dimension (256) - used for state encoder
        hidden_dim: Policy hidden dimension (256 or 512) - used for action encoder
                    This is the dimension of features output by the policy
        action_type: 'discrete' or 'continuous'
        num_patches: Number of patches if discrete
        value_hidden_dim: Fixed hidden dim for value network (default: 256)
    
    Note:
        - State encoder uses feature_dim (256) - image features from encoder
        - Action encoder uses hidden_dim (policy_hidden_dim) - features from policy output
    """
    
    def __init__(
        self,
        feature_dim: int = 256,
        hidden_dim: int = 256,
        action_type: str = 'discrete',
        num_patches: int = 4,
        use_clinical_features: bool = True,
        value_hidden_dim: int = 256,  # Keep value network simple
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.action_type = action_type
        self.num_patches = num_patches
        self.use_clinical_features = use_clinical_features
        self.value_hidden_dim = value_hidden_dim
        
        # State encoder (fixed size, doesn't scale with policy hidden_dim)
        self.state_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(feature_dim * 16, value_hidden_dim),
            nn.ReLU(),
        )
        
        # Action encoder (fixed size)
        # NOTE: action_features have dimension hidden_dim (policy_hidden_dim), not feature_dim!
        if action_type == 'discrete':
            # For discrete: encode selected patch features
            # Input: hidden_dim * num_patches (policy outputs hidden_dim-dim features)
            self.action_encoder = nn.Sequential(
                nn.Linear(hidden_dim * num_patches, value_hidden_dim),
                nn.ReLU(),
            )
        else:
            # For continuous: encode attention-weighted features
            # Input channels: hidden_dim (policy outputs hidden_dim-dim features)
            self.action_encoder = nn.Sequential(
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Linear(hidden_dim * 16, value_hidden_dim),
                nn.ReLU(),
            )
        
        # Clinical encoder (fixed size)
        if use_clinical_features:
            self.clinical_encoder = nn.Sequential(
                nn.Linear(4, value_hidden_dim // 2),
                nn.ReLU(),
            )
            fusion_dim = value_hidden_dim * 2 + value_hidden_dim // 2
        else:
            fusion_dim = value_hidden_dim * 2
        
        # Value head (fixed size)
        self.value_head = nn.Sequential(
            nn.Linear(fusion_dim, value_hidden_dim),
            nn.ReLU(),
            nn.Linear(value_hidden_dim, 1),
        )
    
    def forward(
        self,
        image_features: torch.Tensor,
        action_features: torch.Tensor,
        clinical_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute value estimate.
        
        Args:
            image_features: Image features (B, C, H, W)
            action_features: Action features (B, k, C) for discrete or (B, C, H, W) for continuous
            clinical_features: Clinical features (B, 4)
        
        Returns:
            value: Value estimate (B,)
        """
        # Encode state
        state_emb = self.state_encoder(image_features)  # B x hidden_dim
        
        # Encode action
        if self.action_type == 'discrete':
            # Flatten patch features
            B, k, C = action_features.shape
            action_flat = action_features.view(B, k * C)
            action_emb = self.action_encoder(action_flat)  # B x hidden_dim
        else:
            # Pool weighted features
            action_emb = self.action_encoder(action_features)  # B x hidden_dim
        
        # Fuse
        if self.use_clinical_features and clinical_features is not None:
            clinical_emb = self.clinical_encoder(clinical_features)
            fused = torch.cat([state_emb, action_emb, clinical_emb], dim=1)
        else:
            fused = torch.cat([state_emb, action_emb], dim=1)
        
        # Value
        value = self.value_head(fused).squeeze(-1)  # B
        
        return value
