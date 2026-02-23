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
    
    Three modes controlled by ``attention_mode``:
    1. ``'discrete'``: Sample k patches from a categorical distribution without replacement.
    2. ``'continuous'``: Use a softmax-weighted sum over the full 16x16 feature map
       (fully differentiable, no stochasticity).
    3. ``'bernoulli'``: Sigmoid probabilities over the 16x16 grid → sample a **binary mask**
       from independent Bernoulli distributions.  Each patch is selected/deselected
       independently, so different rollouts produce structurally different masks, giving
       GRPO real within-group reward variance.  At deterministic inference the binary mask
       is replaced by the raw sigmoid probabilities (smooth continuous gate).
    
    Args:
        feature_dim: Input feature dimension (256)
        hidden_dim: Hidden dimension (default: 256)
        num_patches: Number of patches to select in discrete mode (default: 4)
        image_size: Image size (default: 256)
        use_clinical_features: Use clinical features (default: True)
        use_prostate_mask_constraint: Constrain to prostate (default: True)
        attention_mode: One of 'discrete', 'continuous', 'bernoulli' (default: 'discrete').
        discrete_mode: *Deprecated* bool alias kept for backward compat.  If
            ``attention_mode`` is explicitly passed, ``discrete_mode`` is ignored.
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
        attention_mode: Optional[str] = None,  # 'discrete', 'continuous', or 'bernoulli'
        bernoulli_pool_stride: int = 1,  # >1 uses coarser spatial grid for Bernoulli (reduces action space)
        continuous_noise_scale: float = 0.0,  # >0 adds Gaussian noise to logits during training rollouts
                                               # gives genuine GRPO diversity for continuous mode
                                               # default 0.0 = backward-compatible (old behaviour)
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_patches = num_patches
        self.image_size = image_size
        self.use_clinical_features = use_clinical_features
        self.use_prostate_mask_constraint = use_prostate_mask_constraint
        self.boundary_tolerance_patches = boundary_tolerance_patches
        # bernoulli_pool_stride=2 → 8×8=64 variables instead of 16×16=256
        # reduces gradient dilution and makes spatial credit assignment easier
        self.bernoulli_pool_stride = bernoulli_pool_stride
        self.continuous_noise_scale = continuous_noise_scale

        # Resolve attention_mode (new API) vs discrete_mode (deprecated bool API)
        if attention_mode is not None:
            if attention_mode not in ('discrete', 'continuous', 'bernoulli'):
                raise ValueError(
                    f"attention_mode must be 'discrete', 'continuous', or 'bernoulli', "
                    f"got '{attention_mode}'"
                )
            self.attention_mode = attention_mode
        else:
            # Backward-compat: map bool discrete_mode to string
            self.attention_mode = 'discrete' if discrete_mode else 'continuous'
        
        # Keep legacy attribute so old code that reads .discrete_mode still works
        self.discrete_mode = (self.attention_mode == 'discrete')
        
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
        
        if self.attention_mode == 'discrete':
            # Discrete mode: Sample k patches without replacement
            return self._discrete_forward(
                features, attention_logits, attention_map, B, H, W,
                deterministic, action_indices, return_action_indices
            )
        elif self.attention_mode == 'bernoulli':
            # Bernoulli mode: independent binary mask per patch
            # Optionally pool to coarser grid to reduce action-space and
            # increase per-variable gradient strength (bernoulli_pool_stride > 1)
            if self.bernoulli_pool_stride > 1:
                s = self.bernoulli_pool_stride
                coarse_logits = F.avg_pool2d(
                    attention_logits.view(B, 1, H, W),
                    kernel_size=s, stride=s
                ).view(B, -1)          # B x (H/s * W/s)
                coarse_mask_dilated = (
                    F.max_pool2d(
                        prostate_mask_dilated if prostate_mask_dilated is not None
                        else torch.ones(B, 1, H, W, device=features.device),
                        kernel_size=s, stride=s
                    ) if prostate_mask_dilated is not None else None
                )
                Hc, Wc = H // s, W // s
                return self._bernoulli_forward(
                    features, coarse_logits, attention_map, B, H, W,
                    deterministic, action_indices, return_action_indices,
                    prostate_mask_dilated,
                    coarse_H=Hc, coarse_W=Wc,
                    coarse_mask_dilated=coarse_mask_dilated,
                )
            else:
                return self._bernoulli_forward(
                    features, attention_logits, attention_map, B, H, W,
                    deterministic, action_indices, return_action_indices,
                    prostate_mask_dilated
                )
        else:
            # Continuous mode: softmax-weighted sum
            return self._continuous_forward(
                features, attention_logits, attention_map, B, H, W,
                prostate_mask_dilated, deterministic=deterministic,
                action_indices=action_indices,
                return_action_indices=return_action_indices,
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
        deterministic: bool = False,
        action_indices: torch.Tensor = None,
        return_action_indices: bool = False,
    ):
        """Continuous attention mode.

        When self.continuous_noise_scale > 0 and not deterministic, Gaussian noise
        is added to the logits before softmax, creating genuine rollout diversity so
        GRPO advantages are non-trivial.

        Action replay for PPO/GRPO update epochs:
          During rollout: fresh noise is sampled and saved as action_indices.
          During replay: the saved noise is re-used, and log_prob is recomputed
          under the CURRENT policy parameters, enabling proper policy gradient.

        log_prob uses the categorical cross-entropy between the clean softmax
        distribution p(·|θ) and the noisy softmax distribution q(·|θ,ε):
          log_prob = Σ_i q_i * log(p_i)
        This quantity depends on θ (through p_i = softmax(logits_i)) so the
        GRPO ratio exp(log_p_new - log_p_old) is a valid importance weight.

        noise_scale=0 (old/default): -entropy(softmax) — kept for backward compat.
        """
        sigma = self.continuous_noise_scale
        use_noise = (sigma > 0.0) and (not deterministic) and self.training

        if use_noise:
            if action_indices is not None:
                # PPO update epoch: replay the saved noise
                noise = action_indices
            else:
                # Rollout: sample fresh noise
                noise = torch.randn_like(attention_logits) * sigma

            noisy_logits = attention_logits + noise
            attention_probs = F.softmax(noisy_logits, dim=1).view(B, 1, H, W)

            # log_prob that depends on policy parameters θ:
            # Use the categorical log-prob: Σ_i q_i * log(p_i)
            # where p_i = softmax(logits_i) is the clean policy distribution
            #       q_i = softmax(logits_i + noise_i) is the noisy (sampled) distribution
            # This is equivalent to -cross_entropy(q, p) and varies with θ.
            clean_log_probs = F.log_softmax(attention_logits, dim=1)  # log p_i(θ)
            noisy_probs = attention_probs.view(B, -1)  # q_i — detached from graph via softmax(logits+noise)
            # Σ_i q_i * log(p_i):  measures how well the clean policy explains the noisy action
            policy_log_prob = (noisy_probs.detach() * clean_log_probs).sum(dim=1)  # B

            action_indices_out = noise if return_action_indices else None
        else:
            attention_probs = F.softmax(attention_logits, dim=1).view(B, 1, H, W)
            # Legacy surrogate: -entropy used as log_prob.
            # Backward-compatible with all models trained with continuous_noise_scale=0.
            log_probs_cat = F.log_softmax(attention_logits, dim=1)
            entropy = -(attention_probs.view(B, -1) * log_probs_cat).sum(dim=1)
            policy_log_prob = -entropy
            action_indices_out = None

        weighted_features = features * attention_probs

        viz_attention_map = torch.sigmoid(attention_map)
        if prostate_mask_dilated is not None:
            viz_attention_map = viz_attention_map * prostate_mask_dilated

        return weighted_features, policy_log_prob.unsqueeze(1), viz_attention_map, None, action_indices_out

    def _bernoulli_forward(
        self,
        features: torch.Tensor,
        attention_logits: torch.Tensor,  # B x (Hc*Wc) — may be coarser than H×W
        attention_map: torch.Tensor,
        B: int, H: int, W: int,          # full feature-grid size
        deterministic: bool,
        action_indices: Optional[torch.Tensor],
        return_action_indices: bool,
        prostate_mask_dilated: Optional[torch.Tensor] = None,
        coarse_H: Optional[int] = None,  # set when bernoulli_pool_stride > 1
        coarse_W: Optional[int] = None,
        coarse_mask_dilated: Optional[torch.Tensor] = None,
    ):
        """
        Bernoulli sampling over the 16x16 patch grid.

        Each patch is independently included/excluded via a Bernoulli draw, breaking
        the zero-sum competition of softmax (which makes continuous rollouts nearly
        identical and kills GRPO's advantage signal).

        Training:  sample a binary mask M ~ Bernoulli(sigmoid(logits))
        Inference: use sigmoid(logits) directly as a smooth continuous gate
                   (no sampling, fully deterministic)
        Replay:    when ``action_indices`` (the stored mask) is passed, recompute
                   log_prob of that exact mask under the *current* policy — needed
                   for multi-epoch GRPO update steps.

        log_prob is computed as the MEAN (not sum) of per-patch Bernoulli log-probs.
        Summing 256 terms would produce extremely large magnitudes and destabilise
        GRPO normalisation; mean keeps it in the same numerical range as discrete mode.

        Returns:
            weighted_features : (B, hidden_dim, H, W)  features gated by the mask
            log_prob           : (B, 1)                 mean Bernoulli log-prob
            viz_map            : (B, 1, H, W)           sigmoid probs (for visualisation)
            None               : value (not used here)
            action_indices_out : (B, H*W) float binary mask, or None
        """
        # Determine the actual spatial grid used for Bernoulli sampling
        # When bernoulli_pool_stride > 1, attention_logits is already on the coarse grid.
        Hb = coarse_H if coarse_H is not None else H
        Wb = coarse_W if coarse_W is not None else W
        mask_dilated_for_viz = coarse_mask_dilated if coarse_mask_dilated is not None else prostate_mask_dilated

        # Sigmoid probabilities — independent per patch, no competition
        safe_logits = attention_logits.clone()
        safe_logits[safe_logits == float('-inf')] = -50.0  # sigmoid(-50) ≈ 0
        probs_flat = torch.sigmoid(safe_logits)  # B x (Hb*Wb)

        dist = torch.distributions.Bernoulli(probs=probs_flat)

        if action_indices is not None:
            mask_flat = action_indices.float().to(probs_flat.device)
            deterministic_gate = False
        elif deterministic:
            # Deterministic inference should use smooth probabilities directly,
            # not a hard 0.5 threshold that introduces patchy artifacts.
            mask_flat = torch.ones_like(probs_flat)
            deterministic_gate = True
        else:
            mask_flat = dist.sample()
            deterministic_gate = False

        # sqrt-normalised sum log-prob: restores per-logit gradient strength
        # compared to plain mean (which dilutes by N_patches), while avoiding
        # the astronomically large log-ratio magnitudes of plain sum.
        N_b = Hb * Wb
        per_patch_lp = dist.log_prob(mask_flat)                        # B x (Hb*Wb)
        log_prob = per_patch_lp.sum(dim=1, keepdim=True) / N_b ** 0.5  # B x 1

        # Upsample coarse mask/probabilities to full feature grid for decoder modulation
        mask_2d_coarse = mask_flat.view(B, 1, Hb, Wb)
        probs_2d_coarse = probs_flat.view(B, 1, Hb, Wb)

        if Hb != H or Wb != W:
            mask_2d = F.interpolate(mask_2d_coarse, size=(H, W), mode='nearest')
            probs_2d = F.interpolate(probs_2d_coarse, size=(H, W), mode='bilinear', align_corners=False)
        else:
            mask_2d = mask_2d_coarse
            probs_2d = probs_2d_coarse

        # Decoder gating:
        # - training / PPO replay: stochastic Bernoulli gate (probs * mask)
        # - deterministic inference: smooth expected gate (probs)
        if deterministic_gate:
            weighted_features = features * probs_2d
        else:
            weighted_features = features * probs_2d * mask_2d    # B x hidden_dim x H x W

        # Visualisation map at full resolution
        viz_map = probs_2d.clone()
        if prostate_mask_dilated is not None:
            viz_map = viz_map * prostate_mask_dilated

        action_indices_out = mask_flat if return_action_indices else None

        return weighted_features, log_prob, viz_map, None, action_indices_out

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
