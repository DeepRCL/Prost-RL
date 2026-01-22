"""
GDPO - Group reward-Decoupled Normalization Policy Optimization

Based on: https://arxiv.org/abs/2601.05242

Key insight: Instead of summing rewards first and then normalizing,
GDPO normalizes each reward signal SEPARATELY across the group, then sums
the normalized advantages. This prevents reward advantages collapse in
multi-reward RL training.

For ProstNFound-RL, we have two main reward signals:
1. Classification reward: How well the model classifies cancer vs benign
2. Attention/Sparsity reward: How well attention corresponds to involvement

GDPO ensures both signals contribute meaningfully to the advantages,
preventing one from dominating or canceling the other.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import logging


class GDPO:
    """
    Group reward-Decoupled Normalization Policy Optimization.
    
    Unlike standard GRPO which normalizes the summed reward:
        advantage = (sum_of_rewards - mean) / std
    
    GDPO normalizes each reward separately and then sums the advantages:
        advantage_i = (reward_i - mean_i) / std_i
        final_advantage = sum(advantage_i * weight_i)
    
    This preserves the relative differences in each reward signal,
    enabling more faithful preference optimization.
    
    Args:
        clip_eps: PPO clipping epsilon (default: 0.2)
        entropy_coef: Entropy coefficient (default: 0.01)
        max_grad_norm: Max gradient norm (default: 0.5)
        kl_coef: KL penalty coefficient (default: 0.01)
        normalize_advantages: Normalize combined advantages at the end (default: True)
        num_samples_per_image: Samples per image for within-image comparison (default: 4)
        use_value_function: Use value function for PPO (default: False)
        value_coef: Value loss coefficient (default: 0.5)
        reward_weights: Optional weights for each reward component (default: [1.0, 1.0])
        strict_mode: If True, raise exceptions on fallbacks instead of logging (default: True)
    """
    
    def __init__(
        self,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        kl_coef: float = 0.01,
        normalize_advantages: bool = True,
        num_samples_per_image: int = 4,
        use_value_function: bool = False,
        value_coef: float = 0.5,
        reward_weights: Optional[List[float]] = None,
        strict_mode: bool = True,  # Raise exceptions on fallbacks
    ):
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.kl_coef = kl_coef
        self.normalize_advantages = normalize_advantages
        self.num_samples_per_image = num_samples_per_image
        self.use_value_function = use_value_function
        self.value_coef = value_coef
        self.reward_weights = reward_weights if reward_weights is not None else [1.0, 1.0]
        self.strict_mode = strict_mode
        
        # Track if GDPO is actually being used (for verification)
        self._gdpo_used_count = 0
        self._fallback_count = 0
        
        logging.info(
            f"=" * 60 + "\n"
            f"GDPO INITIALIZED (Group reward-Decoupled normalization Policy Optimization)\n"
            f"  clip_eps={clip_eps}, entropy_coef={entropy_coef}, kl_coef={kl_coef}\n"
            f"  num_samples_per_image={num_samples_per_image}\n"
            f"  reward_weights={self.reward_weights}\n"
            f"  strict_mode={strict_mode} (will {'RAISE ERRORS' if strict_mode else 'log warnings'} on fallback)\n"
            f"=" * 60
        )
    
    def normalize_reward_within_image(
        self,
        reward: torch.Tensor,
        num_samples_per_image: int,
    ) -> torch.Tensor:
        """
        Normalize a single reward component within each image group.
        
        This is the key to GDPO: each reward signal is normalized independently
        within its group before being combined with other rewards.
        
        Args:
            reward: Single reward component (B * num_samples,)
            num_samples_per_image: Number of samples per image
            
        Returns:
            advantage: Normalized advantage for this reward (B * num_samples,)
        """
        total_samples = reward.shape[0]
        num_images = total_samples // num_samples_per_image
        
        if total_samples != num_images * num_samples_per_image:
            error_msg = (
                f"[GDPO ERROR] Sample count mismatch!\n"
                f"  Total samples: {total_samples}\n"
                f"  num_samples_per_image: {num_samples_per_image}\n"
                f"  Expected: {num_images * num_samples_per_image} (divisible)\n"
                f"  This indicates a bug in batch replication or reward computation!\n"
                f"  Falling back to simple batch normalization (NOT real GDPO!)"
            )
            logging.error(error_msg)
            self._fallback_count += 1
            if self.strict_mode:
                raise ValueError(error_msg)
            return (reward - reward.mean()) / (reward.std() + 1e-4)
        
        # Reshape to (num_images, num_samples_per_image)
        reward_grouped = reward.view(num_images, num_samples_per_image)
        
        # Compute mean and std within each image
        group_mean = reward_grouped.mean(dim=1, keepdim=True)
        group_std = reward_grouped.std(dim=1, keepdim=True) + 1e-4
        
        # Normalize within each group
        advantage_grouped = (reward_grouped - group_mean) / group_std
        
        # Flatten back
        advantage = advantage_grouped.view(-1)
        
        return advantage
    
    def compute_gdpo_advantages(
        self,
        reward_components: List[torch.Tensor],
        num_samples_per_image: Optional[int] = None,
        values: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute GDPO advantages by normalizing each reward component separately.
        
        This is the core GDPO algorithm:
        1. For each reward component, normalize within image groups
        2. Sum the normalized advantages with optional weights
        3. Optionally normalize the combined advantage (batch-level)
        
        Args:
            reward_components: List of reward tensors, each of shape (B * num_samples,)
            num_samples_per_image: Number of samples per image
            values: Value estimates (B * num_samples,) if PPO mode
            
        Returns:
            advantages: Combined GDPO advantages (B * num_samples,)
            returns: Returns for value loss (B * num_samples,) or None
        """
        num_samples = num_samples_per_image if num_samples_per_image is not None else self.num_samples_per_image
        device = reward_components[0].device
        
        # Step 1: Normalize each reward component separately
        normalized_advantages = []
        for i, reward in enumerate(reward_components):
            if num_samples > 1:
                adv = self.normalize_reward_within_image(reward, num_samples)
            else:
                # Single sample - just center the reward
                adv = reward - reward.mean()
            normalized_advantages.append(adv)
        
        # Step 2: Combine normalized advantages with weights
        # Extend reward_weights if needed
        weights = self.reward_weights
        if len(weights) < len(normalized_advantages):
            weights = weights + [1.0] * (len(normalized_advantages) - len(weights))
        
        combined_advantage = torch.zeros_like(normalized_advantages[0])
        for i, adv in enumerate(normalized_advantages):
            weight = weights[i] if i < len(weights) else 1.0
            combined_advantage = combined_advantage + weight * adv
        
        # Step 3: Optionally normalize the combined advantage (batch normalization)
        # This is the final "whitening" step in GDPO
        if self.normalize_advantages:
            adv_mean = combined_advantage.mean()
            adv_std = combined_advantage.std() + 1e-4
            combined_advantage = (combined_advantage - adv_mean) / adv_std
        
        # Handle value function for PPO mode
        returns = None
        if self.use_value_function and values is not None:
            # Sum rewards for returns (value target)
            total_rewards = sum(reward_components)
            returns = total_rewards.clone()
        
        return combined_advantage, returns
    
    def compute_policy_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute PPO-style clipped policy loss.
        
        Args:
            log_probs: Current log probabilities (B, k) or (B,)
            old_log_probs: Old log probabilities (B, k) or (B,)
            advantages: Advantages (B,)
            
        Returns:
            loss: Policy loss
            info: Logging info
        """
        # Sum log probs if multiple actions
        if log_probs.ndim > 1:
            log_probs_sum = log_probs.sum(dim=1)
            old_log_probs_sum = old_log_probs.sum(dim=1)
        else:
            log_probs_sum = log_probs
            old_log_probs_sum = old_log_probs
        
        # Clamp log ratio for numerical stability
        log_ratio = log_probs_sum - old_log_probs_sum
        log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
        ratio = torch.exp(log_ratio)
        
        # Clipped surrogate
        advantages = advantages.detach()
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # KL penalty (sampled)
        if log_probs.ndim > 1:
            delta = old_log_probs - log_probs
        else:
            delta = old_log_probs_sum - log_probs_sum
            delta = delta.unsqueeze(1)
        
        delta = torch.clamp(delta, min=-20.0, max=20.0)
        per_point_kl = torch.exp(delta) - delta - 1
        kl = per_point_kl.sum(dim=1).mean() if delta.ndim > 1 else per_point_kl.mean()
        
        # Total loss
        total_loss = policy_loss + self.kl_coef * kl
        
        # Check for NaN
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logging.warning(
                f"NaN/Inf in GDPO loss! ratio_range=[{ratio.min().item():.2f}, {ratio.max().item():.2f}]"
            )
            total_loss = torch.zeros_like(total_loss)
        
        info = {
            'policy_loss': policy_loss.item() if not torch.isnan(policy_loss) else 0.0,
            'kl': kl.item() if not torch.isnan(kl) else 0.0,
            'ratio_mean': ratio.mean().item(),
            'ratio_min': ratio.min().item(),
            'ratio_max': ratio.max().item(),
            'advantages_mean': advantages.mean().item(),
            'advantages_std': advantages.std().item(),
        }
        
        return total_loss, info
    
    def compute_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        num_samples_per_image: Optional[int] = None,
        values: Optional[torch.Tensor] = None,
        reward_components: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total GDPO loss.
        
        IMPORTANT: For GDPO to work correctly, you should pass reward_components
        (list of separate reward signals) instead of the summed rewards.
        
        If reward_components is not provided, falls back to standard GRPO
        using the single summed reward.
        
        Args:
            log_probs: Current log probs (B * num_samples, k) or (B * num_samples,)
            old_log_probs: Old log probs (B * num_samples, k) or (B * num_samples,)
            rewards: Summed rewards (B * num_samples,) - used if reward_components is None
            num_samples_per_image: Number of samples per image
            values: Value estimates (B * num_samples,) if PPO mode
            reward_components: List of separate reward tensors for GDPO
            
        Returns:
            total_loss: Loss
            info: Logging info
        """
        num_samples = num_samples_per_image if num_samples_per_image is not None else self.num_samples_per_image
        
        # Compute advantages with GDPO (decoupled normalization)
        if reward_components is not None and len(reward_components) > 0:
            # True GDPO: normalize each reward component separately
            self._gdpo_used_count += 1
            
            # Log GDPO usage only on first call for verification
            if self._gdpo_used_count == 1:
                logging.info(
                    f"[GDPO] First call - Using decoupled normalization with {len(reward_components)} reward components"
                )
            
            advantages, returns = self.compute_gdpo_advantages(
                reward_components,
                num_samples_per_image=num_samples,
                values=values,
            )
        else:
            # This should NOT happen in GDPO mode!
            error_msg = (
                f"[GDPO CRITICAL ERROR] No reward_components provided!\n"
                f"  reward_components: {reward_components}\n"
                f"  This means GDPO is NOT working - falling back to standard GRPO!\n"
                f"  Check that your reward_computer has compute_reward_components() method\n"
                f"  and that train_rl.py is calling it correctly for GDPO mode."
            )
            logging.error(error_msg)
            self._fallback_count += 1
            
            if self.strict_mode:
                raise ValueError(error_msg)
            
            advantages, returns = self._compute_standard_advantages(
                rewards,
                num_samples_per_image=num_samples,
                values=values,
            )
        
        # Policy loss
        policy_loss, info = self.compute_policy_loss(log_probs, old_log_probs, advantages)
        total_loss = policy_loss
        
        # Value loss if PPO
        if self.use_value_function and values is not None and returns is not None:
            value_loss = F.mse_loss(values, returns)
            total_loss = total_loss + self.value_coef * value_loss
            info['value_loss'] = value_loss.item()
        
        info['total_loss'] = total_loss.item()
        
        # Add GDPO-specific metrics
        if reward_components is not None:
            info['num_reward_components'] = len(reward_components)
            for i, rc in enumerate(reward_components):
                info[f'reward_component_{i}_mean'] = rc.mean().item()
                info[f'reward_component_{i}_std'] = rc.std().item()
        
        # Add within-image metrics
        if num_samples is not None and num_samples > 1:
            info['num_samples_per_image'] = num_samples
            num_images = rewards.shape[0] // num_samples
            if rewards.shape[0] == num_images * num_samples:
                rewards_per_image = rewards.view(num_images, num_samples)
                info['within_image_reward_std'] = rewards_per_image.std(dim=1).mean().item()
        
        return total_loss, info
    
    def _compute_standard_advantages(
        self,
        rewards: torch.Tensor,
        num_samples_per_image: Optional[int] = None,
        values: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Standard GRPO advantage computation (fallback when no separate reward components).
        Same as GRPO_V2.compute_advantages().
        """
        num_samples = num_samples_per_image if num_samples_per_image is not None else self.num_samples_per_image
        
        if self.use_value_function and values is not None:
            advantages = rewards - values.detach()
            returns = rewards.clone()
            
            if self.normalize_advantages and num_samples > 1:
                total_samples = advantages.shape[0]
                num_images = total_samples // num_samples
                
                if total_samples == num_images * num_samples:
                    advantages_grouped = advantages.view(num_images, num_samples)
                    group_mean = advantages_grouped.mean(dim=1, keepdim=True)
                    group_std = advantages_grouped.std(dim=1, keepdim=True) + 1e-4
                    advantages_grouped = (advantages_grouped - group_mean) / group_std
                    advantages = advantages_grouped.view(-1)
                else:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-4)
            elif self.normalize_advantages:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-4)
            
            return advantages, returns
        
        # Pure GRPO mode
        returns = None
        
        if num_samples > 1 and self.normalize_advantages:
            total_samples = rewards.shape[0]
            num_images = total_samples // num_samples
            
            if total_samples == num_images * num_samples:
                rewards_grouped = rewards.view(num_images, num_samples)
                group_mean = rewards_grouped.mean(dim=1, keepdim=True)
                group_std = rewards_grouped.std(dim=1, keepdim=True) + 1e-4
                advantages_grouped = (rewards_grouped - group_mean) / group_std
                advantages = advantages_grouped.view(-1)
            else:
                advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
        elif self.normalize_advantages:
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
        else:
            advantages = rewards
        
        return advantages, returns


class GDPORewardComputer:
    """
    Extended reward computer that returns separate reward components for GDPO.
    
    This class wraps RLRewardComputer to also return the individual reward
    components needed for GDPO's decoupled normalization.
    
    For ProstNFound-RL, the two main reward components are:
    1. Classification reward: Based on classification head performance
    2. Attention reward: Based on attention-involvement correspondence
    """
    
    def __init__(
        self,
        base_reward_computer,
        classification_weight: float = 1.0,
        attention_weight: float = 1.0,
    ):
        """
        Args:
            base_reward_computer: RLRewardComputer instance
            classification_weight: Weight for classification reward in GDPO
            attention_weight: Weight for attention reward in GDPO
        """
        self.base_reward_computer = base_reward_computer
        self.classification_weight = classification_weight
        self.attention_weight = attention_weight
    
    def __call__(
        self,
        outputs: Dict[str, torch.Tensor],
        data: Dict[str, torch.Tensor],
        num_samples_per_image: int = 1,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Compute both summed rewards and separate reward components.
        
        Returns:
            total_reward: Summed reward (for backward compatibility)
            reward_components: List of separate reward tensors for GDPO
        """
        # Get the device
        cancer_logits = outputs.get('cancer_logits', outputs.get('mask_logits'))
        device = cancer_logits.device
        
        # Compute classification reward
        cls_reward = self.base_reward_computer.compute_classification_reward(outputs, data)
        
        # Compute attention/sparsity reward based on mode
        reward_mode = self.base_reward_computer.reward_mode
        
        if reward_mode == 'attention_proportional':
            attn_reward = self.base_reward_computer.compute_attention_proportional_reward(outputs, data)
        elif reward_mode == 'classification_gated_needle':
            attn_reward = self.base_reward_computer.compute_classification_gated_needle_reward(outputs, data)
        elif reward_mode == 'sparse_attention':
            attn_reward = self.base_reward_computer.compute_sparse_attention_reward(outputs, data)
        elif reward_mode == 'attention_contrast':
            attn_reward = self.base_reward_computer.compute_attention_contrast_reward(outputs, data)
        elif reward_mode == 'roi_only':
            attn_reward = self.base_reward_computer.compute_roi_involvement_reward(cancer_logits, data)
        elif reward_mode == 'combined_v2':
            # For combined_v2, get the ROI involvement reward as the "attention" component
            attn_reward = self.base_reward_computer.compute_roi_involvement_reward(cancer_logits, data)
        else:
            # For classification_only or other modes, attention reward is zero
            attn_reward = torch.zeros_like(cls_reward)
        
        # Total reward (for backward compatibility and logging)
        total_reward = cls_reward + attn_reward
        
        # Return both
        return total_reward, [cls_reward, attn_reward]


def create_gdpo_optimizer(
    model: nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    """Create optimizer for GDPO training."""
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
