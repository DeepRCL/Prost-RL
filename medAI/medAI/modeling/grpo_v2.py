"""
GRPO V2 - Fixed within-image advantage normalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import logging


class GRPO_V2:
    """
    Group Relative Policy Optimization with proper within-image comparison.
    
    Args:
        clip_eps: PPO clipping epsilon (default: 0.2)
        entropy_coef: Entropy coefficient (default: 0.01)
        max_grad_norm: Max gradient norm (default: 0.5)
        kl_coef: KL penalty coefficient (default: 0.01)
        normalize_advantages: Normalize advantages (default: True)
        num_samples_per_image: Samples per image for within-image comparison (default: 4)
        use_value_function: Use value function for PPO (default: False)
        value_coef: Value loss coefficient (default: 0.5)
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
    ):
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.kl_coef = kl_coef
        self.normalize_advantages = normalize_advantages
        self.num_samples_per_image = num_samples_per_image
        self.use_value_function = use_value_function
        self.value_coef = value_coef
        
        mode = "PPO" if use_value_function else "Pure GRPO"
        logging.info(
            f"Initialized {mode} with clip_eps={clip_eps}, "
            f"entropy_coef={entropy_coef}, kl_coef={kl_coef}, "
            f"num_samples_per_image={num_samples_per_image}"
        )
    
    def compute_advantages(
        self,
        rewards: torch.Tensor,
        num_samples_per_image: Optional[int] = None,
        values: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute advantages with WITHIN-IMAGE normalization (not within-batch).
        
        GRPO mode: A_i = (R_i - mean(R_image)) / std(R_image)
        PPO mode: A_i = R_i - V_i, then normalize within image
        
        Args:
            rewards: Rewards (B * num_samples,)
            num_samples_per_image: Number of samples per image
            values: Value estimates (B * num_samples,) if PPO
            
        Returns:
            advantages: Advantages (B * num_samples,)
            returns: Returns for value loss (B * num_samples,) or None
        """
        num_samples = num_samples_per_image if num_samples_per_image is not None else self.num_samples_per_image
        
        if self.use_value_function and values is not None:
            # PPO mode with value function
            advantages = rewards - values.detach()
            returns = rewards.clone()
            
            # Within-image normalization
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
        
        # Pure GRPO mode (no value function)
        returns = None
        
        if num_samples > 1 and self.normalize_advantages:
            total_samples = rewards.shape[0]
            num_images = total_samples // num_samples
            
            if total_samples == num_images * num_samples:
                # Within-image normalization
                rewards_grouped = rewards.view(num_images, num_samples)
                group_mean = rewards_grouped.mean(dim=1, keepdim=True)
                group_std = rewards_grouped.std(dim=1, keepdim=True) + 1e-4
                advantages_grouped = (rewards_grouped - group_mean) / group_std
                advantages = advantages_grouped.view(-1)
            else:
                logging.warning(
                    f"Within-image normalization failed: total={total_samples}, "
                    f"num_images={num_images}, num_samples={num_samples}. "
                    f"Falling back to batch normalization."
                )
                advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
        elif self.normalize_advantages:
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
        else:
            advantages = rewards
        
        return advantages, returns
    
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
                f"NaN/Inf in loss! ratio_range=[{ratio.min().item():.2f}, {ratio.max().item():.2f}]"
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
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss.
        
        Args:
            log_probs: Current log probs (B * num_samples, k) or (B * num_samples,)
            old_log_probs: Old log probs (B * num_samples, k) or (B * num_samples,)
            rewards: Rewards (B * num_samples,)
            num_samples_per_image: Number of samples per image
            values: Value estimates (B * num_samples,) if PPO
            
        Returns:
            total_loss: Loss
            info: Logging info
        """
        # Compute advantages (within-image normalization)
        advantages, returns = self.compute_advantages(
            rewards,
            num_samples_per_image=num_samples_per_image,
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
        
        # Add within-image metrics
        if num_samples_per_image is not None and num_samples_per_image > 1:
            info['num_samples_per_image'] = num_samples_per_image
            num_images = rewards.shape[0] // num_samples_per_image
            if rewards.shape[0] == num_images * num_samples_per_image:
                rewards_per_image = rewards.view(num_images, num_samples_per_image)
                info['within_image_reward_std'] = rewards_per_image.std(dim=1).mean().item()
        
        return total_loss, info


class BatchedGRPOTrainer:
    """
    Batched GRPO trainer with batched forward passes.
    """
    
    def __init__(
        self,
        grpo: Optional[GRPO_V2] = None,
        num_samples_per_image: int = 4,
        device: str = 'cuda',
    ):
        self.grpo = grpo if grpo is not None else GRPO_V2(num_samples_per_image=num_samples_per_image)
        self.num_samples_per_image = num_samples_per_image
        self.device = device
        
        logging.info(
            f"Initialized BatchedGRPOTrainer with {num_samples_per_image} samples per image"
        )
    
    @staticmethod
    def replicate_batch(data: Dict[str, torch.Tensor], num_samples: int) -> Dict[str, torch.Tensor]:
        """Replicate batch for batched sampling."""
        replicated = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                replicated[key] = value.repeat_interleave(num_samples, dim=0)
            elif isinstance(value, list):
                replicated[key] = [v for v in value for _ in range(num_samples)]
            else:
                replicated[key] = value
        return replicated
    
    @staticmethod
    def group_by_image(tensor: torch.Tensor, batch_size: int, num_samples: int) -> torch.Tensor:
        """Reshape tensor from flat to grouped by image."""
        shape = tensor.shape
        new_shape = (batch_size, num_samples) + shape[1:]
        return tensor.view(*new_shape)


def create_grpo_optimizer(
    model: nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    """Create optimizer for GRPO training."""
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
