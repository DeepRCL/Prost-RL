"""
DRPO - Domain-Robust Policy Optimization

Based on GDPO with added hierarchical advantage scaling.
Paper: https://arxiv.org/abs/2601.05242

Key insight: DRPO = GDPO + hierarchical advantage scaling through:
1. Intra-domain clustering (difficulty detection via K-Means)
2. Domain temperature scaling (balances rare vs. common domains)
3. Cluster temperature scaling (balances hard vs. easy samples)

For ProstNFound-RL with unbalanced data (mostly benign, few cancer cases):
- DRPO helps balance learning between rare cancer cases and common benign cases
- Temperature scaling upweights hard-to-classify cancer samples
- Domain clustering can group by clinical characteristics (PSA, age, grade)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import logging
import numpy as np
from sklearn.cluster import KMeans


class DRPO:
    """
    Domain-Robust Policy Optimization.
    
    Extends GDPO with hierarchical advantage scaling:
    1. GDPO: Normalize each reward component separately
    2. Domain clustering: Group samples by domain and difficulty
    3. Temperature scaling: Upweight rare domains and hard clusters
    4. Hierarchical advantage: Scale advantages by temperatures
    
    This is particularly useful for unbalanced datasets where we want
    to learn on rare but important samples (e.g., cancer detection).
    
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
        num_clusters: Number of clusters for K-Means difficulty grouping (default: 5)
        epsilon: Small constant to prevent division by zero (default: 1e-4)
        strict_mode: If True, raise exceptions on fallbacks (default: True)
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
        num_clusters: int = 5,
        epsilon: float = 1e-4,
        strict_mode: bool = True,
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
        self.num_clusters = num_clusters
        self.epsilon = epsilon
        self.strict_mode = strict_mode
        
        # Track DRPO usage for verification
        self._drpo_used_count = 0
        self._fallback_count = 0
        
        logging.info(
            f"=" * 60 + "\n"
            f"DRPO INITIALIZED (Domain-Robust Policy Optimization)\n"
            f"  clip_eps={clip_eps}, entropy_coef={entropy_coef}, kl_coef={kl_coef}\n"
            f"  num_samples_per_image={num_samples_per_image}\n"
            f"  num_clusters={num_clusters}, epsilon={epsilon}\n"
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
        
        This is inherited from GDPO: each reward signal is normalized independently
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
                f"[DRPO ERROR] Sample count mismatch!\n"
                f"  Total samples: {total_samples}\n"
                f"  num_samples_per_image: {num_samples_per_image}\n"
                f"  Expected: {num_images * num_samples_per_image} (divisible)\n"
                f"  Falling back to simple batch normalization (NOT real DRPO!)"
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
    
    def cluster_by_difficulty(
        self,
        reward_vectors: torch.Tensor,
        domain_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Cluster questions by difficulty using K-Means on reward vectors.
        
        This implements Step A of DRPO: Intra-Domain Clustering.
        For each domain separately, we cluster questions based on their
        reward vectors (rewards across all rollout samples).
        
        Args:
            reward_vectors: (num_images, num_samples_per_image) - reward vector per question
            domain_ids: (num_images,) - domain ID for each question
            
        Returns:
            cluster_ids: (num_images,) - cluster assignment for each question
            cluster_info: Dict with clustering statistics
        """
        num_images = reward_vectors.shape[0]
        device = reward_vectors.device
        
        # Initialize cluster IDs
        cluster_ids = torch.zeros(num_images, dtype=torch.long, device=device)
        
        # Get unique domains
        unique_domains = domain_ids.unique()
        
        cluster_info = {
            'num_domains': len(unique_domains),
            'clusters_per_domain': {},
        }
        
        global_cluster_id = 0
        
        for domain_id in unique_domains:
            # Get samples in this domain
            domain_mask = domain_ids == domain_id
            domain_indices = torch.where(domain_mask)[0]
            domain_reward_vectors = reward_vectors[domain_mask]
            
            n_samples_in_domain = domain_reward_vectors.shape[0]
            
            # Determine number of clusters for this domain
            # Use min to avoid more clusters than samples
            n_clusters = min(self.num_clusters, max(1, n_samples_in_domain // 2))
            
            if n_clusters <= 1 or n_samples_in_domain <= 1:
                # Not enough samples to cluster
                cluster_ids[domain_mask] = global_cluster_id
                cluster_info['clusters_per_domain'][domain_id.item()] = 1
                global_cluster_id += 1
                continue
            
            # Run K-Means clustering (on CPU for sklearn)
            reward_vectors_cpu = domain_reward_vectors.cpu().numpy()
            
            try:
                kmeans = KMeans(
                    n_clusters=n_clusters,
                    n_init='auto',
                    random_state=42,
                    max_iter=50,  # Limit iterations for speed
                )
                labels = kmeans.fit_predict(reward_vectors_cpu)
                
                # Assign cluster IDs (offset by global_cluster_id to make unique)
                for i, idx in enumerate(domain_indices):
                    cluster_ids[idx] = global_cluster_id + labels[i]
                
                cluster_info['clusters_per_domain'][domain_id.item()] = n_clusters
                global_cluster_id += n_clusters
                
            except Exception as e:
                logging.warning(f"K-Means clustering failed for domain {domain_id.item()}: {e}")
                # Fallback: assign all to same cluster
                cluster_ids[domain_mask] = global_cluster_id
                cluster_info['clusters_per_domain'][domain_id.item()] = 1
                global_cluster_id += 1
        
        cluster_info['total_clusters'] = global_cluster_id
        
        return cluster_ids, cluster_info
    
    def compute_domain_temperature(
        self,
        rewards: torch.Tensor,
        domain_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute domain temperature scaling factors.
        
        Temperature formula: T_g = max(sqrt(N_g) * μ_g, ε)
        where:
        - N_g: number of questions in this domain in current batch
        - μ_g: average reward of this domain in current batch
        - ε: small constant to prevent division by zero
        
        Lower temperature = higher advantage scaling (upweights rare/hard domains)
        
        Args:
            rewards: (num_images,) - average reward per question
            domain_ids: (num_images,) - domain ID for each question
            
        Returns:
            temperatures: (num_images,) - temperature for each question
        """
        num_images = rewards.shape[0]
        device = rewards.device
        
        temperatures = torch.zeros(num_images, device=device)
        unique_domains = domain_ids.unique()
        
        for domain_id in unique_domains:
            domain_mask = domain_ids == domain_id
            domain_rewards = rewards[domain_mask]
            
            N_g = domain_rewards.shape[0]
            mu_g = domain_rewards.mean()
            
            # T_g = max(sqrt(N_g) * μ_g, ε)
            # Note: If μ_g is negative, use abs to prevent negative temperature
            T_g = max(np.sqrt(N_g) * abs(mu_g.item()), self.epsilon)
            
            temperatures[domain_mask] = T_g
        
        return temperatures
    
    def compute_cluster_temperature(
        self,
        rewards: torch.Tensor,
        cluster_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cluster temperature scaling factors.
        
        Temperature formula: T_c = max(sqrt(N_c) * μ_c, ε)
        where:
        - N_c: number of questions in this cluster
        - μ_c: average reward of this cluster
        - ε: small constant to prevent division by zero
        
        Lower temperature = higher advantage scaling (upweights hard clusters)
        
        Args:
            rewards: (num_images,) - average reward per question
            cluster_ids: (num_images,) - cluster assignment for each question
            
        Returns:
            temperatures: (num_images,) - temperature for each question
        """
        num_images = rewards.shape[0]
        device = rewards.device
        
        temperatures = torch.zeros(num_images, device=device)
        unique_clusters = cluster_ids.unique()
        
        for cluster_id in unique_clusters:
            cluster_mask = cluster_ids == cluster_id
            cluster_rewards = rewards[cluster_mask]
            
            N_c = cluster_rewards.shape[0]
            mu_c = cluster_rewards.mean()
            
            # T_c = max(sqrt(N_c) * μ_c, ε)
            T_c = max(np.sqrt(N_c) * abs(mu_c.item()), self.epsilon)
            
            temperatures[cluster_mask] = T_c
        
        return temperatures
    
    def compute_drpo_advantages(
        self,
        reward_components: List[torch.Tensor],
        num_samples_per_image: Optional[int] = None,
        values: Optional[torch.Tensor] = None,
        domain_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict]:
        """
        Compute DRPO advantages with hierarchical scaling.
        
        DRPO workflow:
        1. GDPO normalization: Normalize each reward component separately
        2. Combine normalized advantages with weights (GDPO)
        3. Cluster by difficulty: K-Means on reward vectors (NEW)
        4. Temperature scaling: Domain + Cluster temperatures (NEW)
        5. Hierarchical scaling: s_scaled = s / (T_g * T_c) (NEW)
        6. Global re-normalization: Final whitening (NEW)
        
        Args:
            reward_components: List of reward tensors, each of shape (B * num_samples,)
            num_samples_per_image: Number of samples per image
            values: Value estimates (B * num_samples,) if PPO mode
            domain_ids: (B,) domain ID for each image (required for DRPO)
            
        Returns:
            advantages: Combined DRPO advantages (B * num_samples,)
            returns: Returns for value loss (B * num_samples,) or None
            drpo_info: Dictionary with DRPO statistics
        """
        num_samples = num_samples_per_image if num_samples_per_image is not None else self.num_samples_per_image
        device = reward_components[0].device
        
        drpo_info = {}
        
        # Step 1-2: GDPO normalization (same as GDPO)
        normalized_advantages = []
        for i, reward in enumerate(reward_components):
            if num_samples > 1:
                adv = self.normalize_reward_within_image(reward, num_samples)
            else:
                adv = reward - reward.mean()
            normalized_advantages.append(adv)
        
        # Combine normalized advantages with weights
        weights = self.reward_weights
        if len(weights) < len(normalized_advantages):
            weights = weights + [1.0] * (len(normalized_advantages) - len(weights))
        
        combined_advantage = torch.zeros_like(normalized_advantages[0])
        for i, adv in enumerate(normalized_advantages):
            weight = weights[i] if i < len(weights) else 1.0
            combined_advantage = combined_advantage + weight * adv
        
        # If no domain IDs provided, skip hierarchical scaling (fallback to GDPO)
        if domain_ids is None:
            logging.warning(
                "[DRPO] No domain_ids provided - falling back to GDPO (no hierarchical scaling)"
            )
            self._fallback_count += 1
            
            if self.normalize_advantages:
                adv_mean = combined_advantage.mean()
                adv_std = combined_advantage.std() + 1e-4
                combined_advantage = (combined_advantage - adv_mean) / adv_std
            
            returns = None
            if self.use_value_function and values is not None:
                total_rewards = sum(reward_components)
                returns = total_rewards.clone()
            
            drpo_info['hierarchical_scaling'] = False
            return combined_advantage, returns, drpo_info
        
        # Step 3: Cluster by difficulty (DRPO)
        self._drpo_used_count += 1
        
        # Reshape rewards to (num_images, num_samples_per_image) for clustering
        total_samples = reward_components[0].shape[0]
        num_images = total_samples // num_samples
        
        if total_samples != num_images * num_samples:
            logging.error(
                f"[DRPO] Sample count mismatch: {total_samples} != {num_images} * {num_samples}"
            )
            # Fallback to GDPO
            if self.normalize_advantages:
                combined_advantage = (combined_advantage - combined_advantage.mean()) / (combined_advantage.std() + 1e-4)
            returns = None
            if self.use_value_function and values is not None:
                returns = sum(reward_components).clone()
            drpo_info['hierarchical_scaling'] = False
            return combined_advantage, returns, drpo_info
        
        # Create reward vectors for clustering (use total reward)
        total_rewards = sum(reward_components)
        reward_vectors = total_rewards.view(num_images, num_samples)
        
        # Compute average reward per question (for temperature calculation)
        avg_rewards_per_image = reward_vectors.mean(dim=1)
        
        # Cluster by difficulty
        cluster_ids, cluster_info = self.cluster_by_difficulty(reward_vectors, domain_ids)
        
        # Step 4: Compute temperatures
        domain_temps = self.compute_domain_temperature(avg_rewards_per_image, domain_ids)
        cluster_temps = self.compute_cluster_temperature(avg_rewards_per_image, cluster_ids)
        
        # Step 5: Hierarchical advantage scaling
        # Expand temperatures to match combined_advantage shape (B * num_samples,)
        domain_temps_expanded = domain_temps.repeat_interleave(num_samples)
        cluster_temps_expanded = cluster_temps.repeat_interleave(num_samples)
        
        # s_scaled = s / (T_g * T_c)
        combined_temps = domain_temps_expanded * cluster_temps_expanded
        scaled_advantages = combined_advantage / (combined_temps + self.epsilon)
        
        # Step 6: Global re-normalization
        if self.normalize_advantages:
            scaled_advantages = (scaled_advantages - scaled_advantages.mean()) / (scaled_advantages.std() + 1e-4)
        
        # Handle value function for PPO mode
        returns = None
        if self.use_value_function and values is not None:
            returns = total_rewards.clone()
        
        # Collect DRPO statistics
        drpo_info.update({
            'hierarchical_scaling': True,
            'num_domains': cluster_info['num_domains'],
            'num_clusters': cluster_info['total_clusters'],
            'domain_temp_mean': domain_temps.mean().item(),
            'domain_temp_std': domain_temps.std().item(),
            'cluster_temp_mean': cluster_temps.mean().item(),
            'cluster_temp_std': cluster_temps.std().item(),
            'advantage_scaling_factor_mean': (1.0 / (combined_temps + self.epsilon)).mean().item(),
        })
        
        return scaled_advantages, returns, drpo_info
    
    def compute_policy_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute PPO-style clipped policy loss.
        
        Same as GDPO implementation.
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
                f"NaN/Inf in DRPO loss! ratio_range=[{ratio.min().item():.2f}, {ratio.max().item():.2f}]"
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
        domain_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total DRPO loss.
        
        IMPORTANT: For DRPO to work correctly, you must pass:
        - reward_components: List of separate reward signals
        - domain_ids: Domain assignment for each image
        
        Args:
            log_probs: Current log probs (B * num_samples, k) or (B * num_samples,)
            old_log_probs: Old log probs (B * num_samples, k) or (B * num_samples,)
            rewards: Summed rewards (B * num_samples,) - used if reward_components is None
            num_samples_per_image: Number of samples per image
            values: Value estimates (B * num_samples,) if PPO mode
            reward_components: List of separate reward tensors for DRPO
            domain_ids: (B,) domain ID for each image
            
        Returns:
            total_loss: Loss
            info: Logging info with DRPO statistics
        """
        num_samples = num_samples_per_image if num_samples_per_image is not None else self.num_samples_per_image
        
        # Compute advantages with DRPO (hierarchical scaling)
        if reward_components is not None and len(reward_components) > 0:
            # True DRPO: use hierarchical advantage scaling
            if self._drpo_used_count == 0:
                logging.info(
                    f"[DRPO] First call - Using hierarchical advantage scaling with {len(reward_components)} reward components"
                )
            
            advantages, returns, drpo_info = self.compute_drpo_advantages(
                reward_components,
                num_samples_per_image=num_samples,
                values=values,
                domain_ids=domain_ids,
            )
        else:
            # Fallback to standard GRPO
            error_msg = (
                f"[DRPO CRITICAL ERROR] No reward_components provided!\n"
                f"  reward_components: {reward_components}\n"
                f"  DRPO requires separate reward components. Falling back to standard GRPO!"
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
            drpo_info = {'hierarchical_scaling': False}
        
        # Policy loss
        policy_loss, info = self.compute_policy_loss(log_probs, old_log_probs, advantages)
        total_loss = policy_loss
        
        # Value loss if PPO
        if self.use_value_function and values is not None and returns is not None:
            value_loss = F.mse_loss(values, returns)
            total_loss = total_loss + self.value_coef * value_loss
            info['value_loss'] = value_loss.item()
        
        info['total_loss'] = total_loss.item()
        
        # Add DRPO-specific metrics
        info.update(drpo_info)
        
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
        Standard GRPO advantage computation (fallback).
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


def create_drpo_optimizer(
    model: nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    """Create optimizer for DRPO training."""
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
