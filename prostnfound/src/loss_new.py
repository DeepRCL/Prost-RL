import argparse
import json
from typing import Callable
import torch
from torch import nn
import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import repeat, rearrange


class CancerDetectionValidRegionLoss(nn.Module):
    def __init__(
        self,
        base_loss: Callable = F.binary_cross_entropy_with_logits,
        prostate_mask: bool = True,
        needle_mask: bool = True,
    ):
        super().__init__()
        self.base_loss = base_loss
        self.prostate_mask = prostate_mask
        self.needle_mask = needle_mask

    def forward(self, data: dict):
        cancer_logits = data["cancer_logits"]
        label = data["label"].to(cancer_logits.device)
        prostate_mask = data["prostate_mask"].to(cancer_logits.device)
        needle_mask = data["needle_mask"].to(cancer_logits.device)

        masks = []
        for i in range(len(cancer_logits)):
            mask = torch.ones(
                prostate_mask[i].shape, device=prostate_mask[i].device
            ).bool()
            if self.prostate_mask:
                mask &= prostate_mask[i] > 0.5
            if self.needle_mask:
                mask &= needle_mask[i] > 0.5
            masks.append(mask)
        masks = torch.stack(masks)
        predictions, batch_idx = MaskedPredictionModule()(cancer_logits, masks)
        labels = torch.zeros(len(predictions), device=predictions.device)
        for i in range(len(predictions)):
            labels[i] = label[batch_idx[i]]
        labels = labels[..., None]  # needs to match N, C shape of preds

        return self.base_loss(predictions, labels)


class ProportionBCE(nn.Module):
    def __init__(
        self,
        l1_penalty_lambda: float | None = None,
        entropy_penalty_lambda: float | None = None,
        pos_weight: float = 1.0,
    ):
        super().__init__()
        self.l1_penalty_lambda = l1_penalty_lambda
        self.entropy_penalty_lambda = entropy_penalty_lambda
        self.pos_weight = pos_weight

    def forward(self, bag_of_logits, true_prop):
        # Numerical stability: use a larger epsilon to prevent underflow
        eps = 1e-7
        
        # Compute probabilities from logits
        probs = bag_of_logits.sigmoid()
        
        # Clamp individual probabilities to prevent extreme values
        probs = probs.clamp(min=eps, max=1 - eps)
        
        # Compute mean probability (predicted proportion)
        pred_prob = probs.mean()
        
        # CRITICAL: Clamp pred_prob before taking log to prevent NaN
        # This is especially important when model outputs all near-zero values
        pred_prob = pred_prob.clamp(min=eps, max=1 - eps)
        
        # Compute BCE loss on proportions
        # Use log1p for better numerical stability when computing log(1-x)
        loss = -true_prop * torch.log(pred_prob) - (1 - true_prop) * torch.log(1 - pred_prob)
        
        # Apply positive weight if this is a cancer case
        if true_prop.item() > 0:
            loss = loss * self.pos_weight
            
        # L1 penalty on individual probabilities (sparsity regularization)
        if self.l1_penalty_lambda:
            loss = loss + self.l1_penalty_lambda * probs.abs().sum()
            
        # Entropy penalty for exploration/smoothness
        if self.entropy_penalty_lambda:
            # Binary entropy: H(p) = -p*log(p) - (1-p)*log(1-p)
            # Already clamped probs, so this is safe
            entropy = -probs * torch.log(probs) - (1 - probs) * torch.log(1 - probs)
            # Check for NaN in entropy (shouldn't happen with clamping, but be safe)
            entropy = torch.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0)
            loss = loss + self.entropy_penalty_lambda * entropy.mean()

        return loss


# =============================================================================
# NOISE-ROBUST LOSS FUNCTIONS
# =============================================================================

class TopKMILLoss(nn.Module):
    """
    Top-K Involvement Loss (Multiple Instance Learning approach)
    
    Instead of forcing ALL needle pixels to match the label (which forces benign 
    pixels to be predicted as cancer), we only force the Top k% of pixels to be 
    predicted as cancer, where k is the involvement percentage.
    
    This is a form of MIL: if the core has 55% cancer, we assume the "most 
    suspicious" 55% of pixels are the cancer, and the rest are likely benign.
    
    Why it's better: It solves the "blur" problem of ProstNFound+ by allowing 
    the model to predict 0 for the benign parts of the needle trace.
    
    Args:
        pos_weight: Weight for positive (cancer) cases
        min_top_k_fraction: Minimum fraction of pixels to select (default: 0.05)
                           Prevents selecting too few pixels for very low involvement
        use_soft_topk: If True, use differentiable soft top-k. If False, use hard selection.
        benign_weight: Weight for benign pixel loss (default: 1.0, previously 0.5 was too weak)
    """
    
    def __init__(
        self,
        pos_weight: float = 1.0,
        min_top_k_fraction: float = 0.05,
        use_soft_topk: bool = False,
        temperature: float = 0.1,
        benign_weight: float = 1.0,  # INCREASED from 0.5 - need strong push to 0
    ):
        super().__init__()
        self.pos_weight = pos_weight
        self.min_top_k_fraction = min_top_k_fraction
        self.use_soft_topk = use_soft_topk
        self.temperature = temperature
        self.benign_weight = benign_weight
    
    def forward(self, bag_of_logits, true_prop):
        """
        Args:
            bag_of_logits: Tensor of shape (N,) or (N, 1) containing logits for needle pixels
            true_prop: Scalar involvement proportion (0 to 1)
        """
        eps = 1e-7
        
        # Flatten logits
        logits = bag_of_logits.view(-1)
        n_pixels = logits.shape[0]
        
        if n_pixels == 0:
            return torch.tensor(0.0, device=bag_of_logits.device)
        
        # Compute probabilities
        probs = torch.sigmoid(logits)
        
        # Handle benign case (involvement = 0)
        if true_prop.item() == 0:
            # For benign cores, all pixels should be predicted as benign (prob = 0)
            loss = F.binary_cross_entropy_with_logits(
                logits, torch.zeros_like(logits), reduction='mean'
            )
            return loss
        
        # For cancer cases: only supervise top-k pixels
        # k = max(min_top_k_fraction, involvement) * n_pixels
        k_fraction = max(self.min_top_k_fraction, true_prop.item())
        k = max(1, int(k_fraction * n_pixels))
        
        if self.use_soft_topk:
            # Differentiable soft top-k using softmax with temperature
            # Higher logits get higher weights
            weights = F.softmax(logits / self.temperature, dim=0)
            # Weighted BCE: top-k pixels get higher weights
            loss = F.binary_cross_entropy_with_logits(
                logits, torch.ones_like(logits), reduction='none'
            )
            loss = (loss * weights * n_pixels).mean()  # Rescale by n_pixels
        else:
            # Hard top-k: select k pixels with highest probabilities
            _, topk_indices = torch.topk(probs, k, largest=True)
            topk_logits = logits[topk_indices]
            
            # These top-k pixels should be cancer (label = 1)
            cancer_loss = F.binary_cross_entropy_with_logits(
                topk_logits, torch.ones_like(topk_logits), reduction='mean'
            )
            
            # Penalize the remaining pixels to be benign
            # This is CRITICAL for sparsity - without strong benign loss,
            # the model predicts high values everywhere
            remaining_mask = torch.ones(n_pixels, dtype=torch.bool, device=logits.device)
            remaining_mask[topk_indices] = False
            if remaining_mask.sum() > 0:
                remaining_logits = logits[remaining_mask]
                benign_loss = F.binary_cross_entropy_with_logits(
                    remaining_logits, torch.zeros_like(remaining_logits), reduction='mean'
                )
                # Weight benign loss - needs to be strong enough to push to 0!
                loss = cancer_loss + self.benign_weight * benign_loss
            else:
                loss = cancer_loss
        
        # Apply positive weight
        loss = loss * self.pos_weight
        
        return loss


class ThresholdedInvolvementLoss(nn.Module):
    """
    Thresholded Involvement Loss
    
    Instead of comparing mean probability to involvement, we threshold the 
    predictions at 0.5 and compare the fraction of "on" pixels to involvement.
    
    This makes the loss more directly optimize for the calibration metric:
    - If involvement = 50%, we want 50% of pixels to have prob > 0.5
    
    The loss encourages:
    - For 50% involvement: ~50% of needle pixels should exceed 0.5 threshold
    - Sharp predictions (either high or low, not middle values like 0.55)
    
    Args:
        threshold: Probability threshold for considering a pixel as "on"
        use_soft_threshold: If True, use sigmoid approximation of step function
        soft_temperature: Temperature for soft threshold (lower = sharper)
        pos_weight: Weight for positive (cancer) cases
    """
    
    def __init__(
        self,
        threshold: float = 0.5,
        use_soft_threshold: bool = True,
        soft_temperature: float = 0.1,
        pos_weight: float = 1.0,
        entropy_penalty_lambda: float = None,
    ):
        super().__init__()
        self.threshold = threshold
        self.use_soft_threshold = use_soft_threshold
        self.soft_temperature = soft_temperature
        self.pos_weight = pos_weight
        self.entropy_penalty_lambda = entropy_penalty_lambda
    
    def forward(self, bag_of_logits, true_prop):
        """
        Args:
            bag_of_logits: Tensor of shape (N,) or (N, 1) containing logits for needle pixels
            true_prop: Scalar involvement proportion (0 to 1)
        """
        eps = 1e-7
        
        # Flatten logits
        logits = bag_of_logits.view(-1)
        n_pixels = logits.shape[0]
        
        if n_pixels == 0:
            return torch.tensor(0.0, device=bag_of_logits.device)
        
        # Compute probabilities
        probs = torch.sigmoid(logits)
        
        if self.use_soft_threshold:
            # Soft thresholding using sigmoid
            # centered_probs = (probs - threshold) / temperature
            # soft_threshold(probs) = sigmoid((probs - threshold) / temperature)
            centered = (probs - self.threshold) / self.soft_temperature
            thresholded = torch.sigmoid(centered)
        else:
            # Hard thresholding (not differentiable, use with caution)
            thresholded = (probs > self.threshold).float()
        
        # Predicted proportion of "on" pixels
        pred_prop = thresholded.mean().clamp(min=eps, max=1 - eps)
        true_prop_clamped = true_prop.clamp(min=eps, max=1 - eps)
        
        # BCE loss on thresholded proportions
        loss = -true_prop_clamped * torch.log(pred_prop) - (1 - true_prop_clamped) * torch.log(1 - pred_prop)
        
        # Also add standard BCE loss to maintain gradients for learning
        standard_bce = -true_prop * torch.log(probs.mean().clamp(min=eps, max=1 - eps)) \
                       - (1 - true_prop) * torch.log(1 - probs.mean().clamp(min=eps, max=1 - eps))
        
        # Combine thresholded loss with standard BCE
        loss = 0.5 * loss + 0.5 * standard_bce
        
        # Apply positive weight
        if true_prop.item() > 0:
            loss = loss * self.pos_weight
        
        # Entropy penalty to encourage sharp predictions
        if self.entropy_penalty_lambda:
            probs_clamped = probs.clamp(min=eps, max=1 - eps)
            entropy = -probs_clamped * torch.log(probs_clamped) - (1 - probs_clamped) * torch.log(1 - probs_clamped)
            entropy = torch.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0)
            loss = loss + self.entropy_penalty_lambda * entropy.mean()
        
        return loss


class SymmetricCrossEntropyLoss(nn.Module):
    """
    Symmetric Cross Entropy (SCE) for Noise-Robust Learning
    
    From "Symmetric Cross Entropy for Robust Learning with Noisy Labels" (ICCV 2019)
    and used in Manifold DivideMix paper.
    
    Standard CE: Penalizes model when it predicts "Benign" on a "Cancer" pixel 
                 (Hard punishment for missing the noisy label)
    
    Reverse CE: Penalizes the label when the model is confident 
                (Allows model to ignore the label if it strongly disagrees)
    
    SCE = α * CE(p, q) + β * RCE(p, q)
    
    Where:
    - CE = -q * log(p)  (standard cross entropy, q is label, p is prediction)
    - RCE = -p * log(q) (reverse cross entropy)
    
    For proportion-based learning:
    - CE pushes predicted proportion toward true involvement
    - RCE allows predicted proportion to "override" noisy involvement if confident
    
    Args:
        alpha: Weight for standard cross entropy (default: 1.0)
        beta: Weight for reverse cross entropy (default: 1.0)
        pos_weight: Weight for positive (cancer) cases
        epsilon: Small value for numerical stability in RCE
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        pos_weight: float = 1.0,
        epsilon: float = 1e-4,
        entropy_penalty_lambda: float = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.pos_weight = pos_weight
        self.epsilon = epsilon  # Larger epsilon for RCE stability
        self.entropy_penalty_lambda = entropy_penalty_lambda
    
    def forward(self, bag_of_logits, true_prop):
        """
        Args:
            bag_of_logits: Tensor of shape (N,) or (N, 1) containing logits for needle pixels
            true_prop: Scalar involvement proportion (0 to 1)
        """
        eps = 1e-7
        
        # Flatten logits
        logits = bag_of_logits.view(-1)
        
        if logits.shape[0] == 0:
            return torch.tensor(0.0, device=bag_of_logits.device)
        
        # Compute probabilities
        probs = torch.sigmoid(logits).clamp(min=eps, max=1 - eps)
        
        # Predicted proportion
        pred_prop = probs.mean().clamp(min=eps, max=1 - eps)
        
        # Clamp true proportion (label) for RCE
        # Use larger epsilon for labels in RCE to prevent -inf when label is 0 or 1
        true_prop_ce = true_prop.clamp(min=eps, max=1 - eps)
        true_prop_rce = true_prop.clamp(min=self.epsilon, max=1 - self.epsilon)
        
        # Standard Cross Entropy: -q * log(p) - (1-q) * log(1-p)
        # Pushes prediction toward label
        ce_loss = -true_prop_ce * torch.log(pred_prop) - (1 - true_prop_ce) * torch.log(1 - pred_prop)
        
        # Reverse Cross Entropy: -p * log(q) - (1-p) * log(1-q)
        # Allows model to "ignore" noisy labels when confident
        rce_loss = -pred_prop * torch.log(true_prop_rce) - (1 - pred_prop) * torch.log(1 - true_prop_rce)
        
        # Handle potential NaN in RCE (shouldn't happen with clamping, but be safe)
        rce_loss = torch.nan_to_num(rce_loss, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Symmetric Cross Entropy
        loss = self.alpha * ce_loss + self.beta * rce_loss
        
        # Apply positive weight
        if true_prop.item() > 0:
            loss = loss * self.pos_weight
        
        # Entropy penalty for exploration/smoothness
        if self.entropy_penalty_lambda:
            entropy = -probs * torch.log(probs) - (1 - probs) * torch.log(1 - probs)
            entropy = torch.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0)
            loss = loss + self.entropy_penalty_lambda * entropy.mean()
        
        return loss


class TopKInvolvementMILLoss(nn.Module):
    """
    Wrapper that applies TopKMILLoss within the CancerDetectionMILLoss framework.
    
    This is a combination of:
    1. Top-K selection: Only supervise top k% pixels as cancer
    2. MIL framework: Process needle mask and handle batches properly
    
    Args:
        benign_weight: Weight for benign pixel loss (default: 1.0)
                       Higher values push non-topk pixels more strongly to 0.
    """
    
    def __init__(
        self,
        pos_weight: float = 1.0,
        min_top_k_fraction: float = 0.05,
        use_soft_topk: bool = False,
        temperature: float = 0.1,
        treat_gg1_as_benign: bool = False,
        benign_weight: float = 1.0,  # Controls how strongly non-topk pixels are pushed to 0
    ):
        super().__init__()
        self.base_loss = TopKMILLoss(
            pos_weight=pos_weight,
            min_top_k_fraction=min_top_k_fraction,
            use_soft_topk=use_soft_topk,
            temperature=temperature,
            benign_weight=benign_weight,
        )
        self.treat_gg1_as_benign = treat_gg1_as_benign

    def forward(self, data):
        cancer_logits = data["cancer_logits"]
        batch_size = len(cancer_logits)
        prostate_mask = data["prostate_mask"].to(cancer_logits.device)
        needle_mask = data["needle_mask"].to(cancer_logits.device)
        involvement = data["involvement"].to(cancer_logits.device)
        grade_group = data["grade_group"].to(cancer_logits.device)

        if self.treat_gg1_as_benign:
            involvement = involvement.clone()
            involvement[grade_group == 1] = 0.0

        masks = []
        for i in range(len(cancer_logits)):
            mask = torch.ones(
                prostate_mask[i].shape, device=prostate_mask[i].device
            ).bool()
            mask &= prostate_mask[i] > 0.5
            mask &= needle_mask[i] > 0.5
            masks.append(mask)
        masks = torch.stack(masks)
        predictions, batch_idx = MaskedPredictionModule()(cancer_logits, masks)

        loss = torch.tensor(0.0, device=cancer_logits.device)
        for i in range(batch_size):
            bag_i = predictions[batch_idx == i]
            involvement_i = involvement[i]

            loss = loss + self.base_loss(bag_i, involvement_i)

        return loss


class CancerDetectionMILLoss(nn.Module):
    def __init__(self, base_loss=ProportionBCE(), treat_gg1_as_benign=False):

        super().__init__()
        self.base_loss = base_loss
        self.treat_gg1_as_benign = treat_gg1_as_benign

    def forward(self, data):
        cancer_logits = data["cancer_logits"]
        batch_size = len(cancer_logits)
        prostate_mask = data["prostate_mask"].to(cancer_logits.device)
        needle_mask = data["needle_mask"].to(cancer_logits.device)
        involvement = data["involvement"].to(cancer_logits.device)
        grade_group = data["grade_group"].to(cancer_logits.device)

        if self.treat_gg1_as_benign:
            involvement[grade_group == 1] = 0.0

        masks = []
        for i in range(len(cancer_logits)):
            mask = torch.ones(
                prostate_mask[i].shape, device=prostate_mask[i].device
            ).bool()
            mask &= prostate_mask[i] > 0.5
            mask &= needle_mask[i] > 0.5
            masks.append(mask)
        masks = torch.stack(masks)
        predictions, batch_idx = MaskedPredictionModule()(cancer_logits, masks)

        loss = torch.tensor(0, device=cancer_logits.device)
        for i in range(batch_size):
            bag_i = predictions[batch_idx == i]
            involvement_i = involvement[i]

            loss = loss + self.base_loss(bag_i, involvement_i)

        return loss


class InvolvementL1Loss(nn.Module):
    def __init__(self, prostate_penalty=True, pos_weight=1):
        super().__init__()
        self.prostate_penalty = prostate_penalty
        self.pos_weight = pos_weight

    def __call__(self, data):
        avg_needle_heatmap_value = data["average_needle_heatmap_value"]
        B = len(avg_needle_heatmap_value)
        device = avg_needle_heatmap_value.device
        avg_prostate_heatmap_value = data["average_prostate_heatmap_value"]
        involvement = data["involvement"].to(device)
        cores_positive_for_patient = data["cores_positive_for_patient"]

        loss = torch.tensor(0, device=device)
        loss = loss + torch.nn.functional.l1_loss(
            avg_needle_heatmap_value, involvement, reduction="none"
        )
        for idx in range(B):
            if involvement[idx] > 0:
                loss[idx] *= self.pos_weight
        loss = loss.mean()

        if self.prostate_penalty:
            for idx in range(B):
                if cores_positive_for_patient[idx] == 0:
                    loss += avg_prostate_heatmap_value[idx]

        return loss


class InvolvementMSELoss(nn.Module):
    def __call__(self, data):
        avg_needle_heatmap_value = data["average_needle_heatmap_value"]
        B = len(avg_needle_heatmap_value)
        device = avg_needle_heatmap_value.device
        involvement = data["involvement"].to(device)

        loss = torch.nn.functional.mse_loss(avg_needle_heatmap_value, involvement)
        return loss


class MaskedPredictionModule(nn.Module):
    """
    Computes the patch and core predictions and labels within the valid loss region for a heatmap.
    """

    def __init__(self):
        super().__init__()

    def forward(self, heatmap_logits, mask):
        """Computes the patch and core predictions and labels within the valid loss region."""
        B, C, H, W = heatmap_logits.shape

        assert mask.shape == (
            B,
            1,
            H,
            W,
        ), f"Expected mask shape to be {(B, 1, H, W)}, got {mask.shape} instead."

        # mask = mask.float()
        # mask = torch.nn.functional.interpolate(mask, size=(H, W)) > 0.5

        core_idx = torch.arange(B, device=heatmap_logits.device)
        core_idx = repeat(core_idx, "b -> b h w", h=H, w=W)

        core_idx_flattened = rearrange(core_idx, "b h w -> (b h w)")
        mask_flattened = rearrange(mask, "b c h w -> (b h w) c")[..., 0]
        logits_flattened = rearrange(heatmap_logits, "b c h w -> (b h w) c", h=H, w=W)

        logits = logits_flattened[mask_flattened]
        core_idx = core_idx_flattened[mask_flattened]

        patch_logits = logits

        return patch_logits, core_idx


class ImageLevelClassificationLoss(nn.Module):
    def __init__(self, mode, class_weight=None):
        """
        Args:
            mode: "pca" or "cspca" - classification mode
            class_weight: Optional tensor of shape (num_classes,) to weight classes.
                         If None, computes balanced weights automatically from batch.
                         If "balanced", computes weights as: n_samples / (n_classes * np.bincount(y))
        """
        super().__init__()
        self.mode = mode
        self.class_weight = class_weight

    def forward(self, data):
        """
        Computes the image-level classification loss.
        """

        if "image_level_classification_outputs" not in data:
            return torch.tensor(0.0, device=data["label"].device)

        logits = data["image_level_classification_outputs"][0]

        if self.mode == "pca":
            labels = data["label"].to(logits.device)
        else:
            labels = (data["grade_group"] > 2).long().to(logits.device)

        # Compute class weights if needed
        weight = self.class_weight
        if weight == "balanced":
            # Compute balanced weights from current batch
            # weight[class] = n_samples / (n_classes * count[class])
            # For binary classification, we always expect 2 classes
            num_classes = logits.shape[-1]  # Get number of classes from logits
            unique_labels, counts = torch.unique(labels, return_counts=True)
            n_samples = labels.numel()
            
            # Initialize weights for all classes
            weight = torch.ones(num_classes, device=logits.device)
            
            # Update weights for classes present in the batch
            for i, label in enumerate(unique_labels):
                weight[label] = n_samples / (num_classes * counts[i].float())
        elif weight is not None and isinstance(weight, (list, tuple)):
            weight = torch.tensor(weight, device=logits.device, dtype=logits.dtype)
        elif weight is not None:
            # Assume it's already a tensor
            weight = weight.to(logits.device)

        # Compute the cross-entropy loss with optional class weights
        loss = F.cross_entropy(logits, labels, weight=weight)

        return loss


class SumLoss(nn.Module):
    def __init__(self, losses: list[nn.Module], weights=None):
        super().__init__()
        self.losses = nn.ModuleList(losses)
        self.weights = weights if weights is not None else [1.0] * len(losses)

    def forward(self, data):
        loss = self.losses[0](data) * self.weights[0]
        for i in range(1, len(self.losses)):
            loss += self.losses[i](data) * self.weights[i]
        return loss


class OutsideProstatePenaltyLoss(nn.Module):
    """
    Soft penalty for decoder predictions outside prostate.
    
    NOTE: This is a SOFT constraint - it penalizes but doesn't prevent
    predictions outside prostate. For a HARD constraint matching the 
    RL attention mask, use DecoderProstateMaskConstraint instead.
    """

    def forward(self, data: dict):
        cancer_logits = data["cancer_logits"]
        prostate_mask = data["prostate_mask"].to(cancer_logits.device)

        masks = []
        for i in range(len(cancer_logits)):
            mask = torch.ones(
                prostate_mask[i].shape, device=prostate_mask[i].device
            ).bool()
            mask &= prostate_mask[i] < 0.5
            masks.append(mask)

        masks = torch.stack(masks)
        predictions, batch_idx = MaskedPredictionModule()(cancer_logits, masks)

        loss = torch.nn.L1Loss()(predictions, torch.zeros_like(predictions))

        return loss


class DecoderProstateMaskConstraint(nn.Module):
    """
    HARD mask constraint for decoder predictions outside prostate.
    
    This matches the RL attention constraint approach:
    - RL attention: Sets logits to -inf → softmax gives 0
    - Decoder: Sets logits to large negative value → sigmoid gives ~0
    
    Unlike OutsideProstatePenaltyLoss which adds a soft L1 penalty,
    this class directly zeroes out logits outside the prostate,
    making predictions literally impossible outside prostate.
    
    Args:
        boundary_tolerance_pixels: Dilation radius for boundary tolerance (default: 0)
        negative_value: Large negative value to set outside prostate (default: -100)
    """
    
    def __init__(
        self, 
        boundary_tolerance_pixels: int = 0,
        negative_value: float = -100.0,
    ):
        super().__init__()
        self.boundary_tolerance_pixels = boundary_tolerance_pixels
        self.negative_value = negative_value
    
    def forward(self, data: dict) -> dict:
        """
        Apply hard mask to cancer_logits in place.
        
        Modifies data["cancer_logits"] to have large negative values
        outside the prostate mask, so sigmoid gives ~0.
        
        Returns:
            Modified data dict with masked cancer_logits
        """
        cancer_logits = data["cancer_logits"]
        prostate_mask = data["prostate_mask"].to(cancer_logits.device)
        
        B, C, H, W = cancer_logits.shape
        
        # Resize prostate mask to match logits if needed
        if prostate_mask.shape[-2:] != (H, W):
            prostate_mask = F.interpolate(
                prostate_mask.float(),
                size=(H, W),
                mode='nearest'
            )
        
        # Apply dilation for boundary tolerance if specified
        if self.boundary_tolerance_pixels > 0:
            kernel_size = 2 * self.boundary_tolerance_pixels + 1
            dilation_kernel = torch.ones(
                1, 1, kernel_size, kernel_size, 
                device=prostate_mask.device
            )
            prostate_mask = F.conv2d(
                prostate_mask,
                dilation_kernel,
                padding=self.boundary_tolerance_pixels
            )
            prostate_mask = (prostate_mask > 0).float()
        
        # Create mask for outside prostate (where mask < 0.5)
        outside_mask = prostate_mask < 0.5  # (B, 1, H, W)
        
        # Apply hard constraint: set logits to large negative value outside prostate
        # This makes sigmoid(logits) ≈ 0 outside prostate
        cancer_logits = torch.where(
            outside_mask.expand_as(cancer_logits),
            torch.full_like(cancer_logits, self.negative_value),
            cancer_logits
        )
        
        data["cancer_logits"] = cancer_logits
        
        return data


class AttentionAlignmentLoss(nn.Module):
    """
    Direct differentiable loss on the attention map — replaces RL reward with
    a per-pixel gradient signal.
    
    For cancer cores (involvement > 0):
      Encourage attention in the needle region proportional to involvement.
      loss = -involvement * log(mean_attn_in_needle + eps)
    
    For benign cores (involvement == 0):
      Penalize any attention in the needle region.
      loss = mean_attn_in_needle^2  (quadratic to be soft near zero)
    
    This gives the policy a SPATIAL supervision signal that the standard MIL
    loss does not: "where should you attend?" rather than "what's the label?"
    """
    
    def __init__(self, cancer_weight: float = 1.0, benign_weight: float = 1.0):
        super().__init__()
        self.cancer_weight = cancer_weight
        self.benign_weight = benign_weight
    
    def forward(self, data: dict):
        if 'rl_attention_map' not in data:
            return torch.tensor(0.0, device=data['cancer_logits'].device)
        
        attention_map = data['rl_attention_map']
        device = attention_map.device
        B = attention_map.shape[0]
        
        if attention_map.ndim == 4:
            attention = attention_map.squeeze(1)
        else:
            attention = attention_map
        
        needle_mask = data['needle_mask'].to(device)
        if needle_mask.shape[-2:] != attention.shape[-2:]:
            needle_mask = F.interpolate(
                needle_mask.float(), size=attention.shape[-2:], mode='nearest'
            )
        if needle_mask.ndim == 4:
            needle_mask = needle_mask.squeeze(1)
        
        prostate_mask = data.get('prostate_mask', None)
        if prostate_mask is not None:
            prostate_mask = prostate_mask.to(device)
            if prostate_mask.shape[-2:] != attention.shape[-2:]:
                prostate_mask = F.interpolate(
                    prostate_mask.float(), size=attention.shape[-2:], mode='nearest'
                )
            if prostate_mask.ndim == 4:
                prostate_mask = prostate_mask.squeeze(1)
        
        involvement = data['involvement'].to(device).float()
        if involvement.ndim > 1:
            involvement = involvement.squeeze(-1)
        
        eps = 1e-6
        loss = torch.tensor(0.0, device=device)
        
        for i in range(B):
            needle_i = needle_mask[i] > 0.5
            if prostate_mask is not None:
                valid_i = needle_i & (prostate_mask[i] > 0.5)
            else:
                valid_i = needle_i
            
            if valid_i.sum() == 0:
                continue
            
            mean_attn_needle = attention[i][valid_i].mean()
            inv_i = involvement[i].item()
            
            if inv_i > 0:
                target_attn = min(inv_i, 1.0)
                sample_loss = (mean_attn_needle - target_attn) ** 2
                loss = loss + self.cancer_weight * sample_loss
            else:
                loss = loss + self.benign_weight * (mean_attn_needle ** 2)
        
        return loss / max(B, 1)


class StochasticAttentionRegularizationLoss(nn.Module):
    """
    Stochastic Attention Regularization (SAR): adds noise to attention during
    supervised training and penalizes output inconsistency.
    
    Key idea: during training, run the decoder twice — once with clean attention,
    once with noisy attention — and add a consistency loss between outputs.
    This teaches the decoder to be ROBUST to attention variation, improving
    generalization without any RL.
    
    The loss is the KL divergence between the clean and noisy decoder outputs
    (applied to the needle region predictions).
    
    NOTE: This loss requires a special forward mode in the training loop that
    provides both 'cancer_logits' (clean) and 'cancer_logits_noisy' (noisy).
    If 'cancer_logits_noisy' is not present, returns 0 (graceful fallback).
    """
    
    def __init__(self, consistency_weight: float = 1.0):
        super().__init__()
        self.consistency_weight = consistency_weight
    
    def forward(self, data: dict):
        if 'cancer_logits_noisy' not in data:
            return torch.tensor(0.0, device=data['cancer_logits'].device)
        
        clean_logits = data['cancer_logits']
        noisy_logits = data['cancer_logits_noisy']
        device = clean_logits.device
        
        prostate_mask = data['prostate_mask'].to(device)
        needle_mask = data['needle_mask'].to(device)
        
        masks = (prostate_mask > 0.5) & (needle_mask > 0.5)
        
        clean_preds, batch_idx = MaskedPredictionModule()(clean_logits, masks)
        noisy_preds, _ = MaskedPredictionModule()(noisy_logits, masks)
        
        if len(clean_preds) == 0:
            return torch.tensor(0.0, device=device)
        
        clean_probs = clean_preds.sigmoid().clamp(1e-6, 1 - 1e-6)
        noisy_probs = noisy_preds.sigmoid().clamp(1e-6, 1 - 1e-6)
        
        kl_div = clean_probs * (clean_probs.log() - noisy_probs.log()) + \
                 (1 - clean_probs) * ((1 - clean_probs).log() - (1 - noisy_probs).log())
        
        return self.consistency_weight * kl_div.mean()


class DecoderProstateMaskLoss(nn.Module):
    """
    Wrapper that applies hard prostate mask constraint and then computes loss.
    
    This ensures the decoder predictions are hard-masked before any loss computation,
    matching the RL attention constraint approach.
    """
    
    def __init__(
        self,
        base_loss: nn.Module,
        boundary_tolerance_pixels: int = 0,
    ):
        super().__init__()
        self.mask_constraint = DecoderProstateMaskConstraint(
            boundary_tolerance_pixels=boundary_tolerance_pixels
        )
        self.base_loss = base_loss
    
    def forward(self, data: dict):
        # Apply hard mask constraint first
        data = self.mask_constraint(data)
        # Then compute loss with masked logits
        return self.base_loss(data)


def build_heatmap_loss(args):
    if args.loss == "needle_region_ce":
        return CancerDetectionValidRegionLoss()
    elif args.loss == "inv_l1":
        return InvolvementL1Loss(**args.loss_kw)
    elif args.loss == "inv_l1_v2":
        return InvolvementL1Loss(prostate_penalty=False)
    elif args.loss == "inv_mse":
        return InvolvementMSELoss(**args.loss_kw)
    elif args.loss == "mil_prop_bce":
        return CancerDetectionMILLoss()
    elif args.loss == "mil_prop_bce_l1_reg":
        return CancerDetectionMILLoss(
            base_loss=ProportionBCE(l1_penalty_lambda=0.001, pos_weight=1.0), 
            treat_gg1_as_benign=args.treat_gg1_as_benign
        )
    elif args.loss == "mil_prop_bce_entropy_reg":
        return CancerDetectionMILLoss(
            base_loss=ProportionBCE(entropy_penalty_lambda=0.01, pos_weight=1.0),
            treat_gg1_as_benign=args.get('treat_gg1_as_benign', False),
        )
    
    # =========================================================================
    # NOISE-ROBUST LOSS FUNCTIONS
    # =========================================================================
    
    # Top-K MIL Loss: Only supervise top k% pixels as cancer
    elif args.loss == "topk_mil":
        loss_kw = args.get('loss_kw', {})
        return TopKInvolvementMILLoss(
            pos_weight=args.get('pos_weight', 1.0),
            min_top_k_fraction=loss_kw.get('min_top_k_fraction', 0.05),
            use_soft_topk=loss_kw.get('use_soft_topk', False),
            temperature=loss_kw.get('temperature', 0.1),
            treat_gg1_as_benign=args.get('treat_gg1_as_benign', False),
            benign_weight=loss_kw.get('benign_weight', 1.0),  # How strongly to push non-topk to 0
        )
    
    elif args.loss == "topk_mil_soft":
        loss_kw = args.get('loss_kw', {})
        return TopKInvolvementMILLoss(
            pos_weight=args.get('pos_weight', 1.0),
            min_top_k_fraction=loss_kw.get('min_top_k_fraction', 0.05),
            use_soft_topk=True,  # Differentiable
            temperature=loss_kw.get('temperature', 0.1),
            treat_gg1_as_benign=args.get('treat_gg1_as_benign', False),
        )
    
    # Thresholded Involvement Loss: Compare fraction of pixels > 0.5 to involvement
    elif args.loss == "thresholded_involvement":
        loss_kw = args.get('loss_kw', {})
        return CancerDetectionMILLoss(
            base_loss=ThresholdedInvolvementLoss(
                threshold=loss_kw.get('threshold', 0.5),
                use_soft_threshold=loss_kw.get('use_soft_threshold', True),
                soft_temperature=loss_kw.get('soft_temperature', 0.1),
                pos_weight=args.get('pos_weight', 1.0),
                entropy_penalty_lambda=loss_kw.get('entropy_penalty_lambda', None),
            ),
            treat_gg1_as_benign=args.get('treat_gg1_as_benign', False),
        )
    
    elif args.loss == "thresholded_involvement_entropy_reg":
        loss_kw = args.get('loss_kw', {})
        return CancerDetectionMILLoss(
            base_loss=ThresholdedInvolvementLoss(
                threshold=loss_kw.get('threshold', 0.5),
                use_soft_threshold=True,
                soft_temperature=loss_kw.get('soft_temperature', 0.1),
                pos_weight=args.get('pos_weight', 1.0),
                entropy_penalty_lambda=0.01,
            ),
            treat_gg1_as_benign=args.get('treat_gg1_as_benign', False),
        )
    
    # Symmetric Cross Entropy: Noise-robust loss from Manifold DivideMix
    elif args.loss == "symmetric_ce":
        loss_kw = args.get('loss_kw', {})
        return CancerDetectionMILLoss(
            base_loss=SymmetricCrossEntropyLoss(
                alpha=loss_kw.get('alpha', 1.0),
                beta=loss_kw.get('beta', 1.0),
                pos_weight=args.get('pos_weight', 1.0),
                epsilon=loss_kw.get('epsilon', 1e-4),
                entropy_penalty_lambda=loss_kw.get('entropy_penalty_lambda', None),
            ),
            treat_gg1_as_benign=args.get('treat_gg1_as_benign', False),
        )
    
    elif args.loss == "symmetric_ce_entropy_reg":
        loss_kw = args.get('loss_kw', {})
        return CancerDetectionMILLoss(
            base_loss=SymmetricCrossEntropyLoss(
                alpha=loss_kw.get('alpha', 1.0),
                beta=loss_kw.get('beta', 1.0),
                pos_weight=args.get('pos_weight', 1.0),
                epsilon=loss_kw.get('epsilon', 1e-4),
                entropy_penalty_lambda=0.01,
            ),
            treat_gg1_as_benign=args.get('treat_gg1_as_benign', False),
        )
    
    # Combined: Top-K + Symmetric CE (most robust)
    elif args.loss == "topk_symmetric_ce":
        loss_kw = args.get('loss_kw', {})
        # This combines top-k selection with SCE base loss
        # For simplicity, we use TopKInvolvementMILLoss which internally uses BCE
        # and add SCE-style regularization through the entropy penalty
        return TopKInvolvementMILLoss(
            pos_weight=args.get('pos_weight', 1.0),
            min_top_k_fraction=loss_kw.get('min_top_k_fraction', 0.05),
            use_soft_topk=loss_kw.get('use_soft_topk', True),
            temperature=loss_kw.get('temperature', 0.1),
            treat_gg1_as_benign=args.get('treat_gg1_as_benign', False),
        )

    elif args.loss == "none":
        return None
    else:
        raise ValueError(f"Unknown loss function: {args.loss}")


def build_loss(args):

    losses = []

    hmap_loss = build_heatmap_loss(args)

    if hmap_loss is not None:
        # NOTE: Decoder prostate mask constraint is now applied in train_rl.py
        # (ProstNFoundMeta.forward) to avoid double masking.
        # The constraint uses the same boundary tolerance as RL attention.
        if args.get('decoder_prostate_mask_constraint', False):
            print("Decoder prostate mask constraint enabled (applied in forward pass, not loss).")
        losses.append(hmap_loss)

    if args.add_image_clf:
        print(f"Adding image-level classification loss: {args.add_image_clf}")
        class_weight = args.get('image_clf_class_weight', 'balanced')
        losses.append(ImageLevelClassificationLoss(mode=args.image_clf_mode, class_weight=class_weight))

    # Attention Alignment Loss: direct differentiable spatial attention supervision
    attn_align_weight = args.get('attention_alignment_weight', 0.0)
    if attn_align_weight > 0:
        print(f"Adding Attention Alignment Loss with weight={attn_align_weight}")
        attn_align_kw = args.get('attention_alignment_kw', {})
        losses.append(AttentionAlignmentLoss(
            cancer_weight=attn_align_kw.get('cancer_weight', 1.0) * attn_align_weight,
            benign_weight=attn_align_kw.get('benign_weight', 1.0) * attn_align_weight,
        ))

    # Stochastic Attention Regularization: consistency loss under attention noise
    sar_weight = args.get('sar_weight', 0.0)
    if sar_weight > 0:
        print(f"Adding Stochastic Attention Regularization with weight={sar_weight}")
        losses.append(StochasticAttentionRegularizationLoss(consistency_weight=sar_weight))

    # SOFT penalty (legacy approach - NOT recommended for consistency with RL)
    if args.outside_prostate_penalty:
        if args.get('decoder_prostate_mask_constraint', False):
            print("Warning: Both outside_prostate_penalty and decoder_prostate_mask_constraint enabled.")
            print("         The hard mask constraint is already applied. Soft penalty is redundant.")
        else:
            print("Adding outside prostate penalty loss (SOFT constraint).")
            print("         Consider using decoder_prostate_mask_constraint=true for HARD constraint matching RL.")
            losses.append(OutsideProstatePenaltyLoss())

    return SumLoss(losses)



def get_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--loss", default="needle_region_ce")
    parser.add_argument(
        "--outside_prostate_penalty",
        action="store_true",
        default=False,
        help="Whether to penalize the model for making predictions outside the prostate region.",
    )
    return parser
