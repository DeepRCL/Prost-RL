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
