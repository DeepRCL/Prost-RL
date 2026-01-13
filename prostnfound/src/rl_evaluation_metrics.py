# RL-Specific Evaluation Metrics for Publication
# Add these to evaluator.py for comprehensive RL paper evaluation

import numpy as np
import torch
from typing import Dict, Tuple
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score


class RLAttentionEvaluator:
    """
    Specialized evaluator for RL attention model.
    
    Key metrics for publication:
    1. Attention-Involvement Correlation (AIC)
    2. Benign Attention Sparsity (BAS)
    3. Cancer Detection Improvement (CDI)
    4. Attention Precision/Recall
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.attention_maps = []
        self.involvement_values = []
        self.labels = []
        self.cancer_logits = []
        self.prostate_masks = []
        self.needle_masks = []
    
    def update(self, 
               attention_map: torch.Tensor,
               involvement: torch.Tensor,
               label: torch.Tensor,
               cancer_logits: torch.Tensor,
               prostate_mask: torch.Tensor = None,
               needle_mask: torch.Tensor = None):
        """
        Update with batch data.
        
        Args:
            attention_map: RL attention map (B, H, W) or (B, 1, H, W)
            involvement: Ground truth involvement (B,)
            label: Cancer label (B,) 
            cancer_logits: Decoder cancer logits (B, 1, H, W)
            prostate_mask: Prostate mask (B, 1, H, W)
            needle_mask: Needle mask (B, 1, H, W)
        """
        if attention_map.ndim == 4:
            attention_map = attention_map.squeeze(1)
        
        self.attention_maps.append(attention_map.detach().cpu())
        self.involvement_values.append(involvement.detach().cpu())
        self.labels.append(label.detach().cpu())
        self.cancer_logits.append(cancer_logits.detach().cpu())
        
        if prostate_mask is not None:
            self.prostate_masks.append(prostate_mask.detach().cpu())
        if needle_mask is not None:
            self.needle_masks.append(needle_mask.detach().cpu())
    
    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute all RL-specific metrics.
        
        Returns:
            Dictionary of metrics
        """
        attention_maps = torch.cat(self.attention_maps, dim=0).numpy()
        involvement = torch.cat(self.involvement_values, dim=0).numpy()
        labels = torch.cat(self.labels, dim=0).numpy()
        
        B = attention_maps.shape[0]
        
        metrics = {}
        
        # ============================================================
        # 1. ATTENTION-INVOLVEMENT CORRELATION (AIC)
        # Key metric: How well does attention intensity correlate with
        # true cancer involvement?
        # ============================================================
        mean_attention = attention_maps.mean(axis=(1, 2))  # (B,)
        
        # Spearman correlation (rank-based, robust)
        corr, p_value = spearmanr(mean_attention, involvement)
        metrics['rl/attention_involvement_correlation_spearman'] = corr
        metrics['rl/attention_involvement_correlation_pvalue'] = p_value
        
        # Pearson correlation (linear)
        corr_pearson, _ = pearsonr(mean_attention, involvement)
        metrics['rl/attention_involvement_correlation_pearson'] = corr_pearson
        
        # ============================================================
        # 2. BENIGN ATTENTION SPARSITY (BAS)
        # Key metric: Is attention sparse/low in benign cases?
        # Lower is better for benign cases
        # ============================================================
        benign_mask = labels == 0
        cancer_mask = labels == 1
        
        if benign_mask.sum() > 0:
            benign_mean_attention = mean_attention[benign_mask].mean()
            metrics['rl/benign_mean_attention'] = float(benign_mean_attention)
            
            # Sparsity: % of attention below threshold (e.g., 0.1)
            SPARSITY_THRESHOLD = 0.1
            benign_sparsity = (attention_maps[benign_mask] < SPARSITY_THRESHOLD).mean()
            metrics['rl/benign_attention_sparsity'] = float(benign_sparsity)
        
        if cancer_mask.sum() > 0:
            cancer_mean_attention = mean_attention[cancer_mask].mean()
            metrics['rl/cancer_mean_attention'] = float(cancer_mean_attention)
        
        # Attention Contrast: cancer mean - benign mean (higher is better)
        if benign_mask.sum() > 0 and cancer_mask.sum() > 0:
            attention_contrast = cancer_mean_attention - benign_mean_attention
            metrics['rl/attention_contrast'] = float(attention_contrast)
        
        # ============================================================
        # 3. ATTENTION AUROC
        # Can mean attention distinguish cancer from benign?
        # ============================================================
        if len(np.unique(labels)) == 2:
            attention_auroc = roc_auc_score(labels, mean_attention)
            metrics['rl/attention_auroc'] = attention_auroc
        
        # ============================================================
        # 4. ATTENTION CALIBRATION
        # How well calibrated is attention to involvement?
        # ============================================================
        # Bin samples by involvement and compute mean attention per bin
        involvement_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        for i in range(len(involvement_bins) - 1):
            low, high = involvement_bins[i], involvement_bins[i + 1]
            bin_mask = (involvement >= low) & (involvement < high)
            if bin_mask.sum() > 0:
                bin_mean_attention = mean_attention[bin_mask].mean()
                metrics[f'rl/attention_at_involvement_{low:.1f}_{high:.1f}'] = float(bin_mean_attention)
        
        # Calibration error: |mean_attention - involvement|
        calibration_error = np.abs(mean_attention - involvement).mean()
        metrics['rl/attention_calibration_error'] = float(calibration_error)
        
        # ============================================================
        # 5. HIGH INVOLVEMENT PERFORMANCE
        # Performance specifically on csPCa (involvement > 0.4)
        # ============================================================
        high_inv_mask = involvement > 0.4
        benign_or_high_inv = benign_mask | high_inv_mask
        
        if benign_or_high_inv.sum() > 0 and len(np.unique(labels[benign_or_high_inv])) == 2:
            high_inv_auroc = roc_auc_score(
                labels[benign_or_high_inv], 
                mean_attention[benign_or_high_inv]
            )
            metrics['rl/attention_auroc_high_involvement'] = high_inv_auroc
        
        # ============================================================
        # 6. ATTENTION ENTROPY (exploration diversity)
        # Does the model explore diverse regions or collapse?
        # ============================================================
        # Normalize attention to probability distribution per sample
        attention_flat = attention_maps.reshape(B, -1)
        attention_probs = attention_flat / (attention_flat.sum(axis=1, keepdims=True) + 1e-8)
        entropy = -(attention_probs * np.log(attention_probs + 1e-8)).sum(axis=1)
        max_entropy = np.log(attention_probs.shape[1])  # uniform distribution entropy
        normalized_entropy = entropy / max_entropy
        
        metrics['rl/attention_entropy_mean'] = float(normalized_entropy.mean())
        metrics['rl/attention_entropy_std'] = float(normalized_entropy.std())
        
        # ============================================================
        # 7. NEEDLE FOCUS RATIO (Key Interpretability Metric)
        # For CANCER cases: is attention higher INSIDE needle than OUTSIDE?
        # This validates that RL attention is focusing on the right region.
        # 
        # Ratio > 1.0 means attention correctly focuses on needle/tumor
        # Ratio < 1.0 means attention is scattered or in wrong places
        # ============================================================
        if len(self.needle_masks) > 0 and len(self.prostate_masks) > 0:
            needle_masks = torch.cat(self.needle_masks, dim=0).numpy()
            prostate_masks = torch.cat(self.prostate_masks, dim=0).numpy()
            
            if needle_masks.ndim == 4:
                needle_masks = needle_masks.squeeze(1)
            if prostate_masks.ndim == 4:
                prostate_masks = prostate_masks.squeeze(1)
            
            # Resize masks if needed
            if needle_masks.shape[-2:] != attention_maps.shape[-2:]:
                import torch.nn.functional as F
                needle_masks = F.interpolate(
                    torch.from_numpy(needle_masks).unsqueeze(1).float(),
                    size=attention_maps.shape[-2:],
                    mode='nearest'
                ).squeeze(1).numpy()
                prostate_masks = F.interpolate(
                    torch.from_numpy(prostate_masks).unsqueeze(1).float(),
                    size=attention_maps.shape[-2:],
                    mode='nearest'
                ).squeeze(1).numpy()
            
            # Compute for CANCER cases only
            needle_focus_ratios = []
            for i in range(B):
                if labels[i] == 0:  # Skip benign
                    continue
                
                attn_i = attention_maps[i]
                needle_i = needle_masks[i] > 0.5
                prostate_i = prostate_masks[i] > 0.5
                outside_needle = prostate_i & (~needle_i)
                
                if needle_i.sum() > 0 and outside_needle.sum() > 0:
                    mean_inside = attn_i[needle_i].mean()
                    mean_outside = attn_i[outside_needle].mean()
                    if mean_outside > 1e-6:
                        ratio = mean_inside / mean_outside
                        needle_focus_ratios.append(ratio)
            
            if len(needle_focus_ratios) > 0:
                mean_ratio = np.mean(needle_focus_ratios)
                metrics['rl/cancer_needle_focus_ratio'] = float(mean_ratio)
                metrics['rl/cancer_needle_focus_correct_pct'] = float(
                    np.mean([r > 1.0 for r in needle_focus_ratios]) * 100
                )
        
        return metrics
    
    def get_visualization_data(self) -> Dict[str, np.ndarray]:
        """
        Get data for visualization/plotting.
        
        Returns:
            Dictionary with arrays for plotting
        """
        attention_maps = torch.cat(self.attention_maps, dim=0).numpy()
        involvement = torch.cat(self.involvement_values, dim=0).numpy()
        labels = torch.cat(self.labels, dim=0).numpy()
        
        mean_attention = attention_maps.mean(axis=(1, 2))
        
        return {
            'mean_attention': mean_attention,
            'involvement': involvement,
            'labels': labels,
            'attention_maps': attention_maps,
        }


def compute_attention_localization_metrics(
    attention_map: np.ndarray,
    cancer_mask: np.ndarray,
    prostate_mask: np.ndarray = None,
) -> Dict[str, float]:
    """
    Compute localization metrics for attention vs ground truth cancer mask.
    
    This is useful if you have pixel-level cancer annotations.
    
    Args:
        attention_map: Attention map (H, W)
        cancer_mask: Binary cancer mask (H, W)
        prostate_mask: Binary prostate mask (H, W), optional
        
    Returns:
        Localization metrics
    """
    metrics = {}
    
    # Binarize attention at various thresholds
    thresholds = [0.1, 0.25, 0.5, 0.75]
    
    for thresh in thresholds:
        attention_binary = attention_map > thresh
        
        # IoU (Intersection over Union)
        intersection = (attention_binary & cancer_mask).sum()
        union = (attention_binary | cancer_mask).sum()
        iou = intersection / (union + 1e-8)
        metrics[f'attention_iou_t{int(thresh*100)}'] = float(iou)
        
        # Precision: what fraction of attention is on cancer?
        if attention_binary.sum() > 0:
            precision = (attention_binary & cancer_mask).sum() / attention_binary.sum()
            metrics[f'attention_precision_t{int(thresh*100)}'] = float(precision)
        
        # Recall: what fraction of cancer is attended to?
        if cancer_mask.sum() > 0:
            recall = (attention_binary & cancer_mask).sum() / cancer_mask.sum()
            metrics[f'attention_recall_t{int(thresh*100)}'] = float(recall)
    
    return metrics
