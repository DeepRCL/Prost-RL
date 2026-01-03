# Publication-Ready Improvements for ProstNFound-RL

## Executive Summary

This document outlines key improvements to strengthen the publication impact of the RL-based attention model for prostate cancer detection.

---

## ✅ Already Implemented

1. **Benign Sparsity Penalty** (`rl_benign_sparsity_penalty_weight`)
   - Penalizes high attention activation in benign cases
   - Encourages sparse attention for non-cancer samples
   - Config: `rl_benign_sparsity_penalty_weight: 0.5`

2. **RL-Specific Evaluation Metrics** (`prostnfound/src/rl_evaluation_metrics.py`)
   - Attention-Involvement Correlation (Spearman/Pearson)
   - Benign Attention Sparsity
   - Attention AUROC
   - Attention Calibration Error
   - Attention Contrast (cancer vs benign)

---

## 🎯 High Priority Improvements

### 1. Attention-Guided Contrastive Learning

**Problem**: The RL agent learns to attend to "any" region that helps classification, not necessarily the actual cancer region.

**Solution**: Add a contrastive component that:
- Encourages attention in cancer cases to be similar to decoder cancer predictions
- Discourages attention from being uniform/spread out

```yaml
# Proposed new config option
rl_attention_contrastive_weight: 0.3  # Match attention to decoder heatmap for cancer cases
```

### 2. Involvement-Proportional Attention Reward

**Problem**: Current reward only cares about classification, not attention quality.

**Solution**: Add reward that scales with how well attention intensity matches true involvement.

```python
# In rl_loss.py
def compute_involvement_proportional_reward(self, attention_map, data):
    """
    Reward attention that is proportional to true involvement.
    
    For involvement=0.7, reward attention that averages ~0.7
    For involvement=0.0, reward attention that averages ~0.0
    """
    involvement = data['involvement']
    mean_attention = attention_map.mean(dim=(1,2))
    
    # Reward = 1 - |mean_attention - involvement|
    reward = 1.0 - torch.abs(mean_attention - involvement)
    return reward
```

### 3. Multi-Scale Attention

**Problem**: Single-scale attention may miss cancer regions of varying sizes.

**Solution**: Add multi-scale attention branches in the policy network.

```python
# In rl_attention_policy_v3.py
class MultiScaleAttentionPolicy(nn.Module):
    def __init__(self, ...):
        self.attention_scales = nn.ModuleList([
            nn.Conv2d(in_ch, 1, kernel_size=k, padding=k//2)
            for k in [1, 3, 5]  # Multi-scale kernels
        ])
    
    def forward(self, features):
        attentions = [scale(features) for scale in self.attention_scales]
        combined = sum(attentions) / len(attentions)
        return combined.sigmoid()
```

---

## 📊 Evaluation Improvements

### 4. Ablation Study Metrics

For a strong paper, you need comprehensive ablations. Track these:

| Ablation | What to Measure |
|----------|-----------------|
| Without RL | Baseline ProstNFound performance |
| Without Benign Penalty | Effect of sparsity constraint |
| Without Classification Head | RL on heatmap only |
| Different RL algorithms | GRPO vs PPO vs REINFORCE |
| Different reward modes | `classification_only` vs `combined_v2` vs `attention_proportional` |

### 5. Clinical Relevance Metrics

Add clinically meaningful metrics:

```python
# Sensitivity at high specificity thresholds
'sens_at_90_spec'  # Conservative threshold
'sens_at_95_spec'  # Very conservative

# Negative Predictive Value
'npv_at_80_spec'

# Detection rate for csPCa (Grade Group >= 2)
'cspca_detection_rate'

# False negative rate for high involvement cases
'fn_rate_high_involvement'
```

### 6. Visualization for Paper

Key visualizations needed:

1. **Attention Heatmap Comparison**
   - Side-by-side: RL attention vs Decoder heatmap vs Ground truth
   - Show benign cases with low attention vs cancer cases with focused attention

2. **Attention-Involvement Scatter Plot**
   - X-axis: True involvement
   - Y-axis: Mean attention
   - Color: Cancer/Benign
   - Show correlation coefficient

3. **ROC Curves**
   - Compare: Baseline vs RL-enhanced
   - Show both classification head and heatmap head

4. **Calibration Plot**
   - Reliability diagram for attention predictions

---

## 🔬 Training Improvements

### 7. Curriculum Learning

**Problem**: Hard cases early in training can destabilize RL.

**Solution**: Start with easy cases, progressively add harder ones.

```yaml
# Proposed config
rl_curriculum_learning: true
rl_curriculum_warmup_epochs: 10  # Start with high involvement cases only
rl_curriculum_schedule: 'linear'  # or 'cosine'
```

### 8. Experience Replay / Self-Play

**Problem**: RL benefits from diverse experiences.

**Solution**: Store and replay good/bad attention patterns.

### 9. Uncertainty-Guided Attention

**Problem**: Model should attend more to uncertain regions.

**Solution**: Incorporate decoder uncertainty into policy.

```python
# Weight attention by uncertainty
uncertainty = decoder_logits.sigmoid() * (1 - decoder_logits.sigmoid())
attention_weighted = attention * uncertainty
```

---

## 📈 Experimental Design for Paper

### Required Experiments

1. **Main Results Table**
   | Model | Core AUROC | csPCa AUROC | Heatmap AUROC | Sens@80%Spec |
   |-------|------------|-------------|---------------|--------------|
   | ProstNFound (Baseline) | - | - | - | - |
   | + RL Attention | - | - | - | - |
   | + Benign Sparsity | - | - | - | - |
   | + Attention Proportional | - | - | - | - |

2. **Cross-Center Generalization**
   - Train on CRCEO, test on UVA (or vice versa)
   - Show RL attention generalizes better

3. **K-Fold Cross Validation**
   - 5-fold CV with mean ± std
   - Statistical significance tests (p-values)

4. **Qualitative Analysis**
   - Expert radiologist evaluation of attention maps
   - "Does attention highlight clinically relevant regions?"

---

## 🚀 Quick Wins (Easy to Implement)

1. ✅ **Benign Sparsity Penalty** - Done!
2. ✅ **RL Evaluation Metrics** - Done!
3. 🔲 **Attention Logging** - Add to wandb
4. 🔲 **Confusion Matrix by Involvement Level**
5. 🔲 **Attention Overlay Visualizations**

---

## Config Recommendations

### Best Current Config
```yaml
# Best known config for RL
rl_mode: grpo
rl_reward_mode: classification_only
rl_benign_sparsity_penalty_weight: 0.5
rl_num_samples_per_image: 4
rl_entropy_coef: 0.005
rl_cspca_bonus: 2.0
image_clf_class_weight: balanced
```

### Experimental Configs to Try
```yaml
# Experiment 1: Stronger benign penalty
rl_benign_sparsity_penalty_weight: 1.0

# Experiment 2: Combined reward
rl_reward_mode: combined_v2
rl_heatmap_reward_weight: 0.3
rl_classification_reward_weight: 0.7

# Experiment 3: PPO instead of GRPO
rl_mode: ppo
model_kw:
  use_value_function: true
```

---

## Summary: Priority Order

| Priority | Improvement | Impact | Effort |
|----------|-------------|--------|--------|
| 1 | ✅ Benign Sparsity Penalty | High | Low |
| 2 | ✅ RL Evaluation Metrics | High | Medium |
| 3 | 🔲 Involvement-Proportional Reward | High | Medium |
| 4 | 🔲 Attention Visualizations | High | Low |
| 5 | 🔲 Ablation Study Framework | Medium | Medium |
| 6 | 🔲 Cross-Center Validation | High | High |
| 7 | 🔲 Multi-Scale Attention | Medium | High |
| 8 | 🔲 Curriculum Learning | Medium | Medium |

---

## Next Steps

1. Run experiments with benign sparsity penalty
2. Evaluate using new RL metrics
3. Generate visualizations for paper
4. Implement involvement-proportional reward
5. Run ablation studies
6. Prepare cross-center validation
