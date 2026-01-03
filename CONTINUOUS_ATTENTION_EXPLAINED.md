# Continuous Patch-Level Attention - Detailed Explanation

## Overview

**Continuous mode** is an alternative to discrete patch selection where instead of choosing k specific patches, the policy generates a **soft attention distribution** over all patches and uses it to weight the feature maps.

## Architecture Flow

### 1. Policy Forward Pass (`_continuous_forward`)

```python
# Input: image_features (B, 256, 16, 16) - 16x16 patch grid

# Step 1: Process features
features = feature_processor(image_features)  # (B, 256, 16, 16)

# Step 2: Generate attention logits
attention_logits = attention_head(features)  # (B, 1, 16, 16)
attention_logits = attention_logits.view(B, 256)  # Flatten: (B, 256)

# Step 3: Apply prostate mask constraint (if enabled)
# Sets logits to -inf outside prostate region

# Step 4: Convert to probabilities
attention_probs = softmax(attention_logits, dim=1)  # (B, 256)
attention_probs = attention_probs.view(B, 1, 16, 16)  # Reshape back

# Step 5: Weight features by attention
weighted_features = features * attention_probs  # (B, 256, 16, 16)
#                    ↑              ↑
#              Original      Attention weights
#              features      (sums to 1 per image)
```

**Key Point**: Every patch gets a weight between 0 and 1, and all weights sum to 1.0 per image.

### 2. Decoder Conditioning (`prostnfound_rl_v2.py`)

```python
if discrete_attention:
    # DISCRETE MODE:
    # attention_features: (B, k, C) - k selected patches
    patch_embeddings = attention_projection(attention_features)  # (B, k, 256)
    sparse_embedding = concat([sparse_embedding, patch_embeddings], dim=1)
    # Result: Adds k new sparse prompt tokens (like point prompts)
    
else:
    # CONTINUOUS MODE:
    # attention_features: (B, C, H, W) - weighted features
    attention_modulation = attention_modulation(attention_features)  # (B, 256, 16, 16)
    dense_embedding = dense_embedding + attention_modulation
    # Result: Modulates dense embeddings spatially (like mask prompts)
```

## Visual Comparison

### Discrete Mode (k=3 patches selected)
```
Image Features (16×16 patches):
┌─────────────────────────┐
│ 0.1  0.1  0.1  0.1  ... │
│ 0.1  0.8  0.1  0.1  ... │  ← Patch selected (weight=1.0)
│ 0.1  0.1  0.1  0.1  ... │
│ 0.1  0.1  0.9  0.1  ... │  ← Patch selected (weight=1.0)
│ ...  ...  ...  ...  ... │
│ 0.1  0.7  0.1  0.1  ... │  ← Patch selected (weight=1.0)
└─────────────────────────┘

Output: 3 sparse embeddings (one per selected patch)
Decoder sees: 3 specific locations as "important"
```

### Continuous Mode (all patches weighted)
```
Image Features (16×16 patches):
┌─────────────────────────┐
│ 0.01 0.01 0.01 0.01 ... │
│ 0.01 0.15 0.01 0.01 ... │  ← Higher weight
│ 0.01 0.01 0.01 0.01 ... │
│ 0.01 0.01 0.12 0.01 ... │  ← Higher weight
│ ...  ...  ...  ...  ... │
│ 0.01 0.10 0.01 0.01 ... │  ← Higher weight
└─────────────────────────┘
  ↑
  All weights sum to 1.0

Output: Weighted feature map (B, 256, 16, 16)
Decoder sees: Spatial modulation of dense embeddings
```

## Mathematical Details

### Attention Distribution
```python
# Logits over all patches
logits = attention_head(features)  # (B, 256) - one logit per patch

# Softmax to get probability distribution
probs = softmax(logits, dim=1)  # (B, 256)
# Constraint: sum(probs, dim=1) = 1.0 for each image

# Weight features
weighted_features[b, c, h, w] = features[b, c, h, w] * probs[b, h*16+w]
```

### Policy Gradient Log Prob
```python
# For REINFORCE/GRPO, we need log_prob of the action
# In continuous mode, the "action" is the full distribution

log_probs = log_softmax(logits, dim=1)  # (B, 256)
entropy = -sum(probs * log_probs, dim=1)  # (B,)
policy_log_prob = -entropy  # Use negative entropy

# Why negative entropy?
# - Higher entropy = more uniform distribution = less "decisive"
# - Lower entropy = more peaked = more "decisive"
# - For REINFORCE, we want to reward decisive actions
```

## Decoder Conditioning Mechanism

### Discrete → Sparse Embeddings
```
MedSAM Decoder Input:
├─ sparse_embedding: (B, N_sparse, 256)
│  ├─ Base prompts (prostate mask, etc.)
│  └─ k patch embeddings ← Added here
│
└─ dense_embedding: (B, 256, 16, 16)
   └─ Original dense prompts
```

**How it works**: Decoder uses cross-attention to attend to sparse tokens. The k patch embeddings act like "point prompts" telling the decoder "these k patches are important."

### Continuous → Dense Embeddings
```
MedSAM Decoder Input:
├─ sparse_embedding: (B, N_sparse, 256)
│  └─ Base prompts only (no patch embeddings)
│
└─ dense_embedding: (B, 256, 16, 16)
   ├─ Original dense prompts
   └─ + attention_modulation ← Added here
```

**How it works**: Decoder uses the dense embeddings directly. The attention modulation spatially emphasizes important regions, similar to how a mask prompt would work.

## Advantages of Continuous Mode

### 1. **No Information Loss**
- Discrete: Only k patches contribute (others ignored)
- Continuous: All patches contribute proportionally

### 2. **Smoother Gradients**
- Discrete: Gradients only flow through k selected patches
- Continuous: Gradients flow through all patches (weighted by attention)

### 3. **More Flexible**
- Can learn to attend to multiple regions simultaneously
- No hard constraint on number of patches

### 4. **Better for Diffuse Patterns**
- If cancer is spread across many patches, continuous can weight all of them
- Discrete might miss some important patches

## Disadvantages of Continuous Mode

### 1. **Less Interpretable**
- Harder to visualize "which patches" (it's all patches with different weights)
- Discrete gives clear "top-k" patches

### 2. **Potentially Less Focused**
- May spread attention too thinly
- Discrete forces focus on k specific locations

### 3. **Different RL Dynamics**
- Uses entropy-based log_prob instead of categorical sampling
- May require different hyperparameters

## When to Use Continuous Mode

**Use Continuous When:**
- Cancer patterns are diffuse/spread out
- You want the model to consider all patches
- You prefer smooth, differentiable attention
- You want to avoid the "top-k selection" constraint

**Use Discrete When:**
- You want interpretable "top-k" patches
- Cancer is localized to specific regions
- You want to force the model to be decisive
- You prefer the traditional "point prompt" paradigm

## Training Considerations

### Reward Function
Both modes use the same reward function:
```python
reward = classification_reward + diversity_reward + boundary_penalty
```

The reward is computed from the final cancer prediction, not from the attention itself.

### Value Function (PPO)
```python
# Continuous mode value function
value_network(
    image_features,      # (B, 256, 16, 16) - state
    weighted_features,   # (B, 256, 16, 16) - action (weighted features)
    clinical_features,   # (B, 4) - clinical data
)
```

The value network pools the weighted features to get a fixed-size representation.

## Example Output Shapes

### Input
```python
image_features: (B, 256, 16, 16)  # 16×16 patch grid
clinical_features: (B, 4)  # Age, PSA, etc.
```

### Policy Output (Continuous)
```python
attention_features: (B, 256, 16, 16)  # Weighted features
log_probs: (B, 1)  # Negative entropy
attention_map: (B, 1, 16, 16)  # Attention heatmap (for visualization)
```

### Decoder Input (After Conditioning)
```python
sparse_embedding: (B, N_base, 256)  # No patch embeddings added
dense_embedding: (B, 256, 16, 16)  # Original + attention_modulation
```

## Code Locations

1. **Policy Implementation**: `medAI/medAI/modeling/rl_attention_policy_v3.py`
   - `_continuous_forward()` method (lines 176-196)

2. **Decoder Conditioning**: `medAI/medAI/modeling/prostnfound_rl_v2.py`
   - Lines 211-215 (continuous mode path)

3. **Value Network**: `medAI/medAI/modeling/rl_attention_policy_v3.py`
   - `ValueNetwork.forward()` for continuous (lines 388-390)

4. **Config**: `cfg/train/pnf_plus_rl_v2_continuous.yaml`
   - `discrete_attention: false`

## Summary

**Continuous mode** generates a **soft attention distribution** over all patches, weights the feature maps by this distribution, and modulates the decoder's dense embeddings spatially. It's like giving the decoder a "soft mask" of importance rather than "hard points" of importance.

This allows the model to:
- Consider all patches simultaneously
- Learn smooth attention patterns
- Avoid the constraint of selecting exactly k patches
- Potentially capture diffuse cancer patterns better

The trade-off is less interpretability (no clear "top-k" patches) and potentially less focused attention.
