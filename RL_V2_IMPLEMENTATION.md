# RL V2 Implementation - Patch-Level Attention

## What Was Implemented

### 1. New Policy Architectures (`rl_attention_policy_v3.py`)

**PatchAttentionPolicy**: 
- Simplified architecture (256 hidden dim, 2-3 layers)
- Two modes:
  - `discrete_mode=True`: Sample k patches from distribution
  - `discrete_mode=False`: Continuous attention over all patches
- Proper patch-level output (no pixel conversion)
- Reduced parameters to prevent overfitting

**ValueNetwork**:
- Separate value network for PPO
- Takes both state (image features) AND action (selected patches)
- Properly conditions on policy outputs
- Lightweight architecture

### 2. Updated GRPO (`grpo_v2.py`)

**GRPO_V2**:
- Fixed within-image advantage normalization (not within-batch!)
- Properly handles both GRPO and PPO modes
- Support for continuous and discrete actions
- Cleaner separation of concerns

### 3. New Model Architecture (`prostnfound_rl_v2.py`)

**ProstNFoundRLV2**:
- Discrete mode: Adds selected patch features as sparse embeddings to decoder
- Continuous mode: Modulates dense embeddings with attention heatmap
- No fake pixel coordinate conversion
- Proper integration with MedSAM decoder
- Optional value network for PPO

### 4. Configuration Files

**Three configs created:**

1. `pnf_plus_rl_v2_discrete.yaml` - Discrete patch selection with GRPO
2. `pnf_plus_rl_v2_continuous.yaml` - Continuous attention with GRPO
3. `pnf_plus_rl_v2_ppo.yaml` - Discrete with PPO (uses value function)

## Key Changes from V1

| Aspect | V1 | V2 |
|--------|----|----|
| Architecture | 512 hidden, 7 layers | 256 hidden, 3-4 layers |
| Value function | Conditions on image only | Conditions on image + action |
| Attention output | Fake pixel coordinates | Patch-level features/attention |
| Decoder conditioning | Point prompts (via SAM) | Sparse/dense embeddings directly |
| Advantage normalization | Within-batch | Within-image (correct!) |
| Overfitting risk | High | Low |

## How to Use

### Train with Discrete Patch Attention (GRPO)
```bash
python train_rl.py cfg/train/pnf_plus_rl_v2_discrete.yaml
```

### Train with Continuous Attention (GRPO)
```bash
python train_rl.py cfg/train/pnf_plus_rl_v2_continuous.yaml
```

### Train with PPO (with value function)
```bash
python train_rl.py cfg/train/pnf_plus_rl_v2_ppo.yaml
```

## Key Hyperparameters

```yaml
model: prostnfound_rl_v2_adapter_medsam_legacy
model_kw:
  num_attention_patches: 4        # Number of patches to select
  policy_hidden_dim: 256          # Reduced from 512
  discrete_attention: true        # true=discrete, false=continuous
  use_value_function: false       # false=GRPO, true=PPO
  use_prostate_mask_constraint: true

rl_reward_mode: classification_only  # Best performer
rl_num_samples_per_image: 4         # Within-image comparison
rl_mode: grpo                       # or 'ppo'
```

## Architecture Summary

### Discrete Mode
```
Image → Encoder → Policy → Sample k patches → Extract patch features
                                            ↓
                              Sparse embeddings (B, k, 256)
                                            ↓
                              MedSAM Decoder → Output
```

### Continuous Mode
```
Image → Encoder → Policy → Attention heatmap (B, 1, H, W)
                                            ↓
                              Weighted features (B, C, H, W)
                                            ↓
                              Dense modulation → MedSAM Decoder → Output
```

### Value Network (PPO only)
```
Image features + Action features + Clinical → Value estimate (B,)
```

## Files Modified

1. `medAI/medAI/modeling/rl_attention_policy_v3.py` - New policy
2. `medAI/medAI/modeling/grpo_v2.py` - Fixed GRPO
3. `medAI/medAI/modeling/prostnfound_rl_v2.py` - New model
4. `medAI/medAI/modeling/__init__.py` - Added V2 imports
5. `prostnfound/train_rl.py` - Updated to support V2
6. `prostnfound/cfg/train/pnf_plus_rl_v2_*.yaml` - New configs

## Testing Recommendations

1. Test discrete vs continuous attention
2. Compare GRPO vs PPO
3. Verify within-image advantage normalization works
4. Check that value network properly conditions on actions (PPO only)
5. Compare to old V1 implementation

## Expected Improvements

- Less overfitting (simpler architecture)
- Better decoder conditioning (patch-level vs fake pixels)
- Correct advantage normalization (within-image)
- Proper value estimation (conditions on actions)
- Can test continuous attention (new capability)
