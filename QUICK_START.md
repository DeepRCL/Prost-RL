# Quick Start - RL V2

## Run Experiments

### 1. Discrete Patch Attention (GRPO) - Recommended First Test
```bash
cd /home/mahdi.abootorabi/prostnfound/prostnfound
python train_rl.py cfg/train/pnf_plus_rl_v2_discrete.yaml
```

### 2. Continuous Attention (GRPO)
```bash
python train_rl.py cfg/train/pnf_plus_rl_v2_continuous.yaml
```

### 3. Discrete with PPO (Value Function)
```bash
python train_rl.py cfg/train/pnf_plus_rl_v2_ppo.yaml
```

## Key Differences from V1

| Feature | V1 | V2 |
|---------|----|----|
| Model | `prostnfound_rl_adapter_medsam_legacy` | `prostnfound_rl_v2_adapter_medsam_legacy` |
| Hidden Dim | 512 | 256 |
| Layers | 7 | 3-4 |
| Attention Output | Pixel coords | Patch features/attention |
| Value Function | f(image) | f(image, action, clinical) |
| Normalization | Within-batch | Within-image |

## Config Parameters

```yaml
model: prostnfound_rl_v2_adapter_medsam_legacy
model_kw:
  policy_hidden_dim: 256              # Architecture size
  discrete_attention: true            # true=discrete, false=continuous
  use_value_function: false           # false=GRPO, true=PPO
  num_attention_patches: 4            # How many patches

rl_reward_mode: classification_only   # Your best reward
rl_num_samples_per_image: 4          # Within-image comparison
rl_mode: grpo                         # 'grpo' or 'ppo'
```

## Files Created

### Core Implementation
- `medAI/medAI/modeling/rl_attention_policy_v3.py` - New policy
- `medAI/medAI/modeling/grpo_v2.py` - Fixed GRPO  
- `medAI/medAI/modeling/prostnfound_rl_v2.py` - New model

### Configs (Ready to Use)
- `cfg/train/pnf_plus_rl_v2_discrete.yaml` - Discrete + GRPO
- `cfg/train/pnf_plus_rl_v2_continuous.yaml` - Continuous + GRPO
- `cfg/train/pnf_plus_rl_v2_ppo.yaml` - Discrete + PPO

## What Changed

1. **Architecture**: Simpler (256 dim, 3-4 layers) to reduce overfitting
2. **Attention**: Patch-level features/heatmap (not fake pixel coords)
3. **Value Network**: Conditions on actions properly (PPO only)
4. **Normalization**: Within-image advantages (correct for multiple samples)
5. **Decoder Conditioning**: Proper sparse/dense embeddings

## Expected Results

- Better or same performance as V1
- Less overfitting (simpler model)
- Continuous attention as new baseline option
- Properly functioning PPO (if you want to try it)

## Check Installation

```bash
# Verify files compile
cd /home/mahdi.abootorabi/prostnfound
python -m py_compile medAI/medAI/modeling/rl_attention_policy_v3.py
python -m py_compile medAI/medAI/modeling/grpo_v2.py
python -m py_compile medAI/medAI/modeling/prostnfound_rl_v2.py
```

Should print nothing (success) ✓
