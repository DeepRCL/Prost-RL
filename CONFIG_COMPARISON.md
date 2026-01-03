# RL V2 Config Files - Complete Comparison

## Overview

5 configuration files for testing different aspects of the RL V2 architecture:

1. **`pnf_plus_rl_v2_discrete.yaml`** - Baseline (recommended starting point)
2. **`pnf_plus_rl_v2_discrete_noprompts.yaml`** - No clinical prompts
3. **`pnf_plus_rl_v2_discrete_complex.yaml`** - Larger policy (512 hidden)
4. **`pnf_plus_rl_v2_continuous.yaml`** - Continuous attention mode
5. **`pnf_plus_rl_v2_ppo.yaml`** - PPO with value function

---

## Detailed Comparison Table

| Feature | Discrete (Baseline) | No Prompts | Complex | Continuous | PPO |
|---------|---------------------|------------|---------|------------|-----|
| **Config File** | `pnf_plus_rl_v2_discrete.yaml` | `pnf_plus_rl_v2_discrete_noprompts.yaml` | `pnf_plus_rl_v2_discrete_complex.yaml` | `pnf_plus_rl_v2_continuous.yaml` | `pnf_plus_rl_v2_ppo.yaml` |
| **Name** | `PNF-RL-V2-discrete-classification_only-GRPO` | `PNF-RL-V2-discrete-noprompts-classification_only` | `PNF-RL-V2-discrete-complex-classification_only` | `PNF-RL-V2-continuous-classification_only` | `PNF-RL-V2-discrete-PPO-classification_only` |
| | | | | | |
| **Attention Mode** | Discrete | Discrete | Discrete | **Continuous** | Discrete |
| `discrete_attention` | `true` | `true` | `true` | **`false`** | `true` |
| **What it does** | Selects k=4 patches | Selects k=4 patches | Selects k=4 patches | **Weights all patches** | Selects k=4 patches |
| | | | | | |
| **RL Algorithm** | GRPO | GRPO | GRPO | GRPO | **PPO** |
| `rl_mode` | `grpo` | `grpo` | `grpo` | `grpo` | **`ppo`** |
| `use_value_function` | `false` | `false` | `false` | `false` | **`true`** |
| **Value Network** | None | None | None | None | **Separate V(s,a)** |
| | | | | | |
| **Policy Architecture** | | | | | |
| `policy_hidden_dim` | **256** | **256** | **512** | **256** | **256** |
| **Complexity** | Simple (default) | Simple | **More complex** | Simple | Simple |
| **Params** | ~200K | ~150K | ~400K | ~200K | ~250K |
| | | | | | |
| **Clinical Features** | | | | | |
| `prompts` | `[age, psa, psad, pos]` | **`[]`** | `[age, psa, psad, pos]` | `[age, psa, psad, pos]` | `[age, psa, psad, pos]` |
| `use_clinical_in_policy` | `true` | **`false`** | `true` | `true` | `true` |
| **Clinical in decoder** | Yes (4 prompts) | **No** | Yes (4 prompts) | Yes (4 prompts) | Yes (4 prompts) |
| **Clinical in policy** | Yes | **No** | Yes | Yes | Yes |
| | | | | | |
| **Training** | | | | | |
| `epochs` | **30** | 25 | 25 | 25 | 25 |
| `lr` | `2.0e-05` | `2.0e-05` | `2.0e-05` | `2.0e-05` | **`3.0e-05`** |
| `rl_num_samples_per_image` | 4 | 4 | 4 | 4 | 4 |
| `rl_loss_weight` | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 |
| | | | | | |
| **Use Case** | Baseline comparison | Test without prompts | Test larger capacity | Test continuous attention | Test PPO vs GRPO |

---

## 1. Baseline: `pnf_plus_rl_v2_discrete.yaml`

**Purpose**: Recommended starting point for all experiments.

**Key Features**:
- ✅ Discrete patch selection (k=4)
- ✅ GRPO algorithm (no value function)
- ✅ Simple policy (256 hidden dim)
- ✅ Clinical prompts enabled (age, PSA, PSAD, pos)
- ✅ 30 epochs (longer training)

**When to use**:
- First experiment to run
- Baseline for comparing other configs
- Standard setup with all features enabled

**Expected behavior**:
- Selects 4 most important patches per image
- Uses clinical features to guide attention
- GRPO normalizes advantages within-image

---

## 2. No Prompts: `pnf_plus_rl_v2_discrete_noprompts.yaml`

**Purpose**: Test if RL attention works without clinical features.

**Key Differences from Baseline**:
- ❌ **No clinical prompts**: `prompts: []`
- ❌ **No clinical in policy**: `use_clinical_in_policy: false`
- ✅ Same discrete attention (k=4)
- ✅ Same GRPO algorithm
- ✅ Same simple policy (256 hidden)

**When to use**:
- Test if clinical features are necessary
- Compare with/without prompts
- Understand contribution of clinical data

**Expected behavior**:
- Policy learns attention from image features only
- No clinical conditioning in decoder or policy
- May perform worse if clinical features are important

---

## 3. Complex: `pnf_plus_rl_v2_discrete_complex.yaml`

**Purpose**: Test if larger policy capacity improves performance.

**Key Differences from Baseline**:
- ✅ **Larger policy**: `policy_hidden_dim: 512` (vs 256)
- ✅ Same discrete attention (k=4)
- ✅ Same GRPO algorithm
- ✅ Same clinical prompts

**When to use**:
- Test if policy is underfitting
- Compare simple vs complex architectures
- If baseline overfits, this will overfit more

**Expected behavior**:
- More parameters (~400K vs ~200K)
- Potentially better capacity for complex patterns
- Higher risk of overfitting on small datasets
- Value network stays at 256 (doesn't scale)

---

## 4. Continuous: `pnf_plus_rl_v2_continuous.yaml`

**Purpose**: Test continuous attention mode (soft weighting vs hard selection).

**Key Differences from Baseline**:
- ✅ **Continuous attention**: `discrete_attention: false`
- ✅ Weights ALL patches (not just k=4)
- ✅ Same GRPO algorithm
- ✅ Same simple policy (256 hidden)
- ✅ Same clinical prompts

**When to use**:
- Test if soft attention works better than hard selection
- Compare discrete vs continuous paradigms
- Test on diffuse cancer patterns

**Expected behavior**:
- Generates attention distribution over all 256 patches
- Weights features by attention probabilities
- Modulates dense embeddings (not sparse)
- May capture diffuse patterns better
- Less interpretable (no clear "top-k")

---

## 5. PPO: `pnf_plus_rl_v2_ppo.yaml`

**Purpose**: Test PPO algorithm with proper value function vs GRPO.

**Key Differences from Baseline**:
- ✅ **PPO algorithm**: `rl_mode: ppo`
- ✅ **Value function**: `use_value_function: true`
- ✅ Separate value network V(s, a)
- ✅ Higher learning rate: `lr: 3.0e-05` (vs 2.0e-05)
- ✅ Same discrete attention (k=4)
- ✅ Same simple policy (256 hidden)

**When to use**:
- Compare PPO vs GRPO performance
- Test if value function helps
- Standard RL algorithm comparison

**Expected behavior**:
- Value network estimates V(state, action)
- Uses value for advantage estimation (A = R - V)
- PPO clips policy updates
- May be more stable than GRPO
- More parameters (~250K vs ~200K)

---

## Quick Decision Tree

```
Start here: Baseline (discrete.yaml)
    │
    ├─ Want to test without clinical features?
    │  └─→ Use: noprompts.yaml
    │
    ├─ Want to test larger capacity?
    │  └─→ Use: complex.yaml
    │
    ├─ Want to test continuous attention?
    │  └─→ Use: continuous.yaml
    │
    └─ Want to test PPO vs GRPO?
       └─→ Use: ppo.yaml
```

---

## Common Settings (All Configs)

These are the same across all 5 configs:

| Setting | Value | Purpose |
|---------|-------|---------|
| `model` | `prostnfound_rl_v2_adapter_medsam_legacy` | V2 model with patch attention |
| `num_attention_patches` | `4` | Number of patches (discrete mode) |
| `use_prostate_mask_constraint` | `true` | Constrain attention to prostate |
| `rl_reward_mode` | `classification_only` | Reward based on classification |
| `rl_num_samples_per_image` | `4` | Within-image comparison |
| `rl_num_update_epochs` | `2` | PPO/GRPO update epochs |
| `rl_clip_eps` | `0.1` | Policy clipping |
| `rl_entropy_coef` | `0.005` | Entropy regularization |
| `batch_size` | `16` | Training batch size |
| `image_size` | `256` | Input image size |

---

## Expected Results Comparison

### Performance Ranking (Hypothesis)

1. **Baseline** (discrete + prompts): Good baseline
2. **PPO**: Potentially better (value function helps)
3. **Complex**: Better if underfitting, worse if overfitting
4. **Continuous**: May work better for diffuse patterns
5. **No Prompts**: Likely worse (no clinical guidance)

### Training Characteristics

| Config | Training Speed | Memory | Stability |
|--------|---------------|--------|-----------|
| Baseline | Fast | Low | High |
| No Prompts | Fast | Low | High |
| Complex | Slower | Higher | Medium |
| Continuous | Fast | Low | High |
| PPO | Medium | Medium | High |

---

## Running the Configs

```bash
# Baseline
python train_rl.py cfg/train/pnf_plus_rl_v2_discrete.yaml

# No prompts
python train_rl.py cfg/train/pnf_plus_rl_v2_discrete_noprompts.yaml

# Complex
python train_rl.py cfg/train/pnf_plus_rl_v2_discrete_complex.yaml

# Continuous
python train_rl.py cfg/train/pnf_plus_rl_v2_continuous.yaml

# PPO
python train_rl.py cfg/train/pnf_plus_rl_v2_ppo.yaml
```

---

## Summary

| Config | Main Difference | Test Question |
|--------|----------------|---------------|
| **Baseline** | Standard setup | What's the baseline performance? |
| **No Prompts** | No clinical features | Do we need clinical prompts? |
| **Complex** | 512 hidden dim | Does more capacity help? |
| **Continuous** | Soft attention | Is continuous better than discrete? |
| **PPO** | Value function | Is PPO better than GRPO? |

All configs use the same:
- Reward function (`classification_only`)
- Training schedule (except epochs)
- Data augmentation
- Evaluation metrics

The differences are in:
- Attention mode (discrete vs continuous)
- RL algorithm (GRPO vs PPO)
- Policy complexity (256 vs 512)
- Clinical feature usage (with vs without)
