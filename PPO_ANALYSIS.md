# Deep Dive: PPO Training in ProstNFound-RL

## Overview

Your RL training uses **GRPO (Group Relative Policy Optimization)** with an optional **PPO-style value function**. The config file `pnf_plus_rl_kfold_v2.yaml` has `use_value_function: true`, which means you're using **PPO mode** (not pure GRPO).

---

## Step-by-Step Training Flow

### 1. **Initialization** (`train_rl.py` lines 153-180)

```python
# Check if using PPO mode (with value function)
use_value_function = cfg.model_kw.get('use_value_function', False)  # True in your config

# Create GRPO (or PPO if value function is enabled)
grpo = GRPO(
    clip_eps=0.1,                    # PPO clipping epsilon
    entropy_coef=0.005,              # Exploration bonus
    kl_coef=0.01,                    # KL penalty (prevents policy from changing too much)
    use_value_function=True,         # ENABLED = PPO mode
    value_coef=0.5,                  # Weight for value loss
    num_samples_per_image=4,         # Multiple rollouts per image
)
```

**Key Point**: When `use_value_function=True`, GRPO switches to PPO mode, using a value function baseline instead of group normalization.

---

### 2. **Model Architecture** (`rl_attention_policy.py` lines 110-118)

The value function is a **separate head** in your policy network:

```python
# Value function head for PPO (optional)
if use_value_function:
    self.value_head = nn.Sequential(
        nn.AdaptiveAvgPool2d((4, 4)),      # Global pooling
        nn.Flatten(),
        nn.Linear(hidden_dim * 16, hidden_dim),  # 512*16 = 8192 -> 512
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, 1),          # Output: scalar value estimate
    )
```

**What it does**: Takes the same image features as the policy, pools them globally, and outputs a **single scalar value** estimating the expected reward for the current state.

**Forward pass** (line 159-160):
```python
if self.use_value_function:
    value = self.value_head(features).squeeze(-1)  # Shape: (B,)
```

---

### 3. **Rollout Collection** (`train_rl.py` lines 449-464)

**Batched rollout** (optimization: all samples in one forward pass):

```python
with torch.no_grad():
    # Replicate batch: (B,) -> (B * num_samples,)
    batched_data = replicate_batch_for_sampling(data, num_samples_per_image, device)
    
    # Single batched forward pass
    batched_outputs = model(batched_data, deterministic=False)
    
    # Extract RL info
    old_log_probs = batched_outputs.get('rl_log_probs').detach()  # (B*4, k)
    old_values = batched_outputs.get('rl_value').detach()         # (B*4,) - VALUE ESTIMATES
    rollout_action_indices = batched_outputs.get('rl_action_indices')  # Store actions
    
    # Compute rewards for all samples
    all_rewards = reward_computer(batched_outputs, batched_data)  # (B*4,)
```

**Key outputs**:
- `old_log_probs`: Log probabilities of sampled actions (from old policy)
- `old_values`: **Value estimates** from the value head (baseline for advantages)
- `all_rewards`: Actual rewards computed from model predictions

---

### 4. **Advantage Computation** (`grpo.py` lines 119-129)

**PPO Mode** (when `use_value_function=True`):

```python
if self.use_value_function and values is not None:
    # Single-step advantage: A = R - V
    advantages = rewards - values.detach()  # R - V_old
    returns = rewards.clone()  # Target for value function (R)
    
    # Optional normalization
    if self.normalize_advantages:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-4)
    
    return advantages, returns
```

**Formula**: `Advantage = Reward - Value_estimate`

**Why detach values?**: The value baseline should not receive gradients during advantage computation (it's just a baseline). The value function is trained separately via value loss.

**Returns**: The actual rewards are used as targets for training the value function (supervised learning).

---

### 5. **Policy Loss** (`grpo.py` lines 166-244)

**PPO clipped surrogate objective**:

```python
# Compute ratio: π_new / π_old
log_ratio = log_probs_sum - old_log_probs_sum  # Sum across k attention points
log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)  # Numerical stability
ratio = torch.exp(log_ratio)  # (B,)

# Clipped surrogate objective
advantages = advantages.detach()  # Don't backprop through advantages
surr1 = ratio * advantages
surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages

policy_loss = -torch.min(surr1, surr2).mean()  # Maximize (minimize negative)
```

**What this does**:
- `surr1`: Standard policy gradient (ratio * advantage)
- `surr2`: Clipped version (prevents large policy updates)
- Takes the minimum to ensure conservative updates

**KL Penalty** (line 207-210):
```python
# KL penalty to prevent policy from changing too much
delta = old_log_probs - log_probs
per_point_kl = torch.exp(delta) - delta - 1
kl = per_point_kl.sum(dim=1).mean()
```

**Total policy loss**:
```python
total_loss = policy_loss + kl_coef * kl - entropy_coef * entropy
```

---

### 6. **Value Loss** (`grpo.py` lines 279-283)

**Value function training** (supervised learning):

```python
if self.use_value_function and values is not None and returns is not None:
    value_loss = F.mse_loss(values, returns)  # MSE: (V_new - R)^2
    total_loss = total_loss + self.value_coef * value_loss
```

**What this does**:
- Trains the value head to predict the actual rewards
- `returns = rewards` (single-step, no discounting)
- Value function learns: "Given this image, what reward should I expect?"

**Why it helps**:
- Provides a stable baseline for advantages
- Reduces variance in policy gradient estimates
- Better than pure GRPO for complex reward landscapes

---

### 7. **PPO Update Loop** (`train_rl.py` lines 484-520)

**Multiple update epochs** (like standard PPO):

```python
for rl_epoch in range(num_rl_updates):  # Default: 2 epochs
    with torch.amp.autocast('cuda', enabled=args.use_amp):
        # Forward pass with CURRENT policy (reusing same actions)
        current_outputs = model(
            batched_data,
            deterministic=False,
            rl_action_indices=rollout_action_indices,  # Reuse same actions!
        )
        current_log_probs = current_outputs.get('rl_log_probs')  # New policy
        current_values = current_outputs.get('rl_value')          # New value estimates
        
        # Supervised loss (classification/heatmap)
        supervised_loss = criterion(current_outputs)
        
        # RL loss (PPO with value function)
        rl_loss, rl_info = grpo.compute_loss(
            current_log_probs,      # New policy
            old_log_probs,          # Old policy (from rollout)
            all_rewards.detach(),   # Rewards (detached)
            values=current_values,   # New value estimates
        )
        
        # Combined loss
        total_loss = supervised_loss + rl_weight * rl_loss
```

**Key points**:
1. **Same actions**: `rl_action_indices` ensures we evaluate the same actions under the new policy
2. **Multiple epochs**: Policy is updated 2 times on the same batch (PPO standard)
3. **Value updates**: Value function is updated alongside policy

---

## Complete Loss Breakdown

### Total Loss = Supervised Loss + RL Loss

**RL Loss** (when using PPO):
```
RL Loss = Policy Loss + Value Loss + KL Penalty - Entropy Bonus

Where:
- Policy Loss = -min(ratio * A, clip(ratio) * A)
- Value Loss = MSE(V_new, R) * value_coef
- KL Penalty = KL(π_old || π_new) * kl_coef
- Entropy Bonus = H(π) * entropy_coef (currently 0 in your code)
```

**Advantage**:
```
A = R - V_old  (PPO mode)
```

---

## Reward Computation (`rl_loss.py`)

Your config uses `rl_reward_mode: classification_only`, so rewards come from:

```python
def compute_classification_reward(outputs, data):
    cls_logits = outputs['image_level_classification_outputs'][0]  # (B, num_classes)
    probs = F.softmax(cls_logits, dim=1)
    
    # Reward = confidence on correct class
    prob_correct = probs[i, label_i]
    reward = 2.0 * prob_correct - 1.0  # Scale to [-1, 1]
    
    # csPCa bonus
    if grade_group > 2:
        reward *= cspca_bonus  # Default: 2.0
```

**Reward range**: Approximately [-1, 1] (or [-2, 2] for csPCa cases with bonus).

---

## Key Differences: PPO vs Pure GRPO

| Aspect | Pure GRPO | PPO (Your Config) |
|--------|-----------|-------------------|
| **Advantage** | `(R - group_mean) / group_std` | `R - V` |
| **Baseline** | Group statistics (within-image) | Value function |
| **Value Loss** | None | MSE(V, R) |
| **Stability** | Good for within-image comparison | Better for complex rewards |
| **Complexity** | Simpler | Requires value head |

---

## How to Improve Your PPO Method

### 1. **Value Function Architecture**
- **Current**: Simple MLP with global pooling
- **Improvement**: Add attention mechanism or use transformer features
- **Location**: `rl_attention_policy.py` lines 112-118

### 2. **Advantage Normalization**
- **Current**: Batch normalization (line 127 in grpo.py)
- **Improvement**: Try within-image normalization even in PPO mode (like GRPO)
- **Code**: Modify `compute_advantages` to normalize per image group

### 3. **Value Function Pretraining**
- **Current**: Value head starts from scratch
- **Improvement**: Pretrain value function on supervised rewards before RL
- **Benefit**: More stable baseline from the start

### 4. **GAE (Generalized Advantage Estimation)**
- **Current**: Single-step advantage (A = R - V)
- **Improvement**: If you have multi-step rollouts, use GAE with λ
- **Note**: Currently single-step, so GAE not needed

### 5. **Value Loss Weight**
- **Current**: `value_coef=0.5`
- **Improvement**: Try adaptive weighting or schedule (start high, decrease)
- **Config**: `rl_value_coef` in your yaml

### 6. **Value Clipping** (like PPO2)
- **Current**: No clipping on value updates
- **Improvement**: Add value clipping: `V_clipped = V_old + clip(V_new - V_old, -clip_eps, clip_eps)`
- **Code**: Modify `grpo.py` line 281

### 7. **Reward Shaping**
- **Current**: Classification-only reward
- **Improvement**: Try `combined_v2` mode (classification + ROI involvement)
- **Config**: `rl_reward_mode: combined_v2`

### 8. **Multiple Value Heads**
- **Current**: Single scalar value
- **Improvement**: Separate value heads for different reward components
- **Benefit**: Better value estimates for multi-objective rewards

---

## Debugging Tips

### Check Value Estimates
```python
# In train_rl.py, add logging:
step_metrics["train_rl/value_mean"] = current_values.mean().item()
step_metrics["train_rl/value_std"] = current_values.std().item()
step_metrics["train_rl/value_target_mean"] = all_rewards.mean().item()
```

### Check Advantage Statistics
```python
# Already logged in grpo.py:
info['advantages_mean']
info['advantages_std']
```

### Monitor Value Loss
- If value loss is very high → value function is struggling
- If value loss is very low → value function might be overfitting

### Check Ratio Clipping
- `ratio_mean` should be close to 1.0 (policy not changing too much)
- `ratio_max` > 1.1 → policy is updating aggressively
- `ratio_min` < 0.9 → policy is updating conservatively

---

## Summary

Your PPO implementation:
1. ✅ Uses value function baseline (PPO mode)
2. ✅ Clips policy updates (conservative)
3. ✅ Trains value function with MSE loss
4. ✅ Multiple update epochs per batch
5. ✅ Batched forward passes (efficient)

**Main improvement opportunities**:
- Better value function architecture
- Within-image advantage normalization (hybrid GRPO+PPO)
- Value clipping for stability
- Combined reward mode for richer signal
