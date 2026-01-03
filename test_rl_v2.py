"""
Quick test to verify RL V2 implementation works.
"""

import torch
import sys
sys.path.insert(0, '/home/mahdi.abootorabi/prostnfound')

def test_policy_discrete():
    print("\n=== Testing Discrete PatchAttentionPolicy ===")
    from medAI.medAI.modeling.rl_attention_policy_v3 import PatchAttentionPolicy
    
    policy = PatchAttentionPolicy(
        feature_dim=256,
        hidden_dim=256,
        num_patches=4,
        discrete_mode=True,
    )
    
    # Test forward
    B, C, H, W = 2, 256, 16, 16
    image_features = torch.randn(B, C, H, W)
    clinical_features = torch.randn(B, 4)
    
    attention_features, log_probs, attention_map, value, action_indices = policy(
        image_features,
        clinical_features=clinical_features,
        deterministic=False,
    )
    
    print(f"✓ Discrete policy forward pass successful")
    print(f"  - attention_features shape: {attention_features.shape}")  # (B, k, C)
    print(f"  - log_probs shape: {log_probs.shape}")  # (B, k)
    print(f"  - attention_map shape: {attention_map.shape}")  # (B, 1, H, W)
    print(f"  - action_indices shape: {action_indices.shape}")  # (B, k)
    

def test_policy_continuous():
    print("\n=== Testing Continuous PatchAttentionPolicy ===")
    from medAI.medAI.modeling.rl_attention_policy_v3 import PatchAttentionPolicy
    
    policy = PatchAttentionPolicy(
        feature_dim=256,
        hidden_dim=256,
        num_patches=4,
        discrete_mode=False,
    )
    
    # Test forward
    B, C, H, W = 2, 256, 16, 16
    image_features = torch.randn(B, C, H, W)
    clinical_features = torch.randn(B, 4)
    
    attention_features, log_probs, attention_map, value, action_indices = policy(
        image_features,
        clinical_features=clinical_features,
        deterministic=False,
    )
    
    print(f"✓ Continuous policy forward pass successful")
    print(f"  - attention_features shape: {attention_features.shape}")  # (B, C, H, W)
    print(f"  - log_probs shape: {log_probs.shape}")  # (B, 1)
    print(f"  - attention_map shape: {attention_map.shape}")  # (B, 1, H, W)
    

def test_value_network():
    print("\n=== Testing ValueNetwork ===")
    from medAI.medAI.modeling.rl_attention_policy_v3 import ValueNetwork
    
    # Test discrete
    value_net = ValueNetwork(
        feature_dim=256,
        hidden_dim=256,
        action_type='discrete',
        num_patches=4,
    )
    
    B, C, H, W = 2, 256, 16, 16
    image_features = torch.randn(B, C, H, W)
    action_features = torch.randn(B, 4, C)  # Discrete: (B, k, C)
    clinical_features = torch.randn(B, 4)
    
    value = value_net(image_features, action_features, clinical_features)
    print(f"✓ Discrete value network forward pass successful")
    print(f"  - value shape: {value.shape}")  # (B,)
    
    # Test continuous
    value_net_cont = ValueNetwork(
        feature_dim=256,
        hidden_dim=256,
        action_type='continuous',
        num_patches=4,
    )
    
    action_features_cont = torch.randn(B, C, H, W)  # Continuous: (B, C, H, W)
    value_cont = value_net_cont(image_features, action_features_cont, clinical_features)
    print(f"✓ Continuous value network forward pass successful")
    print(f"  - value shape: {value_cont.shape}")  # (B,)
    

def test_grpo_v2():
    print("\n=== Testing GRPO_V2 ===")
    from medAI.medAI.modeling.grpo_v2 import GRPO_V2
    
    # Test GRPO mode (no value function)
    grpo = GRPO_V2(
        num_samples_per_image=4,
        use_value_function=False,
    )
    
    B_total = 8  # 2 images * 4 samples
    k = 3
    log_probs = torch.randn(B_total, k)
    old_log_probs = torch.randn(B_total, k)
    rewards = torch.randn(B_total)
    
    loss, info = grpo.compute_loss(
        log_probs,
        old_log_probs,
        rewards,
        num_samples_per_image=4,
        values=None,
    )
    
    print(f"✓ GRPO_V2 (pure GRPO) loss computation successful")
    print(f"  - loss: {loss.item():.4f}")
    print(f"  - within_image_reward_std: {info.get('within_image_reward_std', 'N/A')}")
    
    # Test PPO mode (with value function)
    grpo_ppo = GRPO_V2(
        num_samples_per_image=4,
        use_value_function=True,
    )
    
    values = torch.randn(B_total)
    loss_ppo, info_ppo = grpo_ppo.compute_loss(
        log_probs,
        old_log_probs,
        rewards,
        num_samples_per_image=4,
        values=values,
    )
    
    print(f"✓ GRPO_V2 (PPO mode) loss computation successful")
    print(f"  - loss: {loss_ppo.item():.4f}")
    print(f"  - value_loss: {info_ppo.get('value_loss', 'N/A')}")
    

if __name__ == '__main__':
    print("=" * 60)
    print("Testing RL V2 Implementation")
    print("=" * 60)
    
    try:
        test_policy_discrete()
        test_policy_continuous()
        test_value_network()
        test_grpo_v2()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
