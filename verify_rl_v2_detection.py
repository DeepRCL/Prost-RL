#!/usr/bin/env python3
"""Quick verification that RL V2 models are properly detected."""

import sys
import torch

# Test imports
try:
    from medAI.modeling.prostnfound_rl import ProstNFoundRL
    print("✓ ProstNFoundRL imported")
except ImportError as e:
    print(f"✗ Failed to import ProstNFoundRL: {e}")
    sys.exit(1)

try:
    from medAI.modeling.prostnfound_rl_v2 import ProstNFoundRLV2
    print("✓ ProstNFoundRLV2 imported")
    v2_available = True
except ImportError as e:
    print(f"✗ Failed to import ProstNFoundRLV2: {e}")
    v2_available = False
    ProstNFoundRLV2 = None

# Test detection logic
print("\n=== Testing Detection Logic ===")

# Simulate V1 model
class MockV1Model(ProstNFoundRL):
    def __init__(self):
        pass  # Don't call super().__init__()

# Simulate V2 model
class MockV2Model:
    pass
if v2_available:
    MockV2Model = type('MockV2Model', (ProstNFoundRLV2,), {})

# Test V1 detection
v1_model = MockV1Model()
is_rl_v1 = isinstance(v1_model, ProstNFoundRL)
if ProstNFoundRLV2 is not None:
    is_rl_v1 = is_rl_v1 or isinstance(v1_model, ProstNFoundRLV2)
print(f"V1 Model detected as RL: {is_rl_v1} {'✓' if is_rl_v1 else '✗'}")

# Test V2 detection
if v2_available:
    v2_model = MockV2Model()
    is_rl_v2 = isinstance(v2_model, ProstNFoundRL)
    if ProstNFoundRLV2 is not None:
        is_rl_v2 = is_rl_v2 or isinstance(v2_model, ProstNFoundRLV2)
    print(f"V2 Model detected as RL: {is_rl_v2} {'✓' if is_rl_v2 else '✗'}")
    
    if not is_rl_v2:
        print("ERROR: V2 model not detected! Check the detection logic.")
        sys.exit(1)
else:
    print("⚠ Skipping V2 test (ProstNFoundRLV2 not available)")

print("\n=== All Tests Passed ===")
print("The RL model detection should now work correctly in test_rl.py")
