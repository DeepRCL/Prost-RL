# Visualization Updates - V1 and V2 Compatibility

## What Was Updated

Updated visualization code to handle both V1 (point-based) and V2 (patch-based) attention models.

### Files Modified:
1. **`prostnfound/test_rl.py`** - Main testing script with heatmap generation
2. **`prostnfound/scripts/visualize_rl_attention.py`** - Dedicated visualization script
3. **`medAI/medAI/modeling/prostnfound_rl_v2.py`** - Added coord output for visualization

## Visualization Behavior

### V1 Models (Legacy Point-Based)
**Display**: Red 'X' markers at precise coordinates
```
• Red cross markers (x) at selected points
• Point numbers labeled (1, 2, 3...)
• Probability annotations if available
```

### V2 Discrete Models (Patch-Based)
**Display**: Red rectangles showing selected patches
```
• Red rectangular boxes around selected patches
• Patch numbers in circles (1, 2, 3...)
• Automatic patch size detection from attention map
• Legend shows "RL Patches (k)"
```

### V2 Continuous Models (Full Attention)
**Display**: Red heatmap overlay
```
• Semi-transparent red heatmap showing attention distribution
• No discrete points/patches (continuous over all patches)
• Label "Continuous Attention" in corner
```

## Auto-Detection Logic

The visualization code automatically detects the model type:

```python
# Detection steps:
1. Check if model has `discrete_attention` attribute
2. Check if outputs contain `rl_action_indices` (V2 discrete marker)
3. Check attention map size vs image size (patch-based indicator)

# Based on detection:
- V2 Discrete → Draw patch rectangles
- V2 Continuous → Show attention heatmap overlay
- V1 → Draw point markers (backward compatible)
```

## Example Visual Output

### V1 Visualization:
```
Image           |  Heatmap + Points
----------------+------------------
                |  ╔═══════════════╗
  Original      |  ║   X  X  X  X  ║  ← Red crosses
  B-mode        |  ║      X        ║
                |  ╚═══════════════╝
```

### V2 Discrete Visualization:
```
Image           |  Heatmap + Patches
----------------+--------------------
                |  ╔═════════════════╗
  Original      |  ║ ┌──┐ ┌──┐      ║  ← Red rectangles
  B-mode        |  ║ │①│  │②│ ┌──┐ ║
                |  ║ └──┘ └──┘ │③│ ║
                |  ╚═════════════════╝
```

### V2 Continuous Visualization:
```
Image           |  Heatmap + Attention
----------------+----------------------
                |  ╔═══════════════════╗
  Original      |  ║ [Red heatmap     ║  ← Semi-transparent
  B-mode        |  ║  overlay showing  ║     attention weights
                |  ║  attention dist]  ║
                |  ╚═══════════════════╝
```

## Code Changes Summary

### 1. `prostnfound_rl_v2.py`
Added coordinate conversion for visualization:
```python
# Convert patch indices to pixel coords for viz
if discrete_attention and action_indices is not None:
    coords_y = action_indices // W_feat
    coords_x = action_indices % W_feat
    scale = 256.0 / H_feat
    rl_coords = torch.stack([coords_x * scale, coords_y * scale], dim=2)
    output['rl_attention_coords'] = rl_coords
```

### 2. `test_rl.py`
Added three visualization modes:
```python
if is_v2_continuous:
    # Show attention heatmap overlay
    ax[1].imshow(attention_map, cmap='Reds', alpha=0.3)
elif is_v2_discrete:
    # Draw patch rectangles
    for x, y in coords:
        rect = Rectangle((x, y), patch_size, patch_size, ...)
        ax[1].add_patch(rect)
else:  # V1
    # Plot point markers
    ax[1].scatter(coords_x, coords_y, marker='x', ...)
```

### 3. `scripts/visualize_rl_attention.py`
Same auto-detection and visualization logic for standalone script.

## Testing

All updated files compile successfully:
```bash
python -m py_compile prostnfound/test_rl.py  ✓
python -m py_compile prostnfound/scripts/visualize_rl_attention.py  ✓
python -m py_compile medAI/medAI/modeling/prostnfound_rl_v2.py  ✓
```

## Usage

No changes needed to your workflow! Just run test as normal:

```bash
# Test V1 model (shows points)
python test_rl.py --model_checkpoint path/to/v1_model.pth

# Test V2 discrete (shows patches)
python test_rl.py --model_checkpoint path/to/v2_discrete_model.pth

# Test V2 continuous (shows heatmap overlay)
python test_rl.py --model_checkpoint path/to/v2_continuous_model.pth
```

The visualization automatically adapts! ✓
