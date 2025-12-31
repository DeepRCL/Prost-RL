# Improved Test Visualization for ProstNFound-RL

## Summary of Changes

I've enhanced the `test_rl.py` script to address your concerns about understanding the difference between RL attention, decoder output, and classification head predictions.

## Key Improvements

### 1. **3-Panel Visualization for Continuous Attention Models**

For continuous V2 models (`discrete_attention: false`), the visualization now shows **3 panels**:

- **Panel 1 (Left)**: Original B-mode ultrasound image with contours
  - White contour: Prostate mask
  - Yellow contour: Needle mask
  
- **Panel 2 (Middle)**: **RL Attention Map** (What the RL policy focuses on)
  - Shows where the RL agent is "looking" or paying attention
  - Uses a red colormap (Reds) to distinguish from decoder output
  - This is the raw attention heatmap before being used by the decoder
  
- **Panel 3 (Right)**: **Decoder Output Heatmap** (Final cancer prediction)
  - Shows the final cancer probability predictions from the decoder
  - Uses viridis colormap
  - This is conditioned on the RL attention from Panel 2

### 2. **Enhanced Title Information**

The figure title now includes:
- **GT**: Ground truth label (Cancer/Benign) and involvement percentage
- **GG**: Grade Group (if available)
- **Cls**: Classification head prediction (probability from classification head) - **This answers your question about what the classification head predicts**
- **ROI**: Average heatmap value in the needle region (Region of Interest)
- **Core**: Core ID

### 3. **High Prediction Case Tracking**

The script now tracks and logs cases where:
- **CLS > 40%**: Classification head predicts cancer with >40% confidence
- **ROI > 40%**: Heatmap average in needle region is >40%

This helps you verify that the model is actually predicting cancer and not always benign.

At the end of testing, you'll see:
```
=== High Prediction Cases (CLS or ROI > 40%) ===
Found X cases with high predictions:
  PATIENT_ID/CORE_ID: CLS=XX%, ROI=XX%, GT=Cancer/Benign (Inv: XX%)
```

These cases are also saved to `high_prediction_cases.csv` for easy review.

### 4. **Better Visual Separation**

- Different colormaps for different outputs:
  - **Reds** for RL attention map
  - **Viridis** for decoder cancer predictions
- Clearer panel titles
- Better layout with `tight_layout()` and higher DPI (150)

## Understanding the Metrics

### What is ROI?
**ROI (Region of Interest)** shows the average probability value from the **decoder's heatmap** within the **needle mask region**. This represents the model's pixel-level cancer probability in the biopsied area.

### What is CLS?
**CLS (Classification)** shows the probability from the **classification head**. This is an image-level prediction (not pixel-level) that says "does this image contain csPCa (clinically significant prostate cancer)?". This comes from the `image_level_classification_outputs` in your model.

### Confusion Matrix for Classification Head

To get a confusion matrix for the classification head predictions, you can look at these metrics in `metrics.json`:

- `core_auc_image_level_cspca`: AUC for image-level csPCa classification
- `sens_at_XX_spe_image_level_cspca`: Sensitivity at various specificity levels
- `sensitivity_image_level_cspca`, `specificity_image_level_cspca`: Overall sensitivity and specificity

The script also outputs `metrics_by_core.csv` which contains per-core predictions. You can generate a confusion matrix from this file.

### How to Generate Confusion Matrix

Run this after testing:

```python
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Load the per-core metrics
df = pd.read_csv('outputs/YOUR_OUTPUT_DIR/metrics_by_core.csv')

# Get ground truth labels
y_true = df['label'].values  # Or 'label_cspca' for csPCa-specific

# Get classification predictions (you may need to threshold probabilities)
# Option 1: Use heatmap-based predictions
y_pred_heatmap = (df['cancer_prediction_max'] > 0.5).astype(int)

# Option 2: If classification head outputs are in the CSV
# y_pred_cls = (df['classification_score'] > 0.5).astype(int)

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred_heatmap)
print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_true, y_pred_heatmap, target_names=['Benign', 'Cancer']))
```

## Next Steps

1. **Re-run your test** with the updated script on your continuous attention model
2. **Check the new 3-panel visualizations** to see the clear difference between RL attention and decoder output
3. **Review `high_prediction_cases.csv`** to verify cancer predictions
4. **Generate confusion matrix** from `metrics_by_core.csv` to see detailed classification performance

The improved visualization should now clearly show:
- Where the RL agent is paying attention (middle panel in red)
- What the decoder predicts as cancer (right panel in viridis)
- What the classification head predicts (in the title as "Cls")
