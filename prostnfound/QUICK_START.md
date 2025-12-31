# Quick Start Guide: Improved Testing & Analysis

## tl;dr - What You Need to Do

```bash
# 1. Run test with improved visualization
export DIR_TEST=PNF-RL-V2-continuous-classification_only
export PNF_RL_CHECKPOINT=/home/mahdi.abootorabi/prostnfound/prostnfound/checkpoints_rl_v2/$DIR_TEST/best_rl.pth
python test_rl.py checkpoint=$PNF_RL_CHECKPOINT output_dir=outputs/$DIR_TEST

# 2. Generate comprehensive analysis (NO --threshold parameter needed!)
python generate_confusion_matrix.py --output_dir outputs/$DIR_TEST

# 3. Done! Check outputs/$DIR_TEST/confusion_matrices/ for results
```

---

## What's Changed - The Enhanced Script Now:

### ✨ **Automatic Threshold Analysis**
- **Tries ALL possible thresholds** (from ROC curve)
- **Finds the best operating points automatically**:
  - 🔴 **Best Sensitivity**: Catches the most cancer cases
  - 🟢 **Best Specificity**: Fewest false alarms  
  - 🟣 **Best Balanced** (Youden's J): Optimal trade-off between sensitivity/specificity
  - 🟠 **Best F1**: Best precision-recall balance

### 📊 **Rich Visualizations Generated**

For each prediction type (heatmap_max, heatmap_roi, classification_score), you'll get:

1. **ROC Curve** (`roc_curve_*.png`)
   - Shows performance at ALL thresholds
   - **Best operating points marked with colored dots**
   - Tells you the AUC score

2. **Confusion Matrices** (`cm_*_best_*.png`)
   - **4 confusion matrices** (one for each "best" criterion)
   - Shows TP, TN, FP, FN counts and percentages
   - Includes threshold and metrics in the title

3. **Threshold Performance Plots** (`threshold_performance_*.png`)
   - **4 subplots** showing how metrics change with threshold:
     - Sensitivity & Specificity vs Threshold
     - F1 Score vs Threshold
     - Youden's J vs Threshold
     - Accuracy vs Threshold

4. **CSV Analysis** (`threshold_analysis_*.csv`)
   - **Complete table** with sensitivity, specificity, F1, etc. at every threshold
   - Use this to pick a custom threshold if needed

### 📋 **Detailed Console Output**

The script will print:

```
================================================================================
ANALYZING: heatmap_roi
================================================================================

✓ AUC-ROC: 0.7846

Finding optimal operating points across all possible thresholds...

✓ Saved threshold analysis to: threshold_analysis_heatmap_roi.csv

================================================================================
BEST OPERATING POINTS:
================================================================================

🎯 Best SENSITIVITY:
   Threshold:    0.0123
   Sensitivity:  1.000
   Specificity:  0.120
   Precision:    0.421
   F1 Score:     0.592
   Accuracy:     0.523
   Saved CM:     cm_heatmap_roi_best_sensitivity.png

🎯 Best SPECIFICITY:
   Threshold:    0.9876
   Sensitivity:  0.089
   Specificity:  1.000
   Precision:    1.000
   F1 Score:     0.163
   Accuracy:     0.589
   Saved CM:     cm_heatmap_roi_best_specificity.png

🎯 Best BALANCED:  ⭐ ← THIS IS USUALLY THE BEST FOR CLINICAL USE
   Threshold:    0.3214
   Sensitivity:  0.895
   Specificity:  0.847
   Precision:    0.823
   F1 Score:     0.857
   Accuracy:     0.869
   Youden's J:   0.742
   Saved CM:     cm_heatmap_roi_best_balanced.png

🎯 Best F1:
   Threshold:    0.2987
   Sensitivity:  0.905
   Specificity:  0.812
   Precision:    0.801
   F1 Score:     0.850
   Accuracy:     0.852
   Saved CM:     cm_heatmap_roi_best_f1.png

📊 Generating visualizations...
✓ ROC curve: roc_curve_heatmap_roi.png
✓ Threshold performance: threshold_performance_heatmap_roi.png
```

---

## Understanding the Output

### **Which Threshold Should I Use?**

| Use Case | Recommended | Why |
|----------|-------------|-----|
| **Clinical screening (don't miss cancer)** | **Best Sensitivity** | Catches the most cases, but more false alarms |
| **Clinical diagnosis (avoid false alarms)** | **Best Specificity** | Fewer false positives, but might miss some cases |
| **General clinical use (balanced)** | **Best Balanced** ⭐ | Optimal trade-off using Youden's J statistic |
| **Research comparison** | **Best F1** | Good balance of precision and recall |

**Most commonly used in medicine: Best Balanced (Youden's J)**

### **What is Youden's J?**
- Youden's J = Sensitivity + Specificity - 1
- Ranges from 0 (bad) to 1 (perfect)
- **Maximizes** both sensitivity and specificity simultaneously
- This is the **recommended starting point** for clinical applications

---

## Example Workflow

```bash
# Setup
export DIR_TEST=PNF-RL-V2-continuous-classification_only
export PNF_RL_CHECKPOINT=checkpoints_rl_v2/$DIR_TEST/best_rl.pth

# 1. Run test
python test_rl.py checkpoint=$PNF_RL_CHECKPOINT output_dir=outputs/$DIR_TEST

# Wait for test to complete...

# 2. Analyze results
python generate_confusion_matrix.py \
    --output_dir outputs/$DIR_TEST \
    --show_high_preds  # Optional: show sample high prediction cases

# 3. Review outputs
ls -lh outputs/$DIR_TEST/confusion_matrices/

# You'll see:
# - cm_heatmap_roi_best_balanced.png     ← Best operating point!
# - cm_heatmap_roi_best_sensitivity.png
# - cm_heatmap_roi_best_specificity.png
# - cm_heatmap_roi_best_f1.png
# - roc_curve_heatmap_roi.png             ← ROC curve with all points marked
# - threshold_performance_heatmap_roi.png ← How metrics change
# - threshold_analysis_heatmap_roi.csv    ← Full threshold table

# 4. Use the recommended threshold
# Look for "Best BALANCED" in the console output, e.g., threshold: 0.3214
# Use this threshold in deployment or further analysis
```

---

## FAQ

### Q: Why don't I need to specify --threshold anymore?
**A:** The script now automatically evaluates **ALL thresholds** and finds the best ones for you!

### Q: I only see confusion matrices, where are the ROC curves?
**A:** Make sure you're looking in `outputs/$DIR_TEST/confusion_matrices/`. The ROC curves are named `roc_curve_*.png`. If they're still not there, check the console output for errors.

### Q: How do I know if my model is good?
**A:** Check:
1. **AUC > 0.7**: Good discrimination  
2. **Best Balanced Youden's J > 0.5**: Reasonable clinical performance
3. **Review the ROC curve**: Should curve well above the diagonal

### Q: Can I still use a specific threshold?
**A:** Yes! The full threshold analysis is saved in `threshold_analysis_*.csv`. You can find the row with your desired threshold and see all the metrics.

### Q: Which prediction type should I focus on?
**A:** 
- **heatmap_roi**: Average in needle region (pixel-level)
- **heatmap_max**: Maximum heatmap value (pixel-level)
- **classification scores**: Image-level csPCa prediction (classification head)

Usually, `heatmap_roi` is most relevant for biopsy guidance, and classification scores for diagnosis.

---

## What You Get vs What You Had Before

| Before | After |
|--------|-------|
| One confusion matrix at threshold=0.5 | 4 confusion matrices at optimal thresholds |
| Manually try different thresholds | Automatically finds best thresholds |
| No ROC curve (or basic ROC) | ROC curve with best points marked |
| No threshold analysis | Complete threshold performance plots + CSV |
| Guess which threshold to use | Clear recommendation: "Best Balanced" |

---

## Still Have Questions?

Check these files:
- `IMPROVEMENTS_SUMMARY.md` - Complete overview of all improvements
- `METRICS_EXPLAINED.md` - What ROI, CLS, and other metrics mean  
- `IMPROVED_VISUALIZATION_GUIDE.md` - Technical details on visualizations

Or just run the script and check the pretty plots! 🎨📊
