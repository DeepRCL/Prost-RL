# Difference Between Heatmap and Classification Metrics

## Table 1 vs Table 2: Key Differences

### Table 1: Classification Head Metrics
**Source**: Classification head outputs (logits/scores)
**What it measures**: How well the **classifier** predicts cancer

| Metric | What it uses | What it predicts |
|--------|--------------|-----------------|
| Core AUROC | Classification logits | All cancer (any grade) |
| Image-Level AUROC (csPCa) | Classification logits | csPCa (Grade Group > 2) |

**Key Point**: These metrics use the **classification head's output** (a single score/logit per core/image).

---

### Table 2: Heatmap Head Metrics
**Source**: Heatmap/segmentation head outputs (spatial predictions)
**What it measures**: How well the **spatial predictions** predict cancer

| Metric | What it uses | What it predicts |
|--------|--------------|-----------------|
| Heatmap AUROC (csPCa) | Average heatmap value in needle region | csPCa (Grade Group > 2) |

**Key Point**: These metrics use the **heatmap/segmentation head's output** (spatial map, averaged over needle region).

---

## Differences Within Table 2 (Heatmap Metrics)

All three metrics in Table 2 use the **same source** (heatmap predictions) but report different aspects:

### 1. Heatmap AUROC (csPCa) - `core_auc_heatmap_cspca`
- **What**: Area Under ROC Curve
- **Range**: 0.0 to 1.0 (higher is better)
- **Meaning**: Overall discriminative ability across all thresholds
- **Use**: Primary metric for overall heatmap performance
- **Example**: 0.794 = 79.4% ability to distinguish csPCa vs non-csPCa

### 2. Sens @ 80% Spec (Heatmap csPCa) - `sens_at_80_spe_heatmap_cspca`
- **What**: Sensitivity at fixed 80% specificity
- **Range**: 0.0 to 1.0 (higher is better)
- **Meaning**: How many csPCa cases detected when 80% of non-cancer cases are correctly identified
- **Use**: Clinical threshold performance (standard in prostate cancer)
- **Example**: 0.603 = 60.3% of csPCa cases detected at 80% specificity

### 3. Sens @ 60% Spec (Heatmap csPCa) - `sens_at_60_spe_heatmap_cspca`
- **What**: Sensitivity at fixed 60% specificity
- **Range**: 0.0 to 1.0 (higher is better)
- **Meaning**: How many csPCa cases detected when 60% of non-cancer cases are correctly identified
- **Use**: Alternative threshold (less conservative, higher sensitivity)
- **Example**: 0.853 = 85.3% of csPCa cases detected at 60% specificity

---

## Visual Comparison

```
Classification Head (Table 1):
Input Image → [Classifier] → Single Score (0.0-1.0)
                              ↓
                         Core AUROC
                         Image-Level AUROC (csPCa)

Heatmap Head (Table 2):
Input Image → [Segmentation] → Spatial Map (H×W)
                              ↓
                         Average over needle region
                              ↓
                         Heatmap AUROC (csPCa)
                         Sens @ 80% Spec (Heatmap)
                         Sens @ 60% Spec (Heatmap)
```

---

## Why Both Tables Matter

### Classification Head (Table 1):
- ✅ **Answer**: "Does this core have cancer?"
- ✅ **Output**: Single probability score
- ✅ **Use**: Core-level decision making
- ✅ **Fast**: One score per core

### Heatmap Head (Table 2):
- ✅ **Answer**: "Where is cancer in this image?"
- ✅ **Output**: Spatial probability map
- ✅ **Use**: Localization, spatial understanding
- ✅ **Informative**: Shows where cancer might be

---

## Key Insight

**Same task (csPCa detection), different approaches:**

| Aspect | Classification Head | Heatmap Head |
|--------|-------------------|--------------|
| **Output type** | Single score | Spatial map |
| **Information** | "How likely is cancer?" | "Where is cancer?" |
| **Aggregation** | Direct logit | Average over region |
| **Use case** | Core-level decision | Spatial localization |
| **Metrics** | Table 1 | Table 2 |

**Both are important** because they evaluate different aspects:
- **Table 1**: How good is the classifier at detecting cancer?
- **Table 2**: How good is the heatmap at localizing cancer?

---

## Example from Your Results

From your metrics.json:
- **Classification AUROC (csPCa)**: 0.780 (Image-Level)
- **Heatmap AUROC (csPCa)**: 0.794

**Interpretation**: 
- The heatmap head (0.794) performs slightly better than the classification head (0.780) for csPCa detection
- This suggests the spatial information helps distinguish clinically significant cancer
- Both are good (>0.75), but heatmap provides additional spatial localization

---

## Summary

**Table 2 metrics are all from the same source (heatmap)** but report:
1. **AUROC**: Overall performance (all thresholds)
2. **Sens @ 80% Spec**: Clinical threshold (standard)
3. **Sens @ 60% Spec**: Alternative threshold (higher sensitivity)

**Table 1 vs Table 2**: Different heads (classifier vs heatmap) solving the same task (csPCa detection) but providing different information (score vs spatial map).
