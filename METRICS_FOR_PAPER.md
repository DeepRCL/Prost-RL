# Selected Metrics for Paper Tables

## Quick Reference: Your Requested Metrics

All requested metrics are included:

| Your Request | JSON Key | Location |
|--------------|----------|----------|
| **Core AUROC** | `core_auc` | Classification - Core-Level |
| **Sens @ 60% Spec** | `sens_at_60_spe` | Classification - Core-Level |
| **Sens @ 80% Spec** | `sens_at_80_spe` | Classification - Core-Level |
| **csPCa** | `core_auc_image_level_cspca` | Classification - Image-Level |
| **Heatmap AUROC** | `core_auc_heatmap_cspca` | Heatmap Head |
| **Sens @ 80% Spec (High Inv.)** | `sens_at_80_spe_high_involvement` | Classification - High Involvement |

---

## Classification Head Metrics

These metrics evaluate the **classification head** performance (core-level and image-level predictions).

### Core-Level Classification (All Cores)
1. **Core AUROC**
   - **JSON Key**: `"core_auc"`
   - **Description**: Overall area under ROC curve for core-level cancer classification
   - **Why Important**: Primary metric for classification performance

2. **Sensitivity @ 80% Specificity**
   - **JSON Key**: `"sens_at_80_spe"`
   - **Description**: Sensitivity achieved when specificity is fixed at 80%
   - **Why Important**: Clinical threshold commonly used in prostate cancer detection

3. **Sensitivity @ 60% Specificity**
   - **JSON Key**: `"sens_at_60_spe"`
   - **Description**: Sensitivity achieved when specificity is fixed at 60%
   - **Why Important**: Alternative threshold for comparison

### Core-Level Classification (High Involvement Only)
4. **Core AUROC (High Involvement)**
   - **JSON Key**: `"core_auc_high_involvement"`
   - **Description**: AUROC for cores with high cancer involvement (>50% typically)
   - **Why Important**: Clinically most relevant subset (clinically significant cancer)

5. **Sensitivity @ 80% Specificity (High Involvement)**
   - **JSON Key**: `"sens_at_80_spe_high_involvement"`
   - **Description**: Sensitivity at 80% specificity for high involvement cores
   - **Why Important**: Performance on clinically significant cases

### Image-Level Classification (csPCa)
6. **Image-Level AUROC (csPCa)**
   - **JSON Key**: `"core_auc_image_level_cspca"`
   - **Description**: AUROC for image-level csPCa classification
   - **Why Important**: Evaluates classifier head at image level

7. **Sensitivity @ 80% Specificity (Image-Level csPCa)**
   - **JSON Key**: `"sens_at_80_spe_image_level_cspca"`
   - **Description**: Sensitivity at 80% specificity for image-level csPCa
   - **Why Important**: Image-level performance at clinical threshold

---

## Heatmap Head Metrics

These metrics evaluate the **heatmap/segmentation head** performance (spatial predictions).

### Heatmap-Level (csPCa)
1. **Heatmap AUROC (csPCa)**
   - **JSON Key**: `"core_auc_heatmap_cspca"`
   - **Description**: AUROC computed from heatmap predictions for csPCa
   - **Why Important**: Primary metric for heatmap/segmentation head performance

2. **Sensitivity @ 80% Specificity (Heatmap csPCa)**
   - **JSON Key**: `"sens_at_80_spe_heatmap_cspca"`
   - **Description**: Sensitivity at 80% specificity from heatmap predictions
   - **Why Important**: Clinical threshold performance for spatial predictions

3. **Sensitivity @ 60% Specificity (Heatmap csPCa)**
   - **JSON Key**: `"sens_at_60_spe_heatmap_cspca"`
   - **Description**: Sensitivity at 60% specificity from heatmap predictions
   - **Why Important**: Alternative threshold for heatmap comparison

---

## Recommended Table Structure

### Table 1: Classification Head Performance
| Metric | JSON Key | Value |
|--------|----------|-------|
| **Core AUROC** | `core_auc` | 0.787 |
| **Sens @ 60% Spec** | `sens_at_60_spe` | 0.808 |
| **Sens @ 80% Spec** | `sens_at_80_spe` | 0.637 |
| Core AUROC (High Inv.) | `core_auc_high_involvement` | 0.853 |
| Sens @ 60% Spec (High Inv.) | `sens_at_60_spe_high_involvement` | 0.875 |
| **Sens @ 80% Spec (High Inv.)** | `sens_at_80_spe_high_involvement` | 0.777 |
| **Image-Level AUROC (csPCa)** | `core_auc_image_level_cspca` | 0.780 |
| Sens @ 80% Spec (Image csPCa) | `sens_at_80_spe_image_level_cspca` | 0.559 |

### Table 2: Heatmap Head Performance
| Metric | JSON Key | Value |
|--------|----------|-------|
| **Heatmap AUROC (csPCa)** | `core_auc_heatmap_cspca` | 0.794 |
| **Sens @ 80% Spec (Heatmap csPCa)** | `sens_at_80_spe_heatmap_cspca` | 0.603 |
| Sens @ 60% Spec (Heatmap csPCa) | `sens_at_60_spe_heatmap_cspca` | 0.853 |

---

## Essential Metrics Summary

**Yes, these are the most important metrics for both tasks.** Here's why:

### Classification Head (Table 1) - Essential Set:
1. ✅ **Core AUROC** - Overall discriminative ability
2. ✅ **Sens @ 60% Spec** - Alternative threshold (less conservative)
3. ✅ **Sens @ 80% Spec** - Clinical threshold (standard)
4. ✅ **Core AUROC (High Inv.)** - Performance on clinically significant cases
5. ✅ **Sens @ 80% Spec (High Inv.)** - Clinical threshold for high-risk cases
6. ✅ **Image-Level AUROC (csPCa)** - Image-level classifier performance

### Heatmap Head (Table 2) - Essential Set:
1. ✅ **Heatmap AUROC (csPCa)** - Overall spatial prediction performance
2. ✅ **Sens @ 80% Spec (Heatmap csPCa)** - Clinical threshold for spatial predictions
3. ✅ **Sens @ 60% Spec (Heatmap csPCa)** - Alternative threshold for comparison

### Optional Additions (if space allows):
- **Sens @ 60% Spec (High Inv.)** - Added to Table 1 for completeness
- **Sens @ 40% Spec** - More lenient threshold (if you want to show high sensitivity)
- **F1 Score** - Balanced metric (if you want precision-recall view)

**Recommendation**: The current set (6 classification + 3 heatmap = 9 metrics) is **sufficient and standard** for a paper comparison table. It covers:
- Overall performance (AUROC)
- Clinical thresholds (80% specificity)
- Alternative thresholds (60% specificity)
- Clinically relevant subset (high involvement)
- Both core-level and image-level for classification
- Spatial predictions for heatmap

---

## Minimal Set (If Space Limited)

### Classification Head (5 metrics)
1. `core_auc` - Core AUROC
2. `core_auc_high_involvement` - Core AUROC (High Inv.)
3. `sens_at_80_spe` - Sens @ 80% Spec
4. `sens_at_80_spe_high_involvement` - Sens @ 80% Spec (High Inv.)
5. `core_auc_image_level_cspca` - Image-Level AUROC (csPCa)

### Heatmap Head (2 metrics)
1. `core_auc_heatmap_cspca` - Heatmap AUROC (csPCa)
2. `sens_at_80_spe_heatmap_cspca` - Sens @ 80% Spec (Heatmap csPCa)

---

## Notes

- **High Involvement**: Cores with significant cancer involvement (typically >50%), clinically most relevant
- **csPCa**: Clinically significant prostate cancer
- **@ 80% Specificity**: Fixed specificity threshold commonly used in clinical practice
- **Core-level**: Prediction aggregated at the core level
- **Image-level**: Prediction at individual image/frame level
- **Heatmap-level**: Spatial predictions from segmentation/heatmap head

These metrics allow you to compare:
- How well the **classification head** performs (core and image level)
- How well the **heatmap head** performs (spatial predictions)
- Performance on **high involvement cases** (most clinically relevant)
- Performance at **clinical thresholds** (80% specificity)
