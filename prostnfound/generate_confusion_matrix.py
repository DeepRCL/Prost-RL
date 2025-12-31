#!/usr/bin/env python3
"""
Enhanced Confusion Matrix and Threshold Analysis Script

This script automatically:
1. Evaluates multiple thresholds
2. Generates confusion matrices for each threshold
3. Generates ROC curves
4. Finds and reports the best operating points:
   - Best sensitivity
   - Best specificity  
   - Best balanced (by Youden's J statistic)
   - Best F1 score

Usage:
    python generate_confusion_matrix.py --output_dir outputs/YOUR_MODEL_OUTPUT
"""

import argparse
import os
import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns


def compute_metrics_at_threshold(y_true, y_score, threshold):
    """Compute all metrics at a specific threshold"""
    y_pred = (y_score > threshold).astype(int)
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    # Calculate metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    # Youden's J statistic (for balanced threshold)
    youden_j = sensitivity + specificity - 1
    
    return {
        'threshold': threshold,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1': f1,
        'accuracy': accuracy,
        'youden_j': youden_j,
        'confusion_matrix': cm,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
    }


def find_best_thresholds(y_true, y_score, thresholds=None):
    """Find optimal thresholds for different criteria"""
    
    if thresholds is None:
        # Use thresholds from ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
    
    results = []
    for threshold in thresholds:
        metrics = compute_metrics_at_threshold(y_true, y_score, threshold)
        results.append(metrics)
    
    df_results = pd.DataFrame(results)
    
    # Find best operating points
    best_sensitivity_idx = df_results['sensitivity'].idxmax()
    best_specificity_idx = df_results['specificity'].idxmax()
    best_youden_idx = df_results['youden_j'].idxmax()
    best_f1_idx = df_results['f1'].idxmax()
    
    best_points = {
        'sensitivity': df_results.loc[best_sensitivity_idx],
        'specificity': df_results.loc[best_specificity_idx],
        'balanced': df_results.loc[best_youden_idx],
        'f1': df_results.loc[best_f1_idx]
    }
    
    return df_results, best_points


def plot_confusion_matrix(cm, metrics, title, output_path):
    """Plot and save confusion matrix with metrics"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['Benign', 'Cancer']
    )
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    
    # Add metrics to title
    title += f"\nThreshold: {metrics['threshold']:.3f} | "
    title += f"Sens: {metrics['sensitivity']:.3f} | Spec: {metrics['specificity']:.3f} | "
    title += f"F1: {metrics['f1']:.3f}"
    
    ax.set_title(title, fontsize=12, pad=20)
    
    # Add percentages
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            total = cm[i, :].sum()
            if total > 0:
                percentage = cm[i, j] / total * 100
                ax.text(j, i + 0.25, f'({percentage:.1f}%)', 
                       ha='center', va='center', fontsize=10, color='gray')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_roc_curve_with_points(y_true, y_score, best_points, title, output_path):
    """Plot ROC curve with best operating points marked"""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(fpr, tpr, linewidth=2.5, label=f'ROC curve (AUC = {auc:.3f})', color='blue')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random classifier')
    
    # Mark best operating points
    colors = {'sensitivity': 'red', 'specificity': 'green', 'balanced': 'purple', 'f1': 'orange'}
    markers = {'sensitivity': 'o', 'specificity': 's', 'balanced': '^', 'f1': 'd'}
    
    for name, point in best_points.items():
        # Calculate FPR and TPR for this point
        point_fpr = 1 - point['specificity']
        point_tpr = point['sensitivity']
        
        ax.scatter(point_fpr, point_tpr, 
                  color=colors[name], marker=markers[name], s=200, 
                  label=f"Best {name} (t={point['threshold']:.3f})",
                  edgecolors='black', linewidths=1.5, zorder=10)
    
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    ax.set_title(title, fontsize=14, pad=20)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_threshold_performance(df_results, output_path):
    """Plot how metrics change with threshold"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Sensitivity and Specificity vs Threshold
    ax = axes[0, 0]
    ax.plot(df_results['threshold'], df_results['sensitivity'], 
            label='Sensitivity', linewidth=2, color='blue')
    ax.plot(df_results['threshold'], df_results['specificity'], 
            label='Specificity', linewidth=2, color='red')
    ax.set_xlabel('Threshold', fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Sensitivity & Specificity vs Threshold', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Plot 2: F1 Score vs Threshold
    ax = axes[0, 1]
    ax.plot(df_results['threshold'], df_results['f1'], 
            linewidth=2, color='green')
    ax.set_xlabel('Threshold', fontsize=11)
    ax.set_ylabel('F1 Score', fontsize=11)
    ax.set_title('F1 Score vs Threshold', fontsize=12)
    ax.grid(alpha=0.3)
    
    # Plot 3: Youden's J vs Threshold
    ax = axes[1, 0]
    ax.plot(df_results['threshold'], df_results['youden_j'], 
            linewidth=2, color='purple')
    ax.set_xlabel('Threshold', fontsize=11)
    ax.set_ylabel("Youden's J", fontsize=11)
    ax.set_title("Youden's J (Balanced) vs Threshold", fontsize=12)
    ax.grid(alpha=0.3)
    
    # Plot 4: Accuracy vs Threshold
    ax = axes[1, 1]
    ax.plot(df_results['threshold'], df_results['accuracy'], 
            linewidth=2, color='orange')
    ax.set_xlabel('Threshold', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title('Accuracy vs Threshold', fontsize=12)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def analyze_prediction_column(df, pred_col, label_col, cm_dir, pred_name):
    """Analyze a single prediction column"""
    
    print("\n" + "="*80)
    print(f"ANALYZING: {pred_name}")
    print("="*80)
    
    y_true = df[label_col].values
    y_score = df[pred_col].values
    
    # Filter out NaN values
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_score))
    if not valid_mask.all():
        n_removed = (~valid_mask).sum()
        print(f"\n⚠️  Filtered out {n_removed} samples with NaN values")
        y_true = y_true[valid_mask]
        y_score = y_score[valid_mask]
    
    if len(y_true) == 0:
        print(f"\n✗ No valid samples to analyze after filtering NaN values")
        return
    
    # Compute AUC
    try:
        auc = roc_auc_score(y_true, y_score)
        print(f"\n✓ AUC-ROC: {auc:.4f}")
    except Exception as e:
        print(f"\n✗ Could not compute AUC: {e}")
        return
    
    # Find best thresholds
    print("\nFinding optimal operating points across all possible thresholds...")
    df_results, best_points = find_best_thresholds(y_true, y_score)
    
    # Save threshold analysis to CSV
    csv_path = os.path.join(cm_dir, f'threshold_analysis_{pred_name}.csv')
    df_results.to_csv(csv_path, index=False)
    print(f"\n✓ Saved threshold analysis to: {csv_path}")
    
    # Report best operating points
    print("\n" + "="*80)
    print("BEST OPERATING POINTS:")
    print("="*80)
    
    for criterion, point in best_points.items():
        print(f"\n🎯 Best {criterion.upper()}:")
        print(f"   Threshold:    {point['threshold']:.4f}")
        print(f"   Sensitivity:  {point['sensitivity']:.3f}")
        print(f"   Specificity:  {point['specificity']:.3f}")
        print(f"   Precision:    {point['precision']:.3f}")
        print(f"   F1 Score:     {point['f1']:.3f}")
        print(f"   Accuracy:     {point['accuracy']:.3f}")
        if criterion == 'balanced':
            print(f"   Youden's J:   {point['youden_j']:.3f}")
        
        # Generate confusion matrix for this operating point
        cm_path = os.path.join(cm_dir, f'cm_{pred_name}_best_{criterion}.png')
        
        # Convert point to dict if it's a Series to access confusion_matrix properly
        point_dict = point.to_dict() if hasattr(point, 'to_dict') else point
        
        # Create metrics dict for plotting
        metrics_dict = {
            'threshold': float(point['threshold']),
            'sensitivity': float(point['sensitivity']),
            'specificity': float(point['specificity']),
            'f1': float(point['f1'])
        }
        
        # Get confusion matrix - it might be stored differently
        if isinstance(point_dict.get('confusion_matrix'), np.ndarray):
            cm = point_dict['confusion_matrix']
        else:
            # Recalculate it
            cm = compute_metrics_at_threshold(y_true, y_score, float(point['threshold']))['confusion_matrix']
        
        plot_confusion_matrix(
            cm, 
            metrics_dict,
            f"{pred_name} - Best {criterion.capitalize()}",
            cm_path
        )
        print(f"   Saved CM:     {os.path.basename(cm_path)}")
    
    # Plot ROC curve with best points marked
    print("\n📊 Generating visualizations...")
    roc_path = os.path.join(cm_dir, f'roc_curve_{pred_name}.png')
    plot_roc_curve_with_points(
        y_true, y_score, best_points,
        f"ROC Curve: {pred_name}",
        roc_path
    )
    print(f"✓ ROC curve: {os.path.basename(roc_path)}")
    
    # Plot threshold performance
    threshold_perf_path = os.path.join(cm_dir, f'threshold_performance_{pred_name}.png')
    plot_threshold_performance(df_results, threshold_perf_path)
    print(f"✓ Threshold performance: {os.path.basename(threshold_perf_path)}")


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced confusion matrix analysis with automatic threshold optimization"
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        required=True,
        help='Path to test output directory containing metrics_by_core.csv'
    )
    parser.add_argument(
        '--show_high_preds',
        action='store_true',
        help='Display all high prediction cases'
    )
    
    args = parser.parse_args()
    
    # Load metrics
    metrics_path = os.path.join(args.output_dir, 'metrics_by_core.csv')
    if not os.path.exists(metrics_path):
        print(f"❌ Error: {metrics_path} not found!")
        print("Please run test_rl.py first to generate the metrics file.")
        return
    
    df = pd.read_csv(metrics_path)
    print(f"\n✓ Loaded {len(df)} test samples from {metrics_path}")
    
    # Get label distribution
    if 'label' in df.columns:
        n_positive = df['label'].sum()
        n_negative = len(df) - n_positive
        print(f"  - Positive (Cancer): {n_positive} ({n_positive/len(df)*100:.1f}%)")
        print(f"  - Negative (Benign): {n_negative} ({n_negative/len(df)*100:.1f}%)")
    
    # Create output directory
    cm_dir = os.path.join(args.output_dir, 'confusion_matrices')
    os.makedirs(cm_dir, exist_ok=True)
    print(f"\n✓ Output directory: {cm_dir}")
    
    # Find prediction columns to analyze
    analysis_configs = []
    
    if 'cancer_prediction_max' in df.columns and 'label' in df.columns:
        analysis_configs.append({
            'pred_col': 'cancer_prediction_max',
            'label_col': 'label',
            'name': 'heatmap_max'
        })
    
    if 'average_needle_heatmap_value' in df.columns and 'label' in df.columns:
        analysis_configs.append({
            'pred_col': 'average_needle_heatmap_value',
            'label_col': 'label',
            'name': 'heatmap_roi'
        })
    
    # Look for classification score columns
    cls_cols = [c for c in df.columns if 'classification' in c.lower() and 'score' in c.lower()]
    for cls_col in cls_cols:
        label_col = 'label_cspca' if 'cspca' in cls_col.lower() and 'label_cspca' in df.columns else 'label'
        analysis_configs.append({
            'pred_col': cls_col,
            'label_col': label_col,
            'name': cls_col
        })
    
    if not analysis_configs:
        print("\n❌ No prediction columns found to analyze!")
        print(f"Available columns: {', '.join(df.columns)}")
        return
    
    # Run analysis for each prediction column
    print(f"\n{'='*80}")
    print(f"Found {len(analysis_configs)} prediction type(s) to analyze")
    print(f"{'='*80}")
    
    for config in analysis_configs:
        analyze_prediction_column(
            df, 
            config['pred_col'], 
            config['label_col'],
            cm_dir,
            config['name']
        )
    
    # Analyze high prediction cases if available
    high_pred_path = os.path.join(args.output_dir, 'high_prediction_cases.csv')
    if os.path.exists(high_pred_path):
        print("\n" + "="*80)
        print("HIGH PREDICTION CASES SUMMARY (CLS or ROI > 40%)")
        print("="*80)
        
        df_high = pd.read_csv(high_pred_path)
        print(f"\n✓ Found {len(df_high)} high prediction cases")
        
        # Summary by ground truth
        cancer_cases = df_high[df_high['gt_label'] == 'Cancer']
        benign_cases = df_high[df_high['gt_label'] == 'Benign']
        
        print(f"\n  True Positives (Cancer correctly flagged): {len(cancer_cases)} ({len(cancer_cases)/len(df_high)*100:.1f}%)")
        print(f"  False Positives (Benign incorrectly flagged): {len(benign_cases)} ({len(benign_cases)/len(df_high)*100:.1f}%)")
        
        if args.show_high_preds and len(df_high) > 0:
            print("\n  Sample cases:")
            for _, case in df_high.head(10).iterrows():
                cls_str = f"{case['cls_score']:.1%}" if isinstance(case['cls_score'], (int, float)) else str(case['cls_score'])
                roi_str = f"{case['roi_avg']:.1%}" if isinstance(case['roi_avg'], (int, float)) else str(case['roi_avg'])
                print(f"    {case['patient_id']}/{case['core_id']}: "
                      f"CLS={cls_str}, ROI={roi_str}, GT={case['gt_label']}")
    
    # Final summary
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nGenerated files in: {cm_dir}/")
    print("\nFile types:")
    print("  - cm_*_best_*.png          : Confusion matrices for best operating points")
    print("  - roc_curve_*.png          : ROC curves with optimal points marked")
    print("  - threshold_performance_*.png : How metrics change with threshold")
    print("  - threshold_analysis_*.csv : Detailed metrics at all thresholds")
    print("\nRecommended next steps:")
    print("  1. Review the ROC curves to understand model discrimination")
    print("  2. Check the 'best balanced' operating point for clinical use")
    print("  3. Examine the threshold performance plots to see trade-offs")
    print("  4. Use the threshold from 'best balanced' in your deployment")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()


