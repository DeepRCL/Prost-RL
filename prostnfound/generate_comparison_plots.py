#!/usr/bin/env python
"""
Generate Model Comparison Plots from Test Results

This script compares multiple models by loading their test results (metrics_by_core.csv)
and generating publication-quality comparison plots.

Usage:
    # Compare multiple models:
    python generate_comparison_plots.py \
        --models "RL-GRPO=outputs/PNF-RL-V2-GRPO" \
                 "RL-PPO=outputs/PNF-RL-V2-PPO" \
                 "Supervised=outputs/Supervised" \
                 "ProstNFound+=outputs/PNF-Plus" \
        --output_dir comparison_plots/
    
    # Or use a config file:
    python generate_comparison_plots.py --config comparison_config.yaml

After running test_rl.py on each model, this script reads the saved CSVs and
generates comparison visualizations.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.model_comparison_plots import ModelComparisonPlotter, compare_models


def load_model_results(
    output_dir: str,
    prediction_column: str = 'average_needle_heatmap_value',
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load model results from test_rl.py output directory.
    
    Args:
        output_dir: Path to model's output directory containing metrics_by_core.csv
        prediction_column: Column name for predictions (default: heatmap average)
        
    Returns:
        predictions: np.ndarray of predictions (NaN-filtered)
        labels: np.ndarray of labels (NaN-filtered)
        results_df: Full results DataFrame
    """
    csv_path = os.path.join(output_dir, 'metrics_by_core.csv')
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Results file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    if prediction_column not in df.columns:
        # Try alternative column names
        alternatives = ['average_needle_heatmap_value', 'topk_score', 'image_level_cancer_logits']
        for alt in alternatives:
            if alt in df.columns:
                prediction_column = alt
                print(f"  Using alternative prediction column: {prediction_column}")
                break
        else:
            raise KeyError(f"Prediction column not found. Available: {df.columns.tolist()}")
    
    predictions = df[prediction_column].values
    labels = df['label'].values
    
    # Filter out NaN values
    valid_mask = ~(np.isnan(predictions) | np.isnan(labels))
    n_nan = (~valid_mask).sum()
    if n_nan > 0:
        print(f"  Filtered out {n_nan} NaN values ({n_nan/len(predictions)*100:.1f}%)")
        predictions = predictions[valid_mask]
        labels = labels[valid_mask]
        df = df[valid_mask].reset_index(drop=True)
    
    return predictions, labels, df



def load_multiple_models(
    model_configs: Dict[str, str],
    prediction_column: str = 'average_needle_heatmap_value',
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Load results from multiple models.
    
    Args:
        model_configs: Dict mapping model name to output directory
        prediction_column: Column name for predictions
        
    Returns:
        Dict mapping model name to (predictions, labels) tuple
    """
    results = {}
    
    for name, output_dir in model_configs.items():
        print(f"Loading model: {name} from {output_dir}")
        try:
            preds, labels, _ = load_model_results(output_dir, prediction_column)
            results[name] = (preds, labels)
            print(f"  Loaded {len(preds)} samples")
        except Exception as e:
            print(f"  Warning: Failed to load {name}: {e}")
    
    return results


def generate_involvement_comparison_plot(
    model_results: Dict[str, Tuple[np.ndarray, np.ndarray, pd.DataFrame]],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Generate a plot comparing model predictions vs involvement across models.
    
    This shows how well each model's predictions correlate with true involvement.
    
    Args:
        model_results: Dict mapping model name to (preds, labels, df) tuple
        save_path: Path to save the figure
        
    Returns:
        Figure object
    """
    n_models = len(model_results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    colors = ['#2E86AB', '#E94F37', '#2E8B57', '#9B59B6', '#F39C12']
    
    for i, (name, (preds, labels, df)) in enumerate(model_results.items()):
        ax = axes[i]
        
        involvement = df['involvement'].values
        
        # Scatter plot with color by label
        scatter = ax.scatter(
            involvement, preds,
            c=labels, cmap='RdYlGn_r',
            alpha=0.6, s=30, edgecolors='white', linewidth=0.5
        )
        
        # Add regression line
        from scipy.stats import pearsonr
        valid_mask = ~(np.isnan(preds) | np.isnan(involvement))
        if valid_mask.sum() > 2:
            z = np.polyfit(involvement[valid_mask], preds[valid_mask], 1)
            p = np.poly1d(z)
            x_line = np.linspace(0, 1, 100)
            ax.plot(x_line, p(x_line), color=colors[i % len(colors)], 
                   linestyle='--', linewidth=2, alpha=0.8)
            
            # Correlation
            corr, _ = pearsonr(involvement[valid_mask], preds[valid_mask])
            ax.annotate(f'r = {corr:.3f}', xy=(0.05, 0.95), xycoords='axes fraction',
                       fontsize=12, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k:', alpha=0.5, linewidth=1)
        
        ax.set_xlabel('True Involvement')
        ax.set_ylabel('Predicted Score')
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=axes[-1])
    cbar.set_label('Label (0=Benign, 1=Cancer)')
    
    fig.suptitle('Prediction vs True Involvement', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def generate_error_by_involvement_bins_plot(
    model_results: Dict[str, Tuple[np.ndarray, np.ndarray, pd.DataFrame]],
    bins: List[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
    error_type: str = 'mae',  # 'mae' or 'mse'
) -> plt.Figure:
    """
    Generate a grouped bar chart showing average prediction error at different involvement bins.
    
    X-axis: Involvement bins (e.g., 0-20%, 20-40%, etc.) + Overall
    Y-axis: Average error (MAE or MSE)
    Each bar group: Different models
    """
    if bins is None:
        bins = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    
    # Add Overall bin (0-100%)
    plotting_bins = bins + [(0.0, 1.0)]
    
    n_models = len(model_results)
    n_bins = len(plotting_bins)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(n_bins)
    width = 0.8 / n_models
    
    colors = ['#2E86AB', '#E94F37', '#2E8B57', '#9B59B6', '#F39C12', '#1ABC9C', '#E74C3C', '#3498DB']
    
    for i, (name, (preds, labels, df)) in enumerate(model_results.items()):
        involvement = df['involvement'].values
        
        bin_errors = []
        bin_counts = []
        
        for j, (low, high) in enumerate(plotting_bins):
            # Normal logic for involvement masks
            if j < len(bins):
                mask = (involvement >= low) & (involvement < high)
                if high == 1.0:  # Include 1.0 in last bin
                    mask = (involvement >= low) & (involvement <= high)
            else:
                # Overall bin (all samples)
                mask = np.ones(len(involvement), dtype=bool)
            
            if mask.sum() > 0:
                bin_preds = preds[mask]
                bin_inv = involvement[mask]
                
                if error_type == 'mae':
                    error = np.abs(bin_preds - bin_inv).mean()
                else:  # mse
                    error = ((bin_preds - bin_inv) ** 2).mean()
                
                bin_errors.append(error)
                bin_counts.append(mask.sum())
            else:
                bin_errors.append(0)
                bin_counts.append(0)
        
        bars = ax.bar(
            x + i * width - width * n_models / 2 + width / 2,
            bin_errors, width,
            label=name,
            color=colors[i % len(colors)],
            edgecolor='white', linewidth=0.5,
        )
        
        # Add value labels on bars
        for bar, val, count in zip(bars, bin_errors, bin_counts):
            if val > 0 and count > 1:  # Only label if enough samples
                ax.annotate(
                    f'{val:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=7, rotation=45
                )
    
    # X-axis labels
    bin_labels = [f'{int(low*100)}-{int(high*100)}%' for low, high in bins] + ['Overall']
    ax.set_xlabel('True Involvement Range', fontsize=12)
    ax.set_ylabel(f'{"Mean Absolute Error (MAE)" if error_type == "mae" else "Mean Squared Error (MSE)"}', fontsize=12)
    ax.set_title('Prediction Error by Involvement Level', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels)
    ax.legend(loc='upper left', ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add vertical line before Overall
    ax.axvline(x=len(bins) - 0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Add sample count annotation
    first_model = list(model_results.values())[0]
    first_df = first_model[2]
    first_inv = first_df['involvement'].values
    
    for j, (low, high) in enumerate(plotting_bins):
        if j < len(bins):
            mask = (first_inv >= low) & (first_inv < high)
            if high == 1.0:
                mask = (first_inv >= low) & (first_inv <= high)
        else:
            mask = np.ones(len(first_inv), dtype=bool)
            
        count = mask.sum()
        ax.annotate(
            f'n={count}',
            xy=(j, 0), xycoords=('data', 'axes fraction'),
            xytext=(0, -25), textcoords='offset points',
            ha='center', va='top', fontsize=8, color='gray',
        )
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def generate_error_by_involvement_bins_plot_thresholded(
    model_results: Dict[str, Tuple[np.ndarray, np.ndarray, pd.DataFrame]],
    bins: List[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
    error_type: str = 'mae',  # 'mae' or 'mse'
) -> plt.Figure:
    """
    Generate a grouped bar chart showing average prediction error at different involvement bins.
    
    This version uses THRESHOLDED involvement calculation for model predictions:
    - Model involvement = mean(sigmoid(logits) > 0.5) instead of mean(sigmoid(logits))
    
    X-axis: Involvement bins (e.g., 0-20%, 20-40%, etc.) + Overall
    Y-axis: Average error (MAE or MSE)
    Each bar group: Different models
    """
    if bins is None:
        bins = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    
    # Add Overall bin (0-100%)
    plotting_bins = bins + [(0.0, 1.0)]
    
    n_models = len(model_results)
    n_bins = len(plotting_bins)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(n_bins)
    width = 0.8 / n_models
    
    colors = ['#2E86AB', '#E94F37', '#2E8B57', '#9B59B6', '#F39C12', '#1ABC9C', '#E74C3C', '#3498DB']
    
    for i, (name, (preds, labels, df)) in enumerate(model_results.items()):
        # Use ground truth involvement for binning
        involvement = df['involvement'].values
        
        bin_errors = []
        bin_counts = []
        
        for j, (low, high) in enumerate(plotting_bins):
            # Normal logic for involvement masks
            if j < len(bins):
                mask = (involvement >= low) & (involvement < high)
                if high == 1.0:  # Include 1.0 in last bin
                    mask = (involvement >= low) & (involvement <= high)
            else:
                # Overall bin (all samples)
                mask = np.ones(len(involvement), dtype=bool)
            
            if mask.sum() > 0:
                # Use thresholded involvement if available, otherwise fall back to regular
                if 'thresholded_needle_involvement' in df.columns:
                    bin_preds = df['thresholded_needle_involvement'].values[mask]
                else:
                    # Fallback: threshold the regular predictions
                    bin_preds = (preds[mask] > 0.5).astype(float)
                
                bin_inv = involvement[mask]
                
                if error_type == 'mae':
                    error = np.abs(bin_preds - bin_inv).mean()
                else:  # mse
                    error = ((bin_preds - bin_inv) ** 2).mean()
                
                bin_errors.append(error)
                bin_counts.append(mask.sum())
            else:
                bin_errors.append(0)
                bin_counts.append(0)
        
        bars = ax.bar(
            x + i * width - width * n_models / 2 + width / 2,
            bin_errors, width,
            label=name,
            color=colors[i % len(colors)],
            edgecolor='white', linewidth=0.5,
        )
        
        # Add value labels on bars
        for bar, val, count in zip(bars, bin_errors, bin_counts):
            if val > 0 and count > 1:  # Only label if enough samples
                ax.annotate(
                    f'{val:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=7, rotation=45
                )
    
    # X-axis labels
    bin_labels = [f'{int(low*100)}-{int(high*100)}%' for low, high in bins] + ['Overall']
    ax.set_xlabel('True Involvement Range', fontsize=12)
    ax.set_ylabel(f'{"Mean Absolute Error (MAE)" if error_type == "mae" else "Mean Squared Error (MSE)"}', fontsize=12)
    ax.set_title('Prediction Error by Involvement Level (Thresholded Involvement)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels)
    ax.legend(loc='upper left', ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add vertical line before Overall
    ax.axvline(x=len(bins) - 0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Add sample count annotation
    first_model = list(model_results.values())[0]
    first_df = first_model[2]
    first_inv = first_df['involvement'].values
    
    for j, (low, high) in enumerate(plotting_bins):
        if j < len(bins):
            mask = (first_inv >= low) & (first_inv < high)
            if high == 1.0:
                mask = (first_inv >= low) & (first_inv <= high)
        else:
            mask = np.ones(len(first_inv), dtype=bool)
            
        count = mask.sum()
        ax.annotate(
            f'n={count}',
            xy=(j, 0), xycoords=('data', 'axes fraction'),
            xytext=(0, -25), textcoords='offset points',
            ha='center', va='top', fontsize=8, color='gray',
        )
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def generate_attention_comparison_plot(
    model_output_dirs: Dict[str, str],
    save_path: Optional[str] = None,
) -> Optional[plt.Figure]:
    """
    Generate attention-specific comparison plots for RL models.
    
    Compares attention-involvement correlation, attention AUROC, etc.
    
    Args:
        model_output_dirs: Dict mapping model name to output directory
        save_path: Path to save the figure
        
    Returns:
        Figure object or None if metrics not available
    """
    rl_metrics = {}
    
    # Load RL-specific metrics from each model
    for name, output_dir in model_output_dirs.items():
        metrics_path = os.path.join(output_dir, 'metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                metrics = json.load(f)
            
            # Extract RL-specific metrics
            rl_specific = {k.replace('rl/', ''): v for k, v in metrics.items() 
                          if k.startswith('rl/')}
            if rl_specific:
                rl_metrics[name] = rl_specific
    
    if not rl_metrics:
        print("No RL-specific metrics found. Skipping attention comparison.")
        return None
    
    # Create comparison bar chart
    metric_names = [
        ('attention_auroc', 'Attention AUROC'),
        ('attention_involvement_correlation_spearman', 'Att-Inv Correlation'),
        ('attention_contrast', 'Attention Contrast'),
        ('benign_attention_sparsity', 'Benign Sparsity'),
        ('cancer_needle_focus_ratio', 'Needle Focus Ratio'),
    ]
    
    # Filter to metrics that exist in at least one model
    available_metrics = []
    for key, display_name in metric_names:
        if any(key in m for m in rl_metrics.values()):
            available_metrics.append((key, display_name))
    
    if not available_metrics:
        print("No common RL metrics found.")
        return None
    
    n_metrics = len(available_metrics)
    n_models = len(rl_metrics)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(n_metrics)
    width = 0.8 / n_models
    
    colors = ['#2E86AB', '#E94F37', '#2E8B57', '#9B59B6', '#F39C12']
    
    for i, (model_name, metrics) in enumerate(rl_metrics.items()):
        values = []
        for key, _ in available_metrics:
            val = metrics.get(key, 0)
            values.append(val if val is not None else 0)
        
        bars = ax.bar(
            x + i * width - width * n_models / 2 + width / 2,
            values, width,
            label=model_name,
            color=colors[i % len(colors)],
            edgecolor='white', linewidth=0.5,
        )
        
        # Add value labels
        for bar, val in zip(bars, values):
            if val != 0:
                ax.annotate(
                    f'{val:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=8,
                )
    
    ax.set_ylabel('Metric Value')
    ax.set_title('RL Attention Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([name for _, name in available_metrics], rotation=15, ha='right')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def generate_classification_metrics_comparison(
    model_results: Dict[str, Tuple[np.ndarray, np.ndarray, pd.DataFrame]],
    save_path: Optional[str] = None,
    threshold: float = 0.5,
) -> plt.Figure:
    """
    Generate a comprehensive comparison of classification metrics across models.
    
    Shows: Accuracy, Precision, Recall, F1, Specificity for each model.
    
    Args:
        model_results: Dict mapping model name to (preds, labels, df) tuple
        save_path: Path to save the figure
        threshold: Classification threshold (default: 0.5)
        
    Returns:
        Figure object
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    metrics_data = {}
    
    for name, (preds, labels, df) in model_results.items():
        # Apply sigmoid if predictions are logits (values outside [0,1])
        if preds.min() < 0 or preds.max() > 1:
            preds_prob = 1 / (1 + np.exp(-preds))
        else:
            preds_prob = preds
        
        # Threshold predictions
        preds_binary = (preds_prob >= threshold).astype(int)
        
        # Calculate metrics
        acc = accuracy_score(labels, preds_binary)
        prec = precision_score(labels, preds_binary, zero_division=0)
        rec = recall_score(labels, preds_binary, zero_division=0)
        f1 = f1_score(labels, preds_binary, zero_division=0)
        
        # Specificity (True Negative Rate)
        tn, fp, fn, tp = confusion_matrix(labels, preds_binary).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        metrics_data[name] = {
            'Accuracy': acc,
            'Precision': prec,
            'Recall (Sens.)': rec,
            'F1 Score': f1,
            'Specificity': spec,
        }
    
    # Create grouped bar chart
    metrics_names = list(next(iter(metrics_data.values())).keys())
    n_metrics = len(metrics_names)
    n_models = len(metrics_data)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(n_metrics)
    width = 0.8 / n_models
    
    colors = ['#2E86AB', '#E94F37', '#2E8B57', '#9B59B6', '#F39C12', '#1ABC9C', '#E74C3C', '#3498DB']
    
    for i, (model_name, metrics) in enumerate(metrics_data.items()):
        values = [metrics[m] for m in metrics_names]
        
        bars = ax.bar(
            x + i * width - width * n_models / 2 + width / 2,
            values, width,
            label=model_name,
            color=colors[i % len(colors)],
            edgecolor='white', linewidth=0.5,
        )
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.annotate(
                f'{val:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3), textcoords='offset points',
                ha='center', va='bottom', fontsize=8, rotation=45
            )
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Classification Metrics Comparison (threshold={threshold})', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def generate_confusion_matrices_comparison(
    model_results: Dict[str, Tuple[np.ndarray, np.ndarray, pd.DataFrame]],
    save_path: Optional[str] = None,
    threshold: float = 0.5,
) -> plt.Figure:
    """
    Generate confusion matrices for all models side-by-side.
    
    Args:
        model_results: Dict mapping model name to (preds, labels, df) tuple
        save_path: Path to save the figure
        threshold: Classification threshold (default: 0.5)
        
    Returns:
        Figure object
    """
    from sklearn.metrics import confusion_matrix
    
    n_models = len(model_results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    
    if n_models == 1:
        axes = [axes]
    
    for ax, (name, (preds, labels, df)) in zip(axes, model_results.items()):
        # Apply sigmoid if predictions are logits
        if preds.min() < 0 or preds.max() > 1:
            preds_prob = 1 / (1 + np.exp(-preds))
        else:
            preds_prob = preds
        
        # Threshold predictions
        preds_binary = (preds_prob >= threshold).astype(int)
        
        # Calculate confusion matrix
        cm = confusion_matrix(labels, preds_binary)
        
        # Plot
        im = ax.imshow(cm, cmap='Blues', aspect='auto')
        ax.set_xlabel('Predicted Label', fontsize=10)
        ax.set_ylabel('True Label', fontsize=10)
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Benign', 'Cancer'])
        ax.set_yticklabels(['Benign', 'Cancer'])
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                text = ax.text(j, i, cm[i, j],
                             ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black",
                             fontsize=16, fontweight='bold')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Add metrics annotation
        tn, fp, fn, tp = cm.ravel()
        acc = (tp + tn) / (tp + tn + fp + fn)
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        metrics_text = f'Acc: {acc:.3f}\nSens: {sens:.3f}\nSpec: {spec:.3f}'
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    fig.suptitle(f'Confusion Matrices (threshold={threshold})', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def generate_confusion_matrices_with_per_model_thresholds(
    model_results: Dict[str, Tuple[np.ndarray, np.ndarray, pd.DataFrame]],
    thresholds_dict: Dict[str, float],
    title_suffix: str = "",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Generate confusion matrices side-by-side, where EACH model uses its OWN threshold.
    
    Args:
        model_results: Dict mapping model name to (preds, labels, df) tuple
        thresholds_dict: Dict mapping model name to the threshold to use
        title_suffix: Suffix for the main figure title
        save_path: Path to save the figure
        
    Returns:
        Figure object
    """
    from sklearn.metrics import confusion_matrix
    
    n_models = len(model_results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5))
    
    if n_models == 1:
        axes = [axes]
    
    for ax, (name, (preds, labels, df)) in zip(axes, model_results.items()):
        # Get threshold for this model
        threshold = thresholds_dict.get(name, 0.5)
        
        # Apply sigmoid if predictions are logits
        if preds.min() < 0 or preds.max() > 1:
            preds_prob = 1 / (1 + np.exp(-preds))
        else:
            preds_prob = preds
        
        # Threshold predictions
        preds_binary = (preds_prob >= threshold).astype(int)
        
        # Calculate confusion matrix
        cm = confusion_matrix(labels, preds_binary)
        
        # Plot
        im = ax.imshow(cm, cmap='Blues', aspect='auto')
        ax.set_xlabel('Predicted Label', fontsize=10)
        ax.set_ylabel('True Label', fontsize=10)
        ax.set_title(f"{name}\n(threshold={threshold:.3f})", fontsize=11, fontweight='bold')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Benign', 'Cancer'])
        ax.set_yticklabels(['Benign', 'Cancer'])
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                text = ax.text(j, i, cm[i, j],
                             ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black",
                             fontsize=16, fontweight='bold')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Add metrics annotation
        tn, fp, fn, tp = cm.ravel()
        acc = (tp + tn) / (tp + tn + fp + fn)
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        metrics_text = f'Acc: {acc:.3f}\nSens: {sens:.3f}\nSpec: {spec:.3f}'
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    fig.suptitle(f'Confusion Matrices {title_suffix}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig

def generate_calibration_curves_comparison(
    model_results: Dict[str, Tuple[np.ndarray, np.ndarray, pd.DataFrame]],
    save_path: Optional[str] = None,
    n_bins: int = 10,
) -> plt.Figure:
    """
    Generate calibration curves comparing predicted probabilities vs actual frequencies.
    
    Args:
        model_results: Dict mapping model name to (preds, labels, df) tuple
        save_path: Path to save the figure
        n_bins: Number of bins for calibration curve
        
    Returns:
        Figure object
    """
    from sklearn.calibration import calibration_curve
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    colors = ['#2E86AB', '#E94F37', '#2E8B57', '#9B59B6', '#F39C12', '#1ABC9C']
    
    for i, (name, (preds, labels, df)) in enumerate(model_results.items()):
        # Apply sigmoid if predictions are logits
        if preds.min() < 0 or preds.max() > 1:
            preds_prob = 1 / (1 + np.exp(-preds))
        else:
            preds_prob = preds
        
        # Calculate calibration curve
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                labels, preds_prob, n_bins=n_bins, strategy='uniform'
            )
            
            # Plot
            ax.plot(mean_predicted_value, fraction_of_positives, 
                   marker='o', linewidth=2, label=name, color=colors[i % len(colors)])
        except Exception as e:
            print(f"Warning: Could not generate calibration curve for {name}: {e}")
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k:', label='Perfect Calibration', linewidth=2)
    
    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives', fontsize=12)
    ax.set_title('Calibration Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def find_optimal_threshold(
    labels: np.ndarray,
    predictions: np.ndarray,
    metric: str = 'balanced_accuracy'
) -> Tuple[float, float]:
    """
    Find optimal classification threshold by maximizing a metric.
    
    Args:
        labels: True binary labels
        predictions: Predicted probabilities (or logits, will be converted)
        metric: Metric to optimize ('balanced_accuracy', 'f1', 'sensitivity', 'specificity')
        
    Returns:
        optimal_threshold: Best threshold value
        max_score: Maximum score achieved for the targeted metric
    """
    from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
    
    # Apply sigmoid if predictions are logits
    if predictions.min() < 0 or predictions.max() > 1:
        preds_prob = 1 / (1 + np.exp(-predictions))
    else:
        preds_prob = predictions
    
    # Try different thresholds
    thresholds = np.linspace(0.01, 0.99, 99)
    results = []
    
    for thresh in thresholds:
        preds_binary = (preds_prob >= thresh).astype(int)
        
        # Calculate base metrics
        tn, fp, fn, tp = confusion_matrix(labels, preds_binary).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        if metric == 'balanced_accuracy':
            score = balanced_accuracy_score(labels, preds_binary)
        elif metric == 'f1':
            score = f1_score(labels, preds_binary, zero_division=0)
        elif metric == 'sensitivity':
            # Target 90% sensitivity if possible, else maximize sensitivity
            # If sensitivity >= 0.9, we return a combined score that favors specificity
            if sens >= 0.9:
                score = 1.0 + spec # Range 1.0 to 2.0
            else:
                score = sens # Range 0.0 to 1.0
        elif metric == 'specificity':
            # Target 90% specificity if possible
            if spec >= 0.9:
                score = 1.0 + sens
            else:
                score = spec
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        results.append((score, sens if metric == 'sensitivity' else (spec if metric == 'specificity' else score)))
    
    scores = np.array([r[0] for r in results])
    best_idx = np.argmax(scores)
    optimal_threshold = thresholds[best_idx]
    
    # Return actual metric value for the second part
    max_score = results[best_idx][1]
    
    return optimal_threshold, max_score


def generate_threshold_analysis_plot(
    model_results: Dict[str, Tuple[np.ndarray, np.ndarray, pd.DataFrame]],
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, Dict[str, Dict[str, float]]]:
    """
    Generate threshold analysis plots showing F1 and Balanced Accuracy vs threshold.
    
    Also finds and returns optimal thresholds for each model.
    
    Args:
        model_results: Dict mapping model name to (preds, labels, df) tuple
        save_path: Path to save the figure
        
    Returns:
        fig: Figure object
        optimal_thresholds: Dict mapping model name to optimal threshold info
    """
    from sklearn.metrics import balanced_accuracy_score, f1_score
    
    n_models = len(model_results)
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    colors = ['#2E86AB', '#E94F37', '#2E8B57', '#9B59B6', '#F39C12', '#1ABC9C', '#E74C3C', '#3498DB']
    thresholds = np.linspace(0.01, 0.99, 99)
    
    optimal_thresholds = {}
    
    for i, (name, (preds, labels, df)) in enumerate(model_results.items()):
        # Apply sigmoid if predictions are logits
        if preds.min() < 0 or preds.max() > 1:
            preds_prob = 1 / (1 + np.exp(-preds))
        else:
            preds_prob = preds
        
        balanced_accs = []
        f1_scores = []
        
        for thresh in thresholds:
            preds_binary = (preds_prob >= thresh).astype(int)
            
            bal_acc = balanced_accuracy_score(labels, preds_binary)
            f1 = f1_score(labels, preds_binary, zero_division=0)
            
            balanced_accs.append(bal_acc)
            f1_scores.append(f1)
        
        balanced_accs = np.array(balanced_accs)
        f1_scores = np.array(f1_scores)
        
        optimal_thresh_bal_acc, max_bal_acc = find_optimal_threshold(labels, preds_prob, 'balanced_accuracy')
        optimal_thresh_f1, max_f1 = find_optimal_threshold(labels, preds_prob, 'f1')
        optimal_thresh_sens, max_sens = find_optimal_threshold(labels, preds_prob, 'sensitivity')
        optimal_thresh_spec, max_spec = find_optimal_threshold(labels, preds_prob, 'specificity')
        
        optimal_thresholds[name] = {
            'threshold_balanced_accuracy': optimal_thresh_bal_acc,
            'max_balanced_accuracy': max_bal_acc,
            'threshold_f1': optimal_thresh_f1,
            'max_f1': max_f1,
            'threshold_sensitivity': optimal_thresh_sens,
            'max_sensitivity': max_sens,
            'threshold_specificity': optimal_thresh_spec,
            'max_specificity': max_spec,
        }
        
        color = colors[i % len(colors)]
        
        # Plot Balanced Accuracy
        axes[0].plot(thresholds, balanced_accs, linewidth=2, label=name, color=color)
        axes[0].scatter([optimal_thresh_bal_acc], [max_bal_acc], s=100, color=color, 
                       marker='*', edgecolor='black', linewidth=1, zorder=10)
        
        # Plot F1 Score
        axes[1].plot(thresholds, f1_scores, linewidth=2, label=name, color=color)
        axes[1].scatter([optimal_thresh_f1], [max_f1], s=100, color=color,
                       marker='*', edgecolor='black', linewidth=1, zorder=10)

    
    # Configure Balanced Accuracy plot
    axes[0].set_xlabel('Classification Threshold', fontsize=12)
    axes[0].set_ylabel('Balanced Accuracy', fontsize=12)
    axes[0].set_title('Balanced Accuracy vs Threshold\n(★ = Optimal Threshold)', 
                     fontsize=13, fontweight='bold')
    axes[0].legend(loc='best', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Random')
    
    # Configure F1 Score plot
    axes[1].set_xlabel('Classification Threshold', fontsize=12)
    axes[1].set_ylabel('F1 Score', fontsize=12)
    axes[1].set_title('F1 Score vs Threshold\n(★ = Optimal Threshold)', 
                     fontsize=13, fontweight='bold')
    axes[1].legend(loc='best', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    # Print optimal thresholds
    print("\n" + "="*60)
    print("OPTIMAL THRESHOLDS FOR EACH MODEL")
    print("="*60)
    for name, thresholds_info in optimal_thresholds.items():
        print(f"\n{name}:")
        print(f"  Balanced Accuracy: threshold={thresholds_info['threshold_balanced_accuracy']:.3f}, "
              f"score={thresholds_info['max_balanced_accuracy']:.3f}")
        print(f"  F1 Score:          threshold={thresholds_info['threshold_f1']:.3f}, "
              f"score={thresholds_info['max_f1']:.3f}")
    print("="*60 + "\n")
    
    return fig, optimal_thresholds


def generate_classification_metrics_with_optimal_thresholds(
    model_results: Dict[str, Tuple[np.ndarray, np.ndarray, pd.DataFrame]],
    optimal_thresholds: Dict[str, Dict[str, float]],
    metric_type: str = 'balanced_accuracy',
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Generate classification metrics comparison where EACH model uses its OWN optimal threshold.
    
    This is more fair than using a single threshold for all models.
    
    Args:
        model_results: Dict mapping model name to (preds, labels, df) tuple
        optimal_thresholds: Dict from generate_threshold_analysis_plot
        metric_type: 'balanced_accuracy' or 'f1' to determine which optimal threshold to use
        save_path: Path to save the figure
        
    Returns:
        Figure object
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, balanced_accuracy_score
    
    metrics_data = {}
    thresholds_used = {}
    
    for name, (preds, labels, df) in model_results.items():
        # Get optimal threshold for this model
        if name in optimal_thresholds:
            threshold = optimal_thresholds[name][f'threshold_{metric_type}']
        else:
            threshold = 0.5  # Fallback
        
        thresholds_used[name] = threshold
        
        # Apply sigmoid if predictions are logits
        if preds.min() < 0 or preds.max() > 1:
            preds_prob = 1 / (1 + np.exp(-preds))
        else:
            preds_prob = preds
        
        # Threshold predictions with model-specific optimal threshold
        preds_binary = (preds_prob >= threshold).astype(int)
        
        # Calculate metrics
        acc = accuracy_score(labels, preds_binary)
        bal_acc = balanced_accuracy_score(labels, preds_binary)
        prec = precision_score(labels, preds_binary, zero_division=0)
        rec = recall_score(labels, preds_binary, zero_division=0)
        f1 = f1_score(labels, preds_binary, zero_division=0)
        
        # Specificity
        tn, fp, fn, tp = confusion_matrix(labels, preds_binary).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        metrics_data[name] = {
            'Accuracy': acc,
            'Balanced Acc': bal_acc,
            'Precision': prec,
            'Recall (Sens.)': rec,
            'F1 Score': f1,
            'Specificity': spec,
        }
    
    # Create grouped bar chart
    metrics_names = list(next(iter(metrics_data.values())).keys())
    n_metrics = len(metrics_names)
    n_models = len(metrics_data)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(n_metrics)
    width = 0.8 / n_models
    
    colors = ['#2E86AB', '#E94F37', '#2E8B57', '#9B59B6', '#F39C12', '#1ABC9C', '#E74C3C', '#3498DB']
    
    for i, (model_name, metrics) in enumerate(metrics_data.items()):
        values = [metrics[m] for m in metrics_names]
        thresh = thresholds_used[model_name]
        
        bars = ax.bar(
            x + i * width - width * n_models / 2 + width / 2,
            values, width,
            label=f'{model_name} (t={thresh:.2f})',
            color=colors[i % len(colors)],
            edgecolor='white', linewidth=0.5,
        )
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.annotate(
                f'{val:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3), textcoords='offset points',
                ha='center', va='bottom', fontsize=7, rotation=45
            )
    
    ax.set_ylabel('Score', fontsize=12)
    title = f'Classification Metrics (Each Model at Optimal {metric_type.replace("_", " ").title()} Threshold)'
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.legend(loc='lower right', fontsize=9, title='Model (threshold)')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)
    
    # Add footnote
    fig.text(0.5, 0.02, '* Each model uses its own optimal threshold maximizing ' + 
             metric_type.replace('_', ' ').title(), 
             ha='center', fontsize=9, style='italic', color='gray')
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def generate_classification_accuracy_by_involvement(
    model_results: Dict[str, Tuple[np.ndarray, np.ndarray, pd.DataFrame]],
    optimal_thresholds: Dict[str, Dict[str, float]],
    metric_type: str = 'balanced_accuracy',
    bins: List[Tuple[float, float]] = None,
    save_path: Optional[str] = None,
    filter_condition: Optional[str] = None,  # 'high_involvement' or 'cspca'
) -> plt.Figure:
    """
    Generate accuracy (not MAE) by involvement bins for classification.
    
    Each model uses its own optimal threshold.
    
    Args:
        model_results: Dict mapping model name to (preds, labels, df) tuple
        optimal_thresholds: Dict from generate_threshold_analysis_plot
        metric_type: 'balanced_accuracy' or 'f1' for threshold selection
        bins: Involvement bins
        save_path: Path to save
        filter_condition: Optional filter ('high_involvement' or 'cspca')
        
    Returns:
        Figure object
    """
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    
    if bins is None:
        bins = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    
    plotting_bins = bins + [(0.0, 1.0)]  # Add Overall
    
    n_models = len(model_results)
    n_bins = len(plotting_bins)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    x = np.arange(n_bins)
    width = 0.8 / n_models
    
    colors = ['#2E86AB', '#E94F37', '#2E8B57', '#9B59B6', '#F39C12', '#1ABC9C', '#E74C3C', '#3498DB']
    
    for i, (name, (preds, labels, df)) in enumerate(model_results.items()):
        # Get optimal threshold
        if name in optimal_thresholds:
            threshold = optimal_thresholds[name][f'threshold_{metric_type}']
        else:
            threshold = 0.5
        
        # Apply sigmoid
        if preds.min() < 0 or preds.max() > 1:
            preds_prob = 1 / (1 + np.exp(-preds))
        else:
            preds_prob = preds
        
        preds_binary = (preds_prob >= threshold).astype(int)
        
        involvement = df['involvement'].values
        
        # Apply filter if specified
        if filter_condition == 'high_involvement':
            filter_mask = involvement >= 0.4  # High involvement threshold
            title_suffix = " (High Involvement ≥40% Only)"
        elif filter_condition == 'cspca':
            if 'grade_group' in df.columns:
                filter_mask = df['grade_group'].values > 2  # csPCa: GG > 2
                title_suffix = " (csPCa: Grade Group >2 Only)"
            else:
                filter_mask = np.ones(len(labels), dtype=bool)
                title_suffix = " (csPCa filter not available)"
        else:
            filter_mask = np.ones(len(labels), dtype=bool)
            title_suffix = ""
        
        # Apply filter
        preds_binary_filtered = preds_binary[filter_mask]
        labels_filtered = labels[filter_mask]
        involvement_filtered = involvement[filter_mask]
        
        if len(labels_filtered) == 0:
            print(f"Warning: No samples for {name} with filter {filter_condition}")
            continue
        
        bin_accs = []
        bin_counts = []
        
        for j, (low, high) in enumerate(plotting_bins):
            if j < len(bins):
                mask = (involvement_filtered >= low) & (involvement_filtered < high)
                if high == 1.0:
                    mask = (involvement_filtered >= low) & (involvement_filtered <= high)
            else:
                # Overall bin
                mask = np.ones(len(involvement_filtered), dtype=bool)
            
            if mask.sum() > 0:
                acc = accuracy_score(labels_filtered[mask], preds_binary_filtered[mask])
                bin_accs.append(acc * 100)  # Convert to percentage
                bin_counts.append(mask.sum())
            else:
                bin_accs.append(0)
                bin_counts.append(0)
        
        bars = ax.bar(
            x + i * width - width * n_models / 2 + width / 2,
            bin_accs, width,
            label=f'{name} (t={threshold:.2f})',
            color=colors[i % len(colors)],
            edgecolor='white', linewidth=0.5,
        )
        
        # Add value labels
        for bar, val, count in zip(bars, bin_accs, bin_counts):
            if val > 0 and count > 1:
                ax.annotate(
                    f'{val:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=7, rotation=45
                )
    
    # X-axis labels
    bin_labels = [f'{int(low*100)}-{int(high*100)}%' for low, high in bins] + ['Overall']
    ax.set_xlabel('True Involvement Range', fontsize=12)
    ax.set_ylabel('Classification Accuracy (%)', fontsize=12)
    ax.set_title(f'Classification Accuracy by Involvement{title_suffix}', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels)
    ax.legend(loc='lower right', fontsize=9, title='Model (threshold)')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 110)
    
    # Add vertical line before Overall
    ax.axvline(x=len(bins) - 0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Add sample counts
    first_model_data = list(model_results.values())[0]
    first_df = first_model_data[2]
    first_inv = first_df['involvement'].values
    
    # Apply same filter
    if filter_condition == 'high_involvement':
        first_filter = first_inv >= 0.4
    elif filter_condition == 'cspca' and 'grade_group' in first_df.columns:
        first_filter = first_df['grade_group'].values > 2
    else:
        first_filter = np.ones(len(first_inv), dtype=bool)
    
    first_inv_filtered = first_inv[first_filter]
    
    for j, (low, high) in enumerate(plotting_bins):
        if j < len(bins):
            mask = (first_inv_filtered >= low) & (first_inv_filtered < high)
            if high == 1.0:
                mask = (first_inv_filtered >= low) & (first_inv_filtered <= high)
        else:
            mask = np.ones(len(first_inv_filtered), dtype=bool)
        
        count = mask.sum()
        ax.annotate(
            f'n={count}',
            xy=(j, 0), xycoords=('data', 'axes fraction'),
            xytext=(0, -30), textcoords='offset points',
            ha='center', va='top', fontsize=8, color='gray',
        )
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def generate_auroc_vs_involvement_threshold_plot(
    model_results: Dict[str, Tuple[np.ndarray, np.ndarray, pd.DataFrame]],
    thresholds: Optional[List[float]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Generate AUROC vs Involvement Threshold plot for csPCa detection performance.
    
    This shows how well models can detect csPCa at different involvement thresholds.
    For each threshold T%, we compute AUROC for detecting cores with involvement >= T%.
    
    Args:
        model_results: Dict mapping model name to (preds, labels, df) tuple
        thresholds: List of involvement thresholds (0-100%). Default: [0, 5, 10, ..., 70]
        save_path: Path to save the figure
        
    Returns:
        Figure object
    """
    from sklearn.metrics import roc_auc_score
    
    if thresholds is None:
        # Start from 5% to ensure we have both positive and negative samples
        # (at 0%, all samples would have involvement >= 0%)
        thresholds = [5, 10, 20, 30, 40, 50, 60, 70]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = ['#2E86AB', '#E94F37', '#2E8B57', '#9B59B6', '#F39C12', '#1ABC9C', '#E74C3C', '#3498DB']
    markers = ['o', '^', 's', 'D', 'v', '<', '>', 'p']
    
    print("\n=== AUROC vs Involvement Threshold ===")
    
    for i, (name, (preds, labels, df)) in enumerate(model_results.items()):
        involvement = df['involvement'].values
        
        # Apply sigmoid if predictions are logits
        if preds.min() < 0 or preds.max() > 1:
            preds_prob = 1 / (1 + np.exp(-preds))
        else:
            preds_prob = preds
        
        aurocs = []
        valid_thresholds = []
        
        for thresh_pct in thresholds:
            thresh = thresh_pct / 100.0
            
            # Create binary labels for this threshold: 1 if involvement >= threshold, 0 otherwise
            binary_labels = (involvement >= thresh).astype(int)
            
            # Need at least 1 positive and 1 negative sample
            if binary_labels.sum() > 0 and binary_labels.sum() < len(binary_labels):
                try:
                    auroc = roc_auc_score(binary_labels, preds_prob)
                    aurocs.append(auroc * 100)  # Convert to percentage
                    valid_thresholds.append(thresh_pct)
                except Exception as e:
                    print(f"  Warning: Could not compute AUROC for {name} at threshold {thresh_pct}%: {e}")
            else:
                print(f"  Skipping {name} at threshold {thresh_pct}% (insufficient samples)")
        
        if len(valid_thresholds) > 0:
            ax.plot(valid_thresholds, aurocs, 
                   marker=markers[i % len(markers)], 
                   linewidth=2.5, 
                   markersize=8,
                   label=name, 
                   color=colors[i % len(colors)])
            
            print(f"{name}:")
            for t, a in zip(valid_thresholds, aurocs):
                print(f"  Threshold {t:3.0f}%: AUROC = {a:5.2f}%")
    
    # Determine y-axis limits based on actual data range
    all_aurocs = []
    for child in ax.get_children():
        if hasattr(child, 'get_ydata'):
            try:
                all_aurocs.extend(child.get_ydata())
            except:
                pass
    
    if all_aurocs:
        min_auroc = min(all_aurocs)
        max_auroc = max(all_aurocs)
        margin = (max_auroc - min_auroc) * 0.1  # 10% margin
        y_min = max(0, min_auroc - margin)
        y_max = min(100, max_auroc + margin)
    else:
        y_min, y_max = 75, 100  # Fallback
    
    ax.set_xlabel('Involvement Threshold (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('AUROC (%)', fontsize=13, fontweight='bold')
    ax.set_title('csPCa Detection Performance', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(-5, max(thresholds) + 5)
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved: {save_path}")
    
    return fig


def generate_activation_vs_involvement_plot(
    model_results: Dict[str, Tuple[np.ndarray, np.ndarray, pd.DataFrame]],
    use_thresholded: bool = False,
    bins: Optional[List[Tuple[float, float]]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Generate Average Activation vs True Percent Cancer in Sample plot.
    
    Shows how model predictions (activation) correlate with true cancer involvement.
    Can use either average activation or thresholded activation (fraction > 0.5).
    
    Args:
        model_results: Dict mapping model name to (preds, labels, df) tuple
        use_thresholded: If True, use thresholded_needle_involvement; else use average
        bins: Involvement bins for grouping. Default: [(0, 0.1), (0.1, 0.2), ..., (0.9, 1.0)]
        save_path: Path to save the figure
        
    Returns:
        Figure object
    """
    if bins is None:
        # Create bins: <10%, 10-20%, 20-30%, ..., 90-100%
        bins = [(0, 0.1)] + [(i/10, (i+1)/10) for i in range(1, 10)]
    
    fig, ax = plt.subplots(figsize=(9, 6))
    
    colors = ['#2E86AB', '#E94F37', '#2E8B57', '#9B59B6', '#F39C12', '#1ABC9C', '#E74C3C', '#3498DB']
    
    print(f"\n=== Activation vs Involvement ({'Thresholded' if use_thresholded else 'Normal'}) ===")
    
    for model_idx, (name, (preds, labels, df)) in enumerate(model_results.items()):
        involvement = df['involvement'].values
        
        # Use thresholded or average activation
        if use_thresholded:
            if 'thresholded_needle_involvement' in df.columns:
                activations = df['thresholded_needle_involvement'].values
                print(f"{name}: Using thresholded_needle_involvement from DataFrame")
            else:
                # Fallback: threshold the predictions
                if preds.min() < 0 or preds.max() > 1:
                    preds_prob = 1 / (1 + np.exp(-preds))
                else:
                    preds_prob = preds
                activations = (preds_prob > 0.5).astype(float)
                print(f"{name}: Thresholding predictions (no thresholded_needle_involvement column)")
        else:
            # Use average predictions
            if preds.min() < 0 or preds.max() > 1:
                activations = 1 / (1 + np.exp(-preds))
            else:
                activations = preds
            print(f"{name}: Using average activation")
        
        # Compute mean and std for each bin
        bin_centers = []
        mean_activations = []
        std_activations = []
        
        for low, high in bins:
            # Find samples in this bin
            if high < 1.0:
                mask = (involvement >= low) & (involvement < high)
            else:
                mask = (involvement >= low) & (involvement <= high)
            
            if mask.sum() > 0:
                bin_center = (low + high) / 2
                mean_act = activations[mask].mean()
                std_act = activations[mask].std() if mask.sum() > 1 else 0
                
                bin_centers.append(bin_center * 100)  # Convert to percentage
                mean_activations.append(mean_act)
                std_activations.append(std_act)
                
                print(f"  {name} - Bin {low*100:.0f}-{high*100:.0f}%: "
                      f"mean={mean_act:.3f}, std={std_act:.3f}, n={mask.sum()}")
        
        # Plot bars with error bars
        bin_centers_arr = np.array(bin_centers)
        mean_activations_arr = np.array(mean_activations)
        std_activations_arr = np.array(std_activations)
        
        # Create bar positions (offset for multiple models)
        n_models = len(model_results)
        bar_width = 8 / n_models  # Adjust based on bin width
        offset = (model_idx - n_models / 2 + 0.5) * bar_width
        
        ax.bar(bin_centers_arr + offset, mean_activations_arr, 
               width=bar_width,
               yerr=std_activations_arr,
               label=name,
               color=colors[model_idx % len(colors)],
               alpha=0.7,
               capsize=3,
               error_kw={'linewidth': 1.5, 'ecolor': 'black'})
    
    # Create custom x-tick labels
    tick_positions = []
    tick_labels = []
    for low, high in bins:
        tick_positions.append((low + high) / 2 * 100)
        if low == 0 and high == 0.1:
            tick_labels.append('<10')
        elif high == 1.0:
            tick_labels.append(f'{int(low*100)}-100')
        else:
            tick_labels.append(f'{int(low*100)}-{int(high*100)}')
    
    ax.set_xlabel('Percent Cancer in Sample (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Avg. activation in needle', fontsize=13, fontweight='bold')
    
    title = 'Activation vs True Involvement'
    if use_thresholded:
        title += ' (Thresholded)'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')
    ax.set_ylim(0, 0.75 if use_thresholded else 1.0)
    ax.legend(loc='upper left', fontsize=10, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved: {save_path}")
    
    return fig


def generate_metrics_summary_table(
    model_output_dirs: Dict[str, str],
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate a summary table of key metrics for all models.
    
    Args:
        model_output_dirs: Dict mapping model name to output directory
        save_path: Path to save CSV
        
    Returns:
        DataFrame with metrics comparison
    """
    summary = []
    
    key_metrics = [
        'core_auc', 'core_auc_high_involvement',
        'core_auc_image_level', 'core_auc_image_level_cspca',
        'rl/attention_auroc', 'rl/attention_involvement_correlation_spearman',
        'rl/attention_contrast', 'rl/benign_mean_attention',
    ]
    
    for name, output_dir in model_output_dirs.items():
        metrics_path = os.path.join(output_dir, 'metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                metrics = json.load(f)
            
            row = {'Model': name}
            for key in key_metrics:
                clean_key = key.replace('rl/', '').replace('_', ' ').title()
                row[clean_key] = metrics.get(key, None)
            
            summary.append(row)
    
    df = pd.DataFrame(summary)
    
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Saved: {save_path}")
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Generate comparison plots from multiple model test results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Compare models using heatmap predictions (default):
    python generate_comparison_plots.py \\
        --models "GRPO=outputs/PNF-RL-GRPO" "PPO=outputs/PNF-RL-PPO" \\
        --output_dir comparison_plots/

    # Compare classification head predictions:
    python generate_comparison_plots.py \\
        --models "RL=outputs/RL" "Supervised=outputs/Supervised" \\
        --prediction_column image_level_cancer_logits \\
        --output_dir comparison_plots_cls/
    
    # Generate BOTH heatmap and classification head comparisons:
    python generate_comparison_plots.py \\
        --models "GRPO=outputs/GRPO" "PPO=outputs/PPO" \\
        --compare_cls \\
        --output_dir comparison_all/
        """
    )
    
    parser.add_argument(
        '--models', nargs='+', required=True,
        help='Model specifications in format "NAME=OUTPUT_DIR" (e.g., "GRPO=outputs/grpo_model")'
    )
    parser.add_argument(
        '--output_dir', default='comparison_plots',
        help='Directory to save comparison plots'
    )
    parser.add_argument(
        '--prediction_column', default='average_needle_heatmap_value',
        help='Column name for predictions (default: average_needle_heatmap_value). '
             'Use "image_level_cancer_logits" for classification head.'
    )
    parser.add_argument(
        '--formats', nargs='+', default=['png', 'pdf'],
        help='File formats to save (default: png pdf)'
    )
    parser.add_argument(
        '--compare_cls', action='store_true',
        help='Also generate comparison plots for classification head (image_level_cancer_logits)'
    )
    parser.add_argument(
        '--involvement_bins', type=str, default='0-20,20-40,40-60,60-80,80-100',
        help='Comma-separated involvement bins in format "low-high" (default: 0-20,20-40,40-60,60-80,80-100)'
    )

    
    args = parser.parse_args()
    
    # Parse model specifications
    model_configs = {}
    for spec in args.models:
        if '=' in spec:
            name, path = spec.split('=', 1)
        else:
            # Use directory name as model name
            name = os.path.basename(spec.rstrip('/'))
            path = spec
        model_configs[name] = path
    
    print(f"Comparing {len(model_configs)} models:")
    for name, path in model_configs.items():
        print(f"  - {name}: {path}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load all model results
    print("\n--- Loading Model Results ---")
    model_data = {}
    for name, output_dir in model_configs.items():
        try:
            preds, labels, df = load_model_results(output_dir, args.prediction_column)
            model_data[name] = (preds, labels, df)
        except Exception as e:
            print(f"Warning: Failed to load {name}: {e}")
    
    if len(model_data) < 1:
        print("Error: No models loaded successfully.")
        return
    
    # Generate standard comparison plots (ROC, PR, etc.)
    print("\n--- Generating Standard Comparison Plots ---")
    plotter = ModelComparisonPlotter()
    for name, (preds, labels, _) in model_data.items():
        plotter.add_model(name, preds, labels)
    
    figures = plotter.generate_all_plots(save_dir=args.output_dir, formats=args.formats)
    
    # Generate combined figure
    combined_path = os.path.join(args.output_dir, 'combined_comparison.png')
    plotter.generate_combined_figure(save_path=combined_path)
    
    # Generate involvement comparison plot
    print("\n--- Generating Involvement Comparison Plot ---")
    inv_path = os.path.join(args.output_dir, 'prediction_vs_involvement.png')
    generate_involvement_comparison_plot(model_data, save_path=inv_path)
    
    # Parse involvement bins
    bins = []
    for bin_str in args.involvement_bins.split(','):
        low, high = bin_str.strip().split('-')
        bins.append((float(low) / 100, float(high) / 100))
    
    # Generate error by involvement bins plot
    print("\n--- Generating Error by Involvement Bins Plot ---")
    error_path = os.path.join(args.output_dir, 'error_by_involvement.png')
    generate_error_by_involvement_bins_plot(model_data, bins=bins, save_path=error_path)
    
    # Generate thresholded error by involvement bins plot
    print("\n--- Generating Error by Involvement Bins Plot (Thresholded) ---")
    error_thresh_path = os.path.join(args.output_dir, 'error_by_involvement_thresholded.png')
    generate_error_by_involvement_bins_plot_thresholded(model_data, bins=bins, save_path=error_thresh_path)
    
    # Generate attention-specific comparison (for RL models)
    print("\n--- Generating Attention Metrics Comparison ---")
    attn_path = os.path.join(args.output_dir, 'attention_metrics_comparison.png')
    generate_attention_comparison_plot(model_configs, save_path=attn_path)
    
    # Generate summary table
    print("\n--- Generating Metrics Summary Table ---")
    summary_path = os.path.join(args.output_dir, 'metrics_comparison.csv')
    summary_df = generate_metrics_summary_table(model_configs, save_path=summary_path)
    print("\nMetrics Summary:")
    print(summary_df.to_string(index=False))
    
    # Generate AUROC vs Involvement Threshold plot (using heatmap predictions)
    print("\n--- Generating AUROC vs Involvement Threshold ---")
    auroc_vs_thresh_path = os.path.join(args.output_dir, 'auroc_vs_involvement_threshold.png')
    generate_auroc_vs_involvement_threshold_plot(model_data, save_path=auroc_vs_thresh_path)
    
    # Generate Activation vs Involvement plots (using heatmap predictions)
    print("\n--- Generating Activation vs Involvement (Normal) ---")
    activation_normal_path = os.path.join(args.output_dir, 'activation_vs_involvement_normal.png')
    generate_activation_vs_involvement_plot(model_data, use_thresholded=False, save_path=activation_normal_path)
    
    print("\n--- Generating Activation vs Involvement (Thresholded) ---")
    activation_thresh_path = os.path.join(args.output_dir, 'activation_vs_involvement_thresholded.png')
    generate_activation_vs_involvement_plot(model_data, use_thresholded=True, save_path=activation_thresh_path)
    
    # ==========================================
    # CLASSIFICATION HEAD COMPARISON (optional)
    # ==========================================
    if args.compare_cls:
        print("\n" + "="*60)
        print("GENERATING CLASSIFICATION HEAD COMPARISON PLOTS")
        print("="*60)
        
        cls_output_dir = os.path.join(args.output_dir, 'classification_head')
        os.makedirs(cls_output_dir, exist_ok=True)
        
        # Load data with classification head predictions
        model_data_cls = {}
        for name, output_dir in model_configs.items():
            try:
                preds, labels, df = load_model_results(output_dir, 'image_level_cancer_logits')
                model_data_cls[name] = (preds, labels, df)
            except Exception as e:
                print(f"  Skipping {name} (no classification head): {e}")
        
        if model_data_cls:
            # Generate standard comparison plots for classification head
            print("\n--- Classification Head: Standard Plots ---")
            plotter_cls = ModelComparisonPlotter()
            for name, (preds, labels, _) in model_data_cls.items():
                plotter_cls.add_model(name, preds, labels)
            
            plotter_cls.generate_all_plots(save_dir=cls_output_dir, formats=args.formats)
            
            # Generate combined figure
            combined_cls_path = os.path.join(cls_output_dir, 'combined_comparison_cls.png')
            plotter_cls.generate_combined_figure(save_path=combined_cls_path)
            
            # Generate error by involvement for classification head
            print("\n--- Classification Head: Error by Involvement ---")
            error_cls_path = os.path.join(cls_output_dir, 'error_by_involvement_cls.png')
            generate_error_by_involvement_bins_plot(model_data_cls, bins=bins, save_path=error_cls_path)
            
            # Generate threshold analysis (finds optimal thresholds)
            print("\n--- Classification Head: Threshold Analysis ---")
            threshold_analysis_path = os.path.join(cls_output_dir, 'threshold_analysis.png')
            _, optimal_thresholds = generate_threshold_analysis_plot(model_data_cls, save_path=threshold_analysis_path)
            
            # Save optimal thresholds to JSON
            import json
            thresholds_json_path = os.path.join(cls_output_dir, 'optimal_thresholds.json')
            with open(thresholds_json_path, 'w') as f:
                json.dump(optimal_thresholds, f, indent=2)
            print(f"Saved optimal thresholds to: {thresholds_json_path}")
            
            # For metrics and confusion matrices, use per-model optimal thresholds
            print("\n--- Classification Head: Metrics with Per-Model Optimal Thresholds ---")
            for name, thresh_info in optimal_thresholds.items():
                optimal_thresh_bal = thresh_info['threshold_balanced_accuracy']
                optimal_thresh_f1 = thresh_info['threshold_f1']
                print(f"  {name}: Bal Acc threshold={optimal_thresh_bal:.3f}, F1 threshold={optimal_thresh_f1:.3f}")

            # Generate classification metrics with per-model optimal thresholds
            print("\n--- Classification Metrics (Each Model at Own Optimal Threshold) ---")
            metrics_optimal_bal_path = os.path.join(cls_output_dir, 'classification_metrics_per_model_optimal_bal_acc.png')
            generate_classification_metrics_with_optimal_thresholds(
                model_data_cls, optimal_thresholds, metric_type='balanced_accuracy', save_path=metrics_optimal_bal_path
            )

            metrics_optimal_f1_path = os.path.join(cls_output_dir, 'classification_metrics_per_model_optimal_f1.png')
            generate_classification_metrics_with_optimal_thresholds(
                model_data_cls, optimal_thresholds, metric_type='f1', save_path=metrics_optimal_f1_path
            )

            # Generate ACCURACY (not MAE) by involvement bins
            print("\n--- Classification Accuracy by Involvement (All Cases) ---")
            acc_by_inv_path = os.path.join(cls_output_dir, 'accuracy_by_involvement.png')
            generate_classification_accuracy_by_involvement(
                model_data_cls, optimal_thresholds, metric_type='balanced_accuracy', 
                bins=bins, save_path=acc_by_inv_path, filter_condition=None
            )

            # Generate ACCURACY by involvement for HIGH INVOLVEMENT cases only
            print("\n--- Classification Accuracy by Involvement (High Involvement ≥40% Only) ---")
            acc_by_inv_high_path = os.path.join(cls_output_dir, 'accuracy_by_involvement_high_involvement.png')
            generate_classification_accuracy_by_involvement(
                model_data_cls, optimal_thresholds, metric_type='balanced_accuracy',
                bins=bins, save_path=acc_by_inv_high_path, filter_condition='high_involvement'
            )

            # Generate ACCURACY by involvement for csPCa cases only
            print("\n--- Classification Accuracy by Involvement (csPCa: GG>2 Only) ---")
            acc_by_inv_cspca_path = os.path.join(cls_output_dir, 'accuracy_by_involvement_cspca.png')
            generate_classification_accuracy_by_involvement(
                model_data_cls, optimal_thresholds, metric_type='balanced_accuracy',
                bins=bins, save_path=acc_by_inv_cspca_path, filter_condition='cspca'
            )

            # Generate confusion matrices with optimal thresholds (4 versions)
            print("\n--- Classification Head: Confusion Matrices (Per-Model Optimal Thresholds) ---")
            
            # 1. Balanced Accuracy
            thresh_dict_bal = {name: info['threshold_balanced_accuracy'] for name, info in optimal_thresholds.items()}
            cm_bal_path = os.path.join(cls_output_dir, 'confusion_matrices_optimal_bal_acc.png')
            generate_confusion_matrices_with_per_model_thresholds(
                model_data_cls, thresh_dict_bal, "(Optimal Balanced Accuracy)", save_path=cm_bal_path
            )
            
            # 2. F1 Score
            thresh_dict_f1 = {name: info['threshold_f1'] for name, info in optimal_thresholds.items()}
            cm_f1_path = os.path.join(cls_output_dir, 'confusion_matrices_optimal_f1.png')
            generate_confusion_matrices_with_per_model_thresholds(
                model_data_cls, thresh_dict_f1, "(Optimal F1 Score)", save_path=cm_f1_path
            )
            
            # 3. Sensitivity (90% target)
            thresh_dict_sens = {name: info['threshold_sensitivity'] for name, info in optimal_thresholds.items()}
            cm_sens_path = os.path.join(cls_output_dir, 'confusion_matrices_optimal_sensitivity.png')
            generate_confusion_matrices_with_per_model_thresholds(
                model_data_cls, thresh_dict_sens, "(Optimal Sensitivity target)", save_path=cm_sens_path
            )
            
            # 4. Specificity (90% target)
            thresh_dict_spec = {name: info['threshold_specificity'] for name, info in optimal_thresholds.items()}
            cm_spec_path = os.path.join(cls_output_dir, 'confusion_matrices_optimal_specificity.png')
            generate_confusion_matrices_with_per_model_thresholds(
                model_data_cls, thresh_dict_spec, "(Optimal Specificity target)", save_path=cm_spec_path
            )
            
            # Also keep 0.5 threshold for baseline comparison
            print("\n--- Classification Head: Confusion Matrices (Fixed Threshold 0.5) ---")
            cm_cls_path_fixed = os.path.join(cls_output_dir, 'confusion_matrices_threshold_0.5.png')
            generate_confusion_matrices_comparison(model_data_cls, save_path=cm_cls_path_fixed, threshold=0.5)


            # Generate calibration curves
            print("\n--- Classification Head: Calibration Curves ---")
            calib_cls_path = os.path.join(cls_output_dir, 'calibration_curves.png')
            generate_calibration_curves_comparison(model_data_cls, save_path=calib_cls_path, n_bins=10)



            print(f"\n✓ Classification head plots saved to: {cls_output_dir}")
        else:
            print("No models with classification head data found.")
    
    print(f"\n✓ All plots saved to: {args.output_dir}")
    print("\nGenerated files:")
    for f in sorted(os.listdir(args.output_dir)):
        fpath = os.path.join(args.output_dir, f)
        if os.path.isdir(fpath):
            print(f"  📁 {f}/")
            for sf in sorted(os.listdir(fpath)):
                print(f"      - {sf}")
        else:
            print(f"  - {f}")


if __name__ == '__main__':
    main()
