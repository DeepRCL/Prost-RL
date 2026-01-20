"""
Model Comparison Plots for Publication

Generate comparison plots to compare different models:
- Sensitivity vs Specificity (Operating Point Analysis)
- ROC Curves with AUC
- Precision-Recall Curves with AP
- Detection Error Tradeoff (DET) curves
- Calibration Curves
- Violin plots for score distributions

Usage:
    from prostnfound.src.model_comparison_plots import ModelComparisonPlotter
    
    plotter = ModelComparisonPlotter()
    plotter.add_model('RL Model', predictions_rl, labels)
    plotter.add_model('Baseline', predictions_baseline, labels)
    plotter.generate_all_plots(save_dir='plots/')
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    confusion_matrix, 
)
from sklearn.calibration import calibration_curve
from typing import Dict, List, Optional, Tuple
import os



class ModelComparisonPlotter:
    """
    Generate publication-quality comparison plots for multiple models.
    
    Key plots for model comparison:
    1. ROC Curves - Overall discrimination ability
    2. Sensitivity vs Specificity - Operating point analysis
    3. Precision-Recall Curves - Important for imbalanced datasets
    4. DET Curves - Error rate visualization
    5. Calibration Curves - Prediction reliability
    6. Score Distributions - Understanding model behavior
    7. Performance at Fixed Sensitivity/Specificity
    """
    
    # Vibrant color palette for publication
    COLORS = [
        '#2E86AB',  # Steel Blue
        '#E94F37',  # Cardinal Red
        '#2E8B57',  # Sea Green
        '#9B59B6',  # Purple
        '#F39C12',  # Orange
        '#1ABC9C',  # Turquoise
        '#E74C3C',  # Bright Red
        '#3498DB',  # Dodger Blue
    ]
    
    LINESTYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2))]
    
    def __init__(self, figsize: Tuple[int, int] = (10, 8)):
        """
        Initialize the plotter.
        
        Args:
            figsize: Default figure size for plots
        """
        self.figsize = figsize
        self.models: Dict[str, Dict] = {}
        
        # Set publication-quality defaults
        plt.rcParams.update({
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 11,
            'figure.dpi': 150,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
        })
    
    def add_model(
        self,
        name: str,
        predictions: np.ndarray,
        labels: np.ndarray,
        color: Optional[str] = None,
        linestyle: Optional[str] = None,
    ):
        """
        Add a model's predictions for comparison.
        
        Args:
            name: Model name for legend
            predictions: Predicted probabilities or scores (N,)
            labels: Ground truth binary labels (N,)
            color: Optional custom color
            linestyle: Optional custom linestyle
        """
        idx = len(self.models)
        self.models[name] = {
            'predictions': np.asarray(predictions).ravel(),
            'labels': np.asarray(labels).ravel(),
            'color': color or self.COLORS[idx % len(self.COLORS)],
            'linestyle': linestyle or self.LINESTYLES[idx % len(self.LINESTYLES)],
        }
    
    def clear_models(self):
        """Reset all models."""
        self.models = {}
    
    def _validate_models(self):
        """Check that we have models to plot."""
        if len(self.models) == 0:
            raise ValueError("No models added. Use add_model() first.")
    
    # =========================================================================
    # PLOT 1: ROC CURVES
    # =========================================================================
    def plot_roc_curves(
        self,
        ax: Optional[plt.Axes] = None,
        show_auc: bool = True,
        show_chance: bool = True,
        title: str = 'ROC Curve Comparison',
    ) -> plt.Figure:
        """
        Plot ROC curves for all models.
        
        The ROC curve shows the trade-off between sensitivity (TPR) and 
        1 - specificity (FPR). Higher AUC indicates better discrimination.
        """
        self._validate_models()
        
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure
        
        for name, data in self.models.items():
            fpr, tpr, _ = roc_curve(data['labels'], data['predictions'])
            auc = roc_auc_score(data['labels'], data['predictions'])
            
            label = f"{name}"
            if show_auc:
                label += f" (AUC = {auc:.3f})"
            
            ax.plot(
                fpr, tpr,
                color=data['color'],
                linestyle=data['linestyle'],
                linewidth=2.5,
                label=label,
            )
        
        if show_chance:
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1, label='Chance')
        
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.set_xlabel('False Positive Rate (1 - Specificity)')
        ax.set_ylabel('True Positive Rate (Sensitivity)')
        ax.set_title(title)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        return fig
    
    # =========================================================================
    # PLOT 2: SENSITIVITY vs SPECIFICITY
    # =========================================================================
    def plot_sensitivity_vs_specificity(
        self,
        ax: Optional[plt.Axes] = None,
        highlight_points: Optional[Dict[str, float]] = None,
        title: str = 'Sensitivity vs Specificity',
    ) -> plt.Figure:
        """
        Plot Sensitivity vs Specificity curves.
        
        This plot shows both metrics directly (unlike ROC which shows 1-specificity).
        The upper-right corner represents perfect classification.
        
        Args:
            ax: Optional axes to plot on
            highlight_points: Dict of {name: threshold} to mark operating points
        """
        self._validate_models()
        
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure
        
        for name, data in self.models.items():
            fpr, tpr, thresholds = roc_curve(data['labels'], data['predictions'])
            specificity = 1 - fpr  # Specificity = 1 - FPR
            sensitivity = tpr  # Sensitivity = TPR
            
            ax.plot(
                specificity, sensitivity,
                color=data['color'],
                linestyle=data['linestyle'],
                linewidth=2.5,
                label=name,
            )
            
            # Highlight specific operating points if provided
            if highlight_points and name in highlight_points:
                thresh = highlight_points[name]
                # Find closest threshold
                idx = np.argmin(np.abs(thresholds - thresh))
                ax.scatter(
                    [specificity[idx]], [sensitivity[idx]],
                    color=data['color'],
                    s=100, marker='o', zorder=5,
                    edgecolors='white', linewidth=2,
                )
                ax.annotate(
                    f'τ={thresh:.2f}',
                    (specificity[idx], sensitivity[idx]),
                    xytext=(10, -10), textcoords='offset points',
                    fontsize=10, color=data['color'],
                )
        
        ax.set_xlim([0, 1.02])
        ax.set_ylim([0, 1.02])
        ax.set_xlabel('Specificity')
        ax.set_ylabel('Sensitivity')
        ax.set_title(title)
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)
        
        # Note: Upper-right is ideal
        ax.annotate(
            'Ideal',
            (1, 1), xytext=(-30, -20), textcoords='offset points',
            fontsize=10, alpha=0.5,
            arrowprops=dict(arrowstyle='->', alpha=0.5)
        )
        
        return fig
    
    # =========================================================================
    # PLOT 3: PRECISION-RECALL CURVES
    # =========================================================================
    def plot_precision_recall_curves(
        self,
        ax: Optional[plt.Axes] = None,
        show_ap: bool = True,
        title: str = 'Precision-Recall Curve Comparison',
    ) -> plt.Figure:
        """
        Plot Precision-Recall curves for all models.
        
        Important for imbalanced datasets where positive class is rare.
        Higher AP (Average Precision) indicates better performance.
        """
        self._validate_models()
        
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure
        
        for name, data in self.models.items():
            precision, recall, _ = precision_recall_curve(
                data['labels'], data['predictions']
            )
            ap = average_precision_score(data['labels'], data['predictions'])
            
            label = f"{name}"
            if show_ap:
                label += f" (AP = {ap:.3f})"
            
            ax.plot(
                recall, precision,
                color=data['color'],
                linestyle=data['linestyle'],
                linewidth=2.5,
                label=label,
            )
        
        # Show class balance line
        first_model = list(self.models.values())[0]
        baseline = first_model['labels'].mean()
        ax.axhline(y=baseline, color='gray', linestyle='--', alpha=0.5, 
                   label=f'Baseline (prevalence={baseline:.2f})')
        
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.set_xlabel('Recall (Sensitivity)')
        ax.set_ylabel('Precision (PPV)')
        ax.set_title(title)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        return fig
    
    # =========================================================================
    # PLOT 4: PERFORMANCE AT FIXED OPERATING POINTS
    # =========================================================================
    def plot_performance_at_fixed_sensitivity(
        self,
        sensitivities: List[float] = [0.80, 0.85, 0.90, 0.95],
        ax: Optional[plt.Axes] = None,
        title: str = 'Specificity at Fixed Sensitivity Levels',
    ) -> plt.Figure:
        """
        Bar chart showing specificity achieved at various sensitivity thresholds.
        
        Useful for comparing models at clinically relevant operating points.
        """
        self._validate_models()
        
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure
        
        model_names = list(self.models.keys())
        x = np.arange(len(sensitivities))
        width = 0.8 / len(model_names)
        
        for i, (name, data) in enumerate(self.models.items()):
            fpr, tpr, _ = roc_curve(data['labels'], data['predictions'])
            specificity = 1 - fpr
            
            specificities_at_sens = []
            for target_sens in sensitivities:
                # Find indices where sensitivity >= target
                valid_indices = np.where(tpr >= target_sens)[0]
                if len(valid_indices) > 0:
                    # Get the best specificity at this sensitivity level
                    best_idx = valid_indices[np.argmax(specificity[valid_indices])]
                    specificities_at_sens.append(specificity[best_idx])
                else:
                    specificities_at_sens.append(0)
            
            bars = ax.bar(
                x + i * width - width * len(model_names) / 2 + width / 2,
                specificities_at_sens,
                width,
                label=name,
                color=data['color'],
                edgecolor='white',
                linewidth=0.5,
            )
            
            # Add value labels on bars
            for bar, val in zip(bars, specificities_at_sens):
                ax.annotate(
                    f'{val:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=9,
                )
        
        ax.set_xlabel('Target Sensitivity')
        ax.set_ylabel('Achieved Specificity')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{s:.0%}' for s in sensitivities])
        ax.legend()
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')
        
        return fig
    
    def plot_performance_at_fixed_specificity(
        self,
        specificities: List[float] = [0.70, 0.80, 0.90, 0.95],
        ax: Optional[plt.Axes] = None,
        title: str = 'Sensitivity at Fixed Specificity Levels',
    ) -> plt.Figure:
        """
        Bar chart showing sensitivity achieved at various specificity thresholds.
        """
        self._validate_models()
        
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure
        
        model_names = list(self.models.keys())
        x = np.arange(len(specificities))
        width = 0.8 / len(model_names)
        
        for i, (name, data) in enumerate(self.models.items()):
            fpr, tpr, _ = roc_curve(data['labels'], data['predictions'])
            specificity = 1 - fpr
            
            sensitivities_at_spec = []
            for target_spec in specificities:
                # Find indices where specificity >= target
                valid_indices = np.where(specificity >= target_spec)[0]
                if len(valid_indices) > 0:
                    # Get the best sensitivity at this specificity level
                    best_idx = valid_indices[np.argmax(tpr[valid_indices])]
                    sensitivities_at_spec.append(tpr[best_idx])
                else:
                    sensitivities_at_spec.append(0)
            
            bars = ax.bar(
                x + i * width - width * len(model_names) / 2 + width / 2,
                sensitivities_at_spec,
                width,
                label=name,
                color=data['color'],
                edgecolor='white',
                linewidth=0.5,
            )
            
            # Add value labels on bars
            for bar, val in zip(bars, sensitivities_at_spec):
                ax.annotate(
                    f'{val:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=9,
                )
        
        ax.set_xlabel('Target Specificity')
        ax.set_ylabel('Achieved Sensitivity')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{s:.0%}' for s in specificities])
        ax.legend()
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')
        
        return fig
    
    # =========================================================================
    # PLOT 5: SCORE DISTRIBUTION PLOTS
    # =========================================================================
    def plot_score_distributions(
        self,
        ax: Optional[plt.Axes] = None,
        plot_type: str = 'violin',  # 'violin', 'box', 'hist'
        title: str = 'Score Distributions by Class',
    ) -> plt.Figure:
        """
        Plot score distributions for positive and negative classes.
        
        Helps understand model calibration and separation.
        
        Args:
            plot_type: 'violin', 'box', or 'hist'
        """
        self._validate_models()
        
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure
        
        if plot_type == 'hist':
            # Overlapping histograms
            for name, data in self.models.items():
                preds = data['predictions']
                labels = data['labels']
                
                ax.hist(
                    preds[labels == 0], bins=30, alpha=0.5,
                    label=f'{name} (Negative)', color=self.COLORS[0],
                    density=True, edgecolor='none'
                )
                ax.hist(
                    preds[labels == 1], bins=30, alpha=0.5,
                    label=f'{name} (Positive)', color=self.COLORS[1],
                    density=True, edgecolor='none'
                )
                break  # Only first model for histogram
            
            ax.set_xlabel('Prediction Score')
            ax.set_ylabel('Density')
        else:
            # Violin or box plots
            plot_data = []
            positions = []
            colors = []
            labels_list = []
            
            for i, (name, data) in enumerate(self.models.items()):
                preds = data['predictions']
                labels = data['labels']
                
                # Negative class
                plot_data.append(preds[labels == 0])
                positions.append(i * 3)
                colors.append(self.COLORS[0])
                labels_list.append(f'{name}\n(Negative)')
                
                # Positive class
                plot_data.append(preds[labels == 1])
                positions.append(i * 3 + 1)
                colors.append(self.COLORS[1])
                labels_list.append(f'{name}\n(Positive)')
            
            if plot_type == 'violin':
                parts = ax.violinplot(plot_data, positions=positions, showmeans=True)
                for i, pc in enumerate(parts['bodies']):
                    pc.set_facecolor(colors[i])
                    pc.set_alpha(0.7)
            else:  # box
                bp = ax.boxplot(plot_data, positions=positions, patch_artist=True)
                for i, patch in enumerate(bp['boxes']):
                    patch.set_facecolor(colors[i])
                    patch.set_alpha(0.7)
            
            ax.set_xticks(positions)
            ax.set_xticklabels(labels_list, fontsize=10)
            ax.set_ylabel('Prediction Score')
        
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')
        
        return fig
    
    # =========================================================================
    # PLOT 6: CALIBRATION CURVES
    # =========================================================================
    def plot_calibration_curves(
        self,
        ax: Optional[plt.Axes] = None,
        n_bins: int = 10,
        title: str = 'Calibration Curves',
    ) -> plt.Figure:
        """
        Plot calibration curves showing predicted vs actual probabilities.
        
        A well-calibrated model should have points close to the diagonal.
        """
        self._validate_models()
        
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
        else:
            fig = ax.figure
        
        for name, data in self.models.items():
            prob_true, prob_pred = calibration_curve(
                data['labels'], data['predictions'], n_bins=n_bins
            )
            
            ax.plot(
                prob_pred, prob_true,
                marker='o', markersize=8,
                color=data['color'],
                linestyle=data['linestyle'],
                linewidth=2,
                label=name,
            )
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
        
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title(title)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        return fig
    
    # =========================================================================
    # PLOT 7: CONFUSION MATRICES AT THRESHOLD
    # =========================================================================
    def plot_confusion_matrices(
        self,
        threshold: float = 0.5,
        figsize: Optional[Tuple[int, int]] = None,
        title: str = 'Confusion Matrices at threshold={threshold}',
    ) -> plt.Figure:
        """
        Plot confusion matrices for all models at a given threshold.
        """
        self._validate_models()
        
        n_models = len(self.models)
        ncols = min(3, n_models)
        nrows = (n_models + ncols - 1) // ncols
        
        if figsize is None:
            figsize = (5 * ncols, 4 * nrows)
        
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, (name, data) in enumerate(self.models.items()):
            preds_binary = (data['predictions'] >= threshold).astype(int)
            cm = confusion_matrix(data['labels'], preds_binary)
            
            sns.heatmap(
                cm, annot=True, fmt='d', cmap='Blues',
                ax=axes[i], cbar=False,
                xticklabels=['Predicted Neg', 'Predicted Pos'],
                yticklabels=['Actual Neg', 'Actual Pos'],
            )
            axes[i].set_title(f'{name}')
        
        # Hide unused axes
        for i in range(n_models, len(axes)):
            axes[i].axis('off')
        
        fig.suptitle(title.format(threshold=threshold), fontsize=14)
        plt.tight_layout()
        
        return fig
    
    # =========================================================================
    # PLOT 8: SUMMARY METRICS TABLE
    # =========================================================================
    def get_summary_metrics(
        self,
        threshold: float = 0.5,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute summary metrics for all models.
        
        Returns:
            Dict mapping model name to metrics dict
        """
        self._validate_models()
        
        results = {}
        for name, data in self.models.items():
            preds = data['predictions']
            labels = data['labels']
            preds_binary = (preds >= threshold).astype(int)
            
            fpr, tpr, _ = roc_curve(labels, preds)
            specificity = 1 - fpr
            
            # Find operating point for 90% sensitivity
            idx_90_sens = np.argmin(np.abs(tpr - 0.90))
            
            results[name] = {
                'AUC': roc_auc_score(labels, preds),
                'AP': average_precision_score(labels, preds),
                f'Sensitivity@{threshold}': tpr[np.argmin(np.abs(specificity - (1 - fpr[np.argmin(np.abs(preds.mean() - threshold))])))] if len(tpr) > 0 else 0,
                'Specificity@90%Sens': specificity[idx_90_sens],
                'Sensitivity@90%Spec': tpr[np.argmin(np.abs(specificity - 0.90))],
            }
        
        return results
    
    def plot_summary_table(
        self,
        ax: Optional[plt.Axes] = None,
        threshold: float = 0.5,
        title: str = 'Model Performance Summary',
    ) -> plt.Figure:
        """
        Create a table visualization of summary metrics.
        """
        self._validate_models()
        
        metrics = self.get_summary_metrics(threshold)
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))
        else:
            fig = ax.figure
        
        ax.axis('off')
        
        # Prepare table data
        columns = ['Model', 'AUC', 'AP', 'Spec@90%Sens', 'Sens@90%Spec']
        cell_text = []
        
        for name, m in metrics.items():
            cell_text.append([
                name,
                f"{m['AUC']:.3f}",
                f"{m['AP']:.3f}",
                f"{m['Specificity@90%Sens']:.3f}",
                f"{m['Sensitivity@90%Spec']:.3f}",
            ])
        
        table = ax.table(
            cellText=cell_text,
            colLabels=columns,
            loc='center',
            cellLoc='center',
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.8)
        
        # Style header
        for i in range(len(columns)):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(color='white', weight='bold')
        
        ax.set_title(title, fontsize=14, pad=20)
        
        return fig
    
    # =========================================================================
    # GENERATE ALL PLOTS
    # =========================================================================
    def generate_all_plots(
        self,
        save_dir: Optional[str] = None,
        prefix: str = '',
        formats: List[str] = ['png', 'pdf'],
    ) -> Dict[str, plt.Figure]:
        """
        Generate all comparison plots and optionally save them.
        
        Args:
            save_dir: Directory to save plots (None = don't save)
            prefix: Prefix for filenames
            formats: File formats to save (e.g., ['png', 'pdf'])
            
        Returns:
            Dictionary of figure names to Figure objects
        """
        self._validate_models()
        
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
        
        figures = {}
        
        # Generate each plot type
        print("Generating comparison plots...")
        
        # 1. ROC Curves
        fig = self.plot_roc_curves()
        figures['roc_curves'] = fig
        
        # 2. Sensitivity vs Specificity
        fig = self.plot_sensitivity_vs_specificity()
        figures['sensitivity_vs_specificity'] = fig
        
        # 3. Precision-Recall Curves
        fig = self.plot_precision_recall_curves()
        figures['precision_recall'] = fig
        
        # 4. Performance at Fixed Sensitivity
        fig = self.plot_performance_at_fixed_sensitivity()
        figures['performance_at_sensitivity'] = fig
        
        # 5. Performance at Fixed Specificity
        fig = self.plot_performance_at_fixed_specificity()
        figures['performance_at_specificity'] = fig
        
        # 6. Score Distributions
        fig = self.plot_score_distributions(plot_type='violin')
        figures['score_distributions'] = fig
        
        # 7. Calibration Curves
        fig = self.plot_calibration_curves()
        figures['calibration'] = fig
        
        # 8. Confusion Matrices
        fig = self.plot_confusion_matrices()
        figures['confusion_matrices'] = fig
        
        # 9. Summary Table
        fig = self.plot_summary_table()
        figures['summary_table'] = fig
        
        # Save if directory provided
        if save_dir is not None:
            for name, fig in figures.items():
                for fmt in formats:
                    filename = os.path.join(save_dir, f'{prefix}{name}.{fmt}')
                    fig.savefig(filename, dpi=300, bbox_inches='tight')
                    print(f"  Saved: {filename}")
        
        print(f"Generated {len(figures)} plots.")
        return figures
    
    def generate_combined_figure(
        self,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Generate a single combined figure with key plots for publication.
        
        Layout:
        - Top row: ROC curves, Sens vs Spec
        - Bottom row: PR curves, Performance at fixed sensitivity
        """
        self._validate_models()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        self.plot_roc_curves(ax=axes[0, 0])
        self.plot_sensitivity_vs_specificity(ax=axes[0, 1])
        self.plot_precision_recall_curves(ax=axes[1, 0])
        self.plot_performance_at_fixed_sensitivity(ax=axes[1, 1])
        
        plt.tight_layout()
        
        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved combined figure to: {save_path}")
        
        return fig


# ==============================================================================
# CONVENIENCE FUNCTION FOR QUICK COMPARISON
# ==============================================================================
def compare_models(
    models_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_dir: Optional[str] = None,
    prefix: str = '',
) -> Dict[str, plt.Figure]:
    """
    Quick function to compare multiple models.
    
    Args:
        models_data: Dict mapping model name to (predictions, labels) tuple
        save_dir: Optional directory to save plots
        prefix: Prefix for saved filenames
        
    Returns:
        Dictionary of figure names to Figure objects
        
    Example:
        >>> models_data = {
        ...     'RL Model': (preds_rl, labels),
        ...     'Baseline': (preds_baseline, labels),
        ... }
        >>> figures = compare_models(models_data, save_dir='plots/')
    """
    plotter = ModelComparisonPlotter()
    
    for name, (preds, labels) in models_data.items():
        plotter.add_model(name, preds, labels)
    
    return plotter.generate_all_plots(save_dir=save_dir, prefix=prefix)


if __name__ == '__main__':
    # Demo with synthetic data
    np.random.seed(42)
    n_samples = 500
    
    # Generate synthetic labels
    labels = np.random.binomial(1, 0.3, n_samples)
    
    # Generate synthetic predictions for different models
    # Model 1: RL Model (better discrimination)
    preds_rl = np.random.beta(2, 5, n_samples)
    preds_rl[labels == 1] = np.random.beta(5, 2, labels.sum())
    
    # Model 2: Baseline (worse discrimination)
    preds_baseline = np.random.beta(2, 3, n_samples)
    preds_baseline[labels == 1] = np.random.beta(3, 2, labels.sum())
    
    # Model 3: Supervised
    preds_supervised = np.random.beta(2, 4, n_samples)
    preds_supervised[labels == 1] = np.random.beta(4, 2, labels.sum())
    
    # Create comparison plots
    plotter = ModelComparisonPlotter()
    plotter.add_model('RL Attention', preds_rl, labels)
    plotter.add_model('Supervised', preds_supervised, labels)
    plotter.add_model('Baseline', preds_baseline, labels)
    
    # Generate combined figure for demonstration
    fig = plotter.generate_combined_figure()
    plt.show()
    
    # Print summary metrics
    metrics = plotter.get_summary_metrics()
    print("\nSummary Metrics:")
    for name, m in metrics.items():
        print(f"\n{name}:")
        for k, v in m.items():
            print(f"  {k}: {v:.3f}")
