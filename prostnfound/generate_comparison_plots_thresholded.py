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

