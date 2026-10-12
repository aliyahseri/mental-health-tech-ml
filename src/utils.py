"""
Utility functions for the Mental Health in Tech Survey ML project.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def ensure_dir(directory):
    """
    Ensure a directory exists, create if it doesn't.
    
    Parameters:
    -----------
    directory : str
        Path to directory
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_figure(fig, filename, output_dir='../outputs/figures', dpi=300):
    """
    Save a matplotlib figure to the specified directory.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure object to save
    filename : str
        Name of the file (without extension)
    output_dir : str
        Directory to save to
    dpi : int
        Resolution in dots per inch
    """
    ensure_dir(output_dir)
    filepath = os.path.join(output_dir, f'{filename}.png')
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f"Saved figure to {filepath}")


def plot_confusion_matrix(cm, class_names, title='Confusion Matrix'):
    """
    Plot a confusion matrix heatmap.
    
    Parameters:
    -----------
    cm : array-like
        Confusion matrix
    class_names : list
        List of class names
    title : str
        Plot title
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title)
    return fig


def print_metrics_report(y_true, y_pred, model_name):
    """
    Print classification metrics.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    model_name : str
        Name of the model
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    print(f"\n{'='*50}")
    print(f"{model_name} - Performance Metrics")
    print(f"{'='*50}")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, average='macro'):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, average='macro'):.4f}")
    print(f"F1 Score:  {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(f"{'='*50}\n")


def plot_feature_importance(importances, feature_names, top_n=15, title='Feature Importance'):
    """
    Plot feature importance as a bar chart.
    
    Parameters:
    -----------
    importances : array-like
        Feature importance values
    feature_names : list
        List of feature names
    top_n : int
        Number of top features to display
    title : str
        Plot title
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    # Create DataFrame and sort
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(data=importance_df, x='importance', y='feature', 
                palette='viridis', ax=ax)
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    ax.set_title(title)
    plt.tight_layout()
    
    return fig


def plot_model_comparison(metrics_dict, models):
    """
    Plot side-by-side comparison of model metrics.
    
    Parameters:
    -----------
    metrics_dict : dict
        Dictionary with metric names as keys and lists of values as values
    models : list
        List of model names
        
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(models))
    width = 0.2
    
    metrics = list(metrics_dict.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        offset = (i - len(metrics)/2 + 0.5) * width
        ax.bar(x + offset, metrics_dict[metric], width, label=metric, color=color)
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim([0, 1])
    plt.tight_layout()
    
    return fig
