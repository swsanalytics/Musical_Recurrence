def plot_shap_heatmap(shap_values_single, X, y):
    """
    Create heatmap of SHAP values across features and outputs
    """
    shap_matrix = np.abs(shap_values_single[0])  # Shape: (n_features, n_outputs)
    
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    im = plt.imshow(shap_matrix.T, aspect='auto', cmap='YlOrRd')
    plt.colorbar(im, label='|SHAP| Value')
    
    # Set ticks
    plt.yticks(range(len(y.columns)), y.columns)
    plt.xticks(range(len(X.columns)), X.columns, rotation=45, ha='right')
    
    plt.xlabel('Features')
    plt.ylabel('Outputs')
    plt.title('SHAP Values Heatmap: Features × Outputs')
    
    # Add text annotations
    for i in range(len(y.columns)):
        for j in range(len(X.columns)):
            text = plt.text(j, i, f'{shap_matrix[j, i]:.3f}',
                           ha="center", va="center", color="black", fontsize=8)
    
    plt.tight_layout()
    plt.show()

# Call the function
plot_shap_heatmap(shap_values_single, X, y)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Reset to defaults
plt.rcParams.update(plt.rcParamsDefault)

def plot_shap_heatmap(shap_values_single, X, y):
    """
    Create heatmap with aerospace dashboard blue-grey color scheme
    """
    shap_matrix = np.abs(shap_values_single[0])
    
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('#f8fafc')
    
    # Create custom colormap with blues and greys (matching dashboard)
    colors_list = [
        '#f1f5f9',  # very light grey
        '#e2e8f0',  # light grey
        '#cbd5e1',  # grey
        '#94a3b8',  # blue-grey
        '#64748b',  # darker blue-grey
        '#475569',  # dark blue-grey
        '#4f46e5',  # primary blue
        '#4338ca',  # darker blue
        '#3730a3'   # darkest blue
    ]
    n_bins = 256
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list('dashboard', colors_list, N=n_bins)
    
    # Create heatmap
    im = ax.imshow(shap_matrix.T, aspect='auto', cmap=cmap, interpolation='nearest')
    
    # Style colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('|SHAP| Value', fontsize=11, color='#475569', weight='bold')
    cbar.ax.tick_params(labelsize=9, colors='#64748b')
    cbar.outline.set_edgecolor('#cbd5e1')
    cbar.outline.set_linewidth(1)
    
    # Set ticks
    ax.set_yticks(range(len(y.columns)))
    ax.set_yticklabels(y.columns, fontsize=10, color='#475569')
    ax.set_xticks(range(len(X.columns)))
    ax.set_xticklabels(X.columns, rotation=45, ha='right', fontsize=10, color='#475569')
    
    # Labels
    ax.set_xlabel('Features', fontsize=12, color='#1e293b', weight='bold')
    ax.set_ylabel('Outputs', fontsize=12, color='#1e293b', weight='bold')
    
    # Title
    ax.set_title('SHAP Values Heatmap: Features × Outputs', 
                fontsize=14, color='#1e293b', weight='bold', pad=20)
    
    # Add text annotations with smart coloring
    threshold = shap_matrix.max() * 0.6
    for i in range(len(y.columns)):
        for j in range(len(X.columns)):
            value = shap_matrix[j, i]
            # White text for dark cells, dark text for light cells
            text_color = 'white' if value > threshold else '#1e293b'
            text = ax.text(j, i, f'{value:.3f}',
                          ha="center", va="center", 
                          color=text_color, 
                          fontsize=9, 
                          weight='medium')
    
    # Add subtle grid
    ax.set_xticks(np.arange(len(X.columns)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(y.columns)) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linewidth=2)
    ax.tick_params(which='minor', size=0)
    
    # Style the plot
    ax.set_facecolor('#ffffff')
    for spine in ax.spines.values():
        spine.set_edgecolor('#cbd5e1')
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    plt.show()

def plot_normalized_shap_importance(shap_values_single, X, y):
    """
    Plot stacked bar chart with aerospace dashboard colors
    """
    shap_matrix = np.abs(shap_values_single[0])
    
    # Normalize by output
    shap_normalized = shap_matrix / (shap_matrix.sum(axis=0) + 1e-10)
    
    # Create DataFrame
    df_norm = pd.DataFrame(shap_normalized, index=X.columns, columns=y.columns)
    
    # Sort by total contribution
    df_norm['total'] = df_norm.sum(axis=1)
    df_norm = df_norm.sort_values('total', ascending=False).drop('total', axis=1)
    
    # Select top 10 features
    top_features = df_norm.head(10)
    
    # Dashboard color palette with blues, purples, and greys
    feature_colors = [
        '#4f46e5',  # primary blue
        '#6366f1',  # lighter blue  
        '#818cf8',  # light blue
        '#10b981',  # teal/green
        '#64748b',  # grey-blue
        '#94a3b8',  # light grey-blue
        '#8b5cf6',  # purple
        '#a78bfa',  # light purple
        '#475569',  # dark grey
        '#cbd5e1'   # light grey
    ]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('#f8fafc')
    
    # Create the stacked bar chart
    bottom = np.zeros(len(y.columns))
    
    for idx, (feature_name, row) in enumerate(top_features.iterrows()):
        color = feature_colors[idx % len(feature_colors)]
        bars = ax.bar(range(len(y.columns)), 
                      row.values, 
                      bottom=bottom, 
                      label=feature_name,
                      color=color,
                      edgecolor='white',
                      linewidth=1.5,
                      alpha=0.9)
        
        # Add percentage labels
        for i, (val, b) in enumerate(zip(row.values, bottom)):
            if val > 0.03:  # Only show if segment is large enough
                percentage = val * 100
                # Use white text on darker colors
                text_color = 'white' if idx < 5 else '#1e293b'
                ax.text(i, b + val/2, f'{percentage:.0f}%', 
                       ha='center', va='center',
                       color=text_color,
                       fontsize=9, weight='bold')
        
        bottom += row.values
    
    # Styling
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.6, len(y.columns) - 0.4)
    
    # Set ticks and labels
    ax.set_xticks(range(len(y.columns)))
    ax.set_xticklabels(y.columns, fontsize=11, color='#475569', rotation=45, ha='right')
    
    # Y-axis as percentages
    y_ticks = np.arange(0, 1.1, 0.1)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{int(x*100)}%' for x in y_ticks], fontsize=10, color='#64748b')
    
    # Labels
    ax.set_xlabel('Output', fontsize=12, color='#1e293b', weight='bold')
    ax.set_ylabel('Normalized Feature Contribution (%)', fontsize=12, color='#1e293b', weight='bold')
    
    # Title
    ax.set_title('Relative Feature Importance Distribution Across Outputs', 
                fontsize=14, color='#1e293b', weight='bold', pad=20)
    
    # Grid with dashboard styling
    ax.grid(axis='y', alpha=0.4, color='#e2e8f0', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Background color
    ax.set_facecolor('#ffffff')
    
    # Style spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cbd5e1')
    ax.spines['bottom'].set_color('#cbd5e1')
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # Legend with dashboard styling
    legend = ax.legend(title='Features', 
                      bbox_to_anchor=(1.05, 1), 
                      loc='upper left',
                      frameon=True,
                      facecolor='white',
                      edgecolor='#cbd5e1',
                      fontsize=10)
    legend.get_frame().set_linewidth(1.5)
    legend.get_title().set_color('#475569')
    legend.get_title().set_weight('bold')
    
    plt.tight_layout()
    plt.show()

# Alternative: Purple-blue gradient version
def plot_shap_heatmap_purple(shap_values_single, X, y):
    """
    Create heatmap with purple-blue gradient matching dashboard
    """
    shap_matrix = np.abs(shap_values_single[0])
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Use seaborn's mako colormap (blue-purple) or create custom
    cmap = 'mako_r'  # Reversed mako goes from light to dark blue-purple
    # Or use: cmap = 'BuPu' for Blue-Purple colormap
    
    # Create heatmap
    im = ax.imshow(shap_matrix.T, aspect='auto', cmap=cmap, interpolation='nearest')
    
    # Rest of the styling remains the same...
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('|SHAP| Value', fontsize=11, color='#475569', weight='bold')
    
    ax.set_yticks(range(len(y.columns)))
    ax.set_yticklabels(y.columns, fontsize=10, color='#475569')
    ax.set_xticks(range(len(X.columns)))
    ax.set_xticklabels(X.columns, rotation=45, ha='right', fontsize=10, color='#475569')
    
    ax.set_xlabel('Features', fontsize=12, color='#1e293b', weight='bold')
    ax.set_ylabel('Outputs', fontsize=12, color='#1e293b', weight='bold')
    ax.set_title('SHAP Values Heatmap: Features × Outputs', 
                fontsize=14, color='#1e293b', weight='bold', pad=20)
    
    # Add text annotations
    threshold = shap_matrix.max() * 0.5
    for i in range(len(y.columns)):
        for j in range(len(X.columns)):
            value = shap_matrix[j, i]
            text_color = 'white' if value > threshold else '#1e293b'
            text = ax.text(j, i, f'{value:.3f}',
                          ha="center", va="center", 
                          color=text_color, 
                          fontsize=9, 
                          weight='medium')
    
    # Grid
    ax.set_xticks(np.arange(len(X.columns)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(y.columns)) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linewidth=2)
    ax.tick_params(which='minor', size=0)
    
    ax.set_facecolor('#f8fafc')
    for spine in ax.spines.values():
        spine.set_edgecolor('#cbd5e1')
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    plt.show()

# Call the functions with the blue-grey theme
plot_shap_heatmap(shap_values_single, X, y)
plot_normalized_shap_importance(shap_values_single, X, y)

# Or try the purple-blue version:
# plot_shap_heatmap_purple(shap_values_single, X, y)


def create_waterfall_plot(shap_values, X, y, instance_idx=0, output_idx=0):
    """Create waterfall plot for a specific instance and output"""
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values[instance_idx][:, output_idx],
            base_values=0,
            data=X.iloc[instance_idx],
            feature_names=X.columns.tolist()
        )
    )

def create_force_plot(shap_values, X, y, instance_idx=0, output_idx=0):
    """Create force plot for a specific instance and output"""
    shap.force_plot(
        base_value=0,
        shap_values=shap_values[instance_idx][:, output_idx],
        features=X.iloc[instance_idx],
        feature_names=X.columns.tolist()
    )

def create_summary_plot_multi_output(shap_values, X, y):
    """Create summary plots for all outputs"""
    fig, axes = plt.subplots(1, len(y.columns), figsize=(5*len(y.columns), 4))
    
    for i, output_name in enumerate(y.columns):
        plt.sca(axes[i] if len(y.columns) > 1 else axes)
        shap.summary_plot(
            shap_values[:, :, i],
            X,
            show=False,
            plot_size=None
        )
        axes[i].set_title(f'SHAP Summary: {output_name}')
    
    plt.tight_layout()
    plt.show()


plot_normalized_shap_importance(shap_values_single, X, y)
