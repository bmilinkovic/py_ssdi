"""
Visualization tools for plotting results.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def plot_optimization_history(history, title="Dynamical Dependence Optimization", positive_dd=True, 
                         use_log_scale='auto'):
    """
    Plot the optimization history of dynamical dependence.
    
    Parameters
    ----------
    history : list or ndarray
        History of dynamical dependence values during optimization.
        If a list of lists is provided (multiple restarts), the best history will be plotted.
    title : str, optional
        Plot title
    positive_dd : bool, optional
        Whether the history contains positive DD values (MATLAB-style) or original DD values
    use_log_scale : str or bool, optional
        Whether to use a logarithmic scale for the y-axis.
        'auto': Use log scale if the range of values exceeds 1000x
        True: Always use log scale
        False: Never use log scale
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Handle multiple histories (from multiple restarts)
    if isinstance(history, list) and len(history) > 0 and isinstance(history[0], list):
        # Find the best history (the one with the lowest final value)
        best_history = min(history, key=lambda h: h[-1])
        history = best_history
    
    # Determine if we should use log scale
    if use_log_scale == 'auto':
        if min(history) > 0:  # Only consider positive values for log scale
            max_val = max(history)
            min_val = min(history)
            use_log = (max_val / min_val > 1000)
        else:
            use_log = False
    else:
        use_log = use_log_scale
    
    # Plot the optimization history
    ax.plot(history, 'b-', linewidth=2)
    
    # Apply log scale if needed
    if use_log:
        ax.set_yscale('log')
    
    # Customize the plot
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Dynamical Dependence')
    ax.set_title(title)
    ax.grid(True)
    
    return fig


def plot_optimization_runs(all_histories, best_idx=None, use_log_scale='auto'):
    """
    Plot all optimization runs on the same figure.
    
    Parameters
    ----------
    all_histories : list of lists
        Histories of dynamical dependence values for all runs
    best_idx : int, optional
        Index of the best run
    use_log_scale : str or bool, optional
        Whether to use a logarithmic scale for the y-axis.
        'auto': Use log scale if the range of values exceeds 1000x
        True: Always use log scale
        False: Never use log scale
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Flatten all values for scale determination
    all_values = [v for history in all_histories for v in history if not np.isnan(v)]
    
    # Determine if we should use log scale
    if use_log_scale == 'auto':
        if all(v > 0 for v in all_values):  # Only consider positive values for log scale
            max_val = max(all_values)
            min_val = min(all_values)
            use_log = (max_val / min_val > 1000)
        else:
            use_log = False
    else:
        use_log = use_log_scale
    
    # Apply log scale if needed
    if use_log and all(v > 0 for v in all_values):
        ax.set_yscale('log')
        
        # Add annotation about log scale
        max_val = max(all_values)
        min_val = min(all_values)
        if max_val / min_val > 1000:
            note_text = ("Note: Log scale used due to large range of values.\n"
                         "Starting DD values can be thousands of times higher\n"
                         "than final optimized values.")
            ax.text(0.02, 0.05, note_text, transform=ax.transAxes, 
                    fontsize=8, alpha=0.7, bbox=dict(boxstyle="round,pad=0.5", 
                                                    fc="lightyellow", ec="orange", alpha=0.5))
    
    # Plot each run
    for i, history in enumerate(all_histories):
        if i == best_idx:
            ax.plot(history, 'r-', linewidth=3, label=f'Run {i+1} (Best)')
        else:
            ax.plot(history, 'b-', alpha=0.5, linewidth=1, label=f'Run {i+1}')
    
    # Add best value line if best_idx is provided
    if best_idx is not None:
        min_value = min(all_histories[best_idx])
        ax.axhline(y=min_value, color='r', linestyle='--', alpha=0.7)
        
        # Format the minimum value text based on magnitude
        if min_value > 1000:
            min_text = f'Min: {min_value:.2e}'
        else:
            min_text = f'Min: {min_value:.6f}'
        
        # Position the text appropriately based on scale
        if use_log:
            text_y_position = min_value * 1.5  # Above the line in log scale
        else:
            # For linear scale, position depends on the range
            max_run_value = max(all_histories[best_idx])
            value_range = max_run_value - min_value
            if value_range > 0:
                text_y_position = min_value + (value_range * 0.05)
            else:
                text_y_position = min_value * 1.1
        
        ax.text(len(all_histories[best_idx]) * 0.8, text_y_position, min_text, 
                color='r', fontsize=12)
    
    ax.set_xlabel('Iteration', fontsize=14)
    
    # Set appropriate y-axis label based on scale
    label = 'Dynamical Dependence (Positive)'
    if use_log:
        label += ' - Log Scale'
    
    ax.set_ylabel(label, fontsize=14)
    ax.set_title('Dynamical Dependence Optimization - All Runs', fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Only show legend if there are not too many runs
    if len(all_histories) <= 10:
        ax.legend(loc='upper right')
    
    plt.tight_layout()
    return fig


def plot_causal_graph(model, threshold=0.1, title="Causal Graph"):
    """
    Plot the causal graph of a model.
    
    Parameters
    ----------
    model : StateSpaceModel or VARModel
        The model to visualize
    threshold : float, optional
        Threshold for including edges in the graph
    title : str, optional
        Plot title
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    # Convert to VAR model if needed
    from py_ssdi.models.state_space import StateSpaceModel
    if isinstance(model, StateSpaceModel):
        from py_ssdi.models.var import VARModel
        var_model = model.to_var()
    else:
        var_model = model
    
    # Get model dimension
    n = var_model.n
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes
    for i in range(n):
        G.add_node(i, label=f'X{i+1}')
    
    # Compute Granger causality-like measure from VAR coefficients
    gc_matrix = np.zeros((n, n))
    for k in range(var_model.p):
        gc_matrix += np.abs(var_model.A[:, :, k])
    
    # Add edges with weights above threshold
    for i in range(n):
        for j in range(n):
            if i != j and gc_matrix[i, j] > threshold:
                G.add_edge(j, i, weight=gc_matrix[i, j])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Set node positions using spring layout
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue', 
                          alpha=0.8, ax=ax)
    
    # Draw edges with width proportional to weight
    edges = G.edges(data=True)
    weights = [d['weight'] * 2 for _, _, d in edges]
    nx.draw_networkx_edges(G, pos, width=weights, alpha=0.7, 
                          edge_color='gray', arrows=True, 
                          arrowsize=15, ax=ax)
    
    # Draw labels
    labels = {node: data['label'] for node, data in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=12, 
                           font_weight='bold', ax=ax)
    
    # Set title and remove axis
    ax.set_title(title, fontsize=14)
    ax.axis('off')
    
    plt.tight_layout()
    return fig


def plot_projection_comparison(model, projections, labels=None, title="Projection Comparison", 
                         use_positive_dd=True, use_log_scale='auto'):
    """
    Compare multiple projections in terms of dynamical dependence and causal emergence.
    
    Parameters
    ----------
    model : StateSpaceModel or VARModel
        The model to analyze (n-dimensional microscopic system)
    projections : list of ndarray
        List of projection matrices to compare, each L is an n×m matrix where m < n,
        representing a mapping from n-dimensional microscopic space to 
        m-dimensional macroscopic space
    labels : list of str, optional
        Labels for the projections
    title : str, optional
        Plot title
    use_positive_dd : bool, optional
        Whether to use the positive dynamical dependence measure (MATLAB-style)
        or the original measure that can be negative
    use_log_scale : str or bool, optional
        Whether to use a logarithmic scale for the dynamical dependence plot.
        'auto': Use log scale if the ratio of max/min values exceeds 100
        True: Always use log scale
        False: Never use log scale
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    from py_ssdi.metrics.dynamical_independence import (
        dynamical_dependence,
        causal_emergence,
    )
    
    # Import the positive dynamical dependence function from examples
    # This is a temporary solution until it's properly integrated into the metrics module
    from py_ssdi.examples.basic_example import dynamical_dependence_positive
    
    # Calculate metrics for each projection
    dd_values = []
    ce_values = []
    dd_dims = []
    
    for L in projections:
        # Calculate dynamical dependence using either positive or original method
        if use_positive_dd:
            dd = dynamical_dependence_positive(model, L)
        else:
            dd = dynamical_dependence(model, L)
            
        ce = causal_emergence(model, L)
        dd_values.append(dd)
        ce_values.append(ce)
        
        # Store the macroscopic dimension of each projection
        dd_dims.append(L.shape[1])
    
    # Create default labels if not provided
    if labels is None:
        labels = [f'Proj {i+1} (m={dd_dims[i]})' for i in range(len(projections))]
    else:
        # Append dimension information to existing labels
        labels = [f'{label} (m={dd_dims[i]})' for i, label in enumerate(labels)]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Determine if we should use log scale
    if use_log_scale == 'auto':
        # Compute max/min ratio, ignoring negative values
        positive_values = [v for v in dd_values if v > 0]
        if positive_values:
            max_val = max(positive_values)
            min_val = min(positive_values)
            use_log = (max_val / min_val > 100) if min_val > 0 else True
        else:
            use_log = False
    else:
        use_log = use_log_scale
    
    # Plot dynamical dependence
    x = np.arange(len(projections))
    bars = ax1.bar(x, dd_values, width=0.6, color='skyblue', alpha=0.8)
    
    # Apply log scale if needed
    if use_log and all(v > 0 for v in dd_values):
        ax1.set_yscale('log')
        # Add value labels above each bar
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height * 1.05,
                    f'{dd_values[i]:.2e}' if dd_values[i] > 1000 else f'{dd_values[i]:.2f}',
                    ha='center', va='bottom', rotation=0, fontsize=8)
        
        # Add explanatory note about log scale
        if max(dd_values) / min(dd_values) > 1000:
            note_text = ("Note: Log scale used because random projections often have\n"
                         "much higher DD values than optimized projections.\n"
                         "Lower values are better - optimization minimizes DD.")
            ax1.text(0.02, 0.05, note_text, transform=ax1.transAxes, 
                     fontsize=8, alpha=0.7, bbox=dict(boxstyle="round,pad=0.5", 
                                                     fc="lightyellow", ec="orange", alpha=0.5))
    else:
        # Add value labels for non-log scale
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height * 1.05,
                    f'{dd_values[i]:.2f}',
                    ha='center', va='bottom', rotation=0, fontsize=8)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.set_ylabel('Dynamical Dependence', fontsize=12)
    
    # Update title based on scale and type
    dd_type = "Positive" if use_positive_dd else "Original"
    scale_type = "Log Scale" if use_log else ""
    ax1.set_title(f'Dynamical Dependence Comparison ({dd_type}) {scale_type}', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Add n and m dimensions to the plot
    microscopic_dim = model.n
    ax1.text(0.02, 0.95, f'Microscopic dim (n): {microscopic_dim}', 
             transform=ax1.transAxes, fontsize=10, verticalalignment='top')
    
    # Plot causal emergence
    ax2.bar(x, ce_values, width=0.6, color='salmon', alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.set_ylabel('Causal Emergence', fontsize=12)
    ax2.set_title('Causal Emergence Comparison', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Add value labels for causal emergence
    for i, v in enumerate(ce_values):
        ax2.text(i, v + 0.5, f'{v:.2f}', ha='center', fontsize=8)
    
    # Set overall title
    fig.suptitle(title, fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig 