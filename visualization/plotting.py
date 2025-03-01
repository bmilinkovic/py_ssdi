"""
Visualization tools for plotting results.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def plot_optimization_history(history, title="Dynamical Dependence Optimization"):
    """
    Plot the optimization history of dynamical dependence.
    
    Parameters
    ----------
    history : list or ndarray
        History of dynamical dependence values during optimization
    title : str, optional
        Plot title
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history, 'b-', linewidth=2)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Dynamical Dependence', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add horizontal line at minimum value
    min_value = min(history)
    ax.axhline(y=min_value, color='r', linestyle='--', alpha=0.7)
    ax.text(len(history) * 0.8, min_value * 1.1, f'Min: {min_value:.6f}', 
            color='r', fontsize=10)
    
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


def plot_projection_comparison(model, projections, labels=None, title="Projection Comparison"):
    """
    Compare multiple projections in terms of dynamical dependence and causal emergence.
    
    Parameters
    ----------
    model : StateSpaceModel or VARModel
        The model to analyze
    projections : list of ndarray
        List of projection matrices to compare
    labels : list of str, optional
        Labels for the projections
    title : str, optional
        Plot title
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    from py_ssdi.metrics.dynamical_independence import (
        dynamical_dependence,
        causal_emergence,
    )
    
    # Calculate metrics for each projection
    dd_values = []
    ce_values = []
    
    for L in projections:
        dd = dynamical_dependence(model, L)
        ce = causal_emergence(model, L)
        dd_values.append(dd)
        ce_values.append(ce)
    
    # Create default labels if not provided
    if labels is None:
        labels = [f'Proj {i+1}' for i in range(len(projections))]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot dynamical dependence
    x = np.arange(len(projections))
    ax1.bar(x, dd_values, width=0.6, color='skyblue', alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.set_ylabel('Dynamical Dependence', fontsize=12)
    ax1.set_title('Dynamical Dependence Comparison', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Plot causal emergence
    ax2.bar(x, ce_values, width=0.6, color='salmon', alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right')
    ax2.set_ylabel('Causal Emergence', fontsize=12)
    ax2.set_title('Causal Emergence Comparison', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Set overall title
    fig.suptitle(title, fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig 