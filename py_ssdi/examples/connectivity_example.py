"""
Connectivity patterns for state-space models example.

This script demonstrates the usage of the connectivity module to create
state-space models with different modular connectivity patterns.
"""

import numpy as np
import matplotlib.pyplot as plt
from py_ssdi.connectivity import (
    create_canonical_9node_model,
    create_canonical_16node_model,
    create_canonical_68node_model,
    visualize_connectivity
)
from py_ssdi.metrics import dynamical_dependence, causal_emergence
from py_ssdi.utils import optimize_dynamical_dependence, random_orthonormal


def dynamical_dependence_positive(model, L):
    """
    Calculate dynamical dependence using the Frobenius norm approach (MATLAB-style),
    which always returns positive values.
    
    Parameters
    ----------
    model : StateSpaceModel or VARModel
        The model to analyze
    L : ndarray
        Orthonormal subspace basis (n x m), where m < n
        
    Returns
    -------
    float
        Dynamical dependence value (always positive)
    """
    # Convert to state-space model if needed
    from py_ssdi.models.var import VARModel
    if isinstance(model, VARModel):
        from py_ssdi.models.state_space import StateSpaceModel
        model = model.to_state_space()
    
    # Ensure model has normalized residuals
    model = model.transform_to_normalized()
    
    # Extract model parameters
    A, C, K = model.A, model.C, model.K
    
    # Ensure L is orthonormal
    L = np.asarray(L)
    q, r = np.linalg.qr(L)
    L = q
    
    # Get CAK sequence (similar to MATLAB implementation)
    r = model.r
    n = model.n
    CAK = np.zeros((n, n, r))
    for k in range(r):
        CAK[:, :, k] = C @ np.linalg.matrix_power(A, k) @ K
    
    # Calculate dynamical dependence (MATLAB-style)
    D = 0
    for k in range(r):
        LCAKk = L.T @ CAK[:, :, k]
        LCAKLTk = LCAKk @ L
        D1k = LCAKk**2
        D2k = LCAKLTk**2
        D = D + np.sum(D1k) - np.sum(D2k)
    
    return D


def visualize_model_with_patterns(model, module_sizes, title):
    """Visualize a model's connectivity and its causal properties."""
    # Create a figure to plot connectivity
    fig = visualize_connectivity(model, module_sizes)
    fig.suptitle(title, fontsize=16)
    
    # For dynamical dependence calculation, create an appropriate projection
    n = model.n
    m = len(module_sizes)  # Use number of modules as projection dimension
    
    # Create a random projection for initial measurements
    L = random_orthonormal(n, m)
    
    # Calculate and display dynamical dependence using positive version
    dd = dynamical_dependence_positive(model, L)
    ce = causal_emergence(model, L)
    
    print(f"\n{title}:")
    print(f"  Number of variables: {model.n}")
    print(f"  State dimension: {model.r}")
    print(f"  Number of modules: {len(module_sizes)}")
    print(f"  Module sizes: {module_sizes}")
    print(f"  Dynamical dependence (random projection to {m} dimensions): {dd:.4f}")
    print(f"  Causal emergence (random projection to {m} dimensions): {ce:.4f}")
    
    return fig


def optimize_dynamical_dependence_positive(model, m, method='gradient', 
                                          max_iter=100, tol=1e-6, 
                                          n_runs=5, verbose=False):
    """
    Optimize dynamical dependence using positive measure to find the optimal projection.
    
    Parameters
    ----------
    model : StateSpaceModel or VARModel
        The model to analyze
    m : int
        Dimension of the projection (m < n)
    method : str, optional
        Optimization method ('gradient' or 'evolutionary')
    max_iter : int, optional
        Maximum number of iterations
    tol : float, optional
        Convergence tolerance
    n_runs : int, optional
        Number of random restarts
    verbose : bool, optional
        Whether to print progress
        
    Returns
    -------
    dict
        Dictionary containing optimization results
    """
    from py_ssdi.utils import random_orthonormal, orthonormalize
    
    # Get model dimension
    n = model.n
    
    # Initialize best solution
    best_L = None
    best_dd = float('inf')
    best_history = []
    best_run = -1
    all_histories = []
    
    # Run optimization with multiple restarts
    for run in range(n_runs):
        if verbose:
            print(f"Run {run+1}/{n_runs}...")
        
        # Initialize random projection
        L = random_orthonormal(n, m)
        
        # Initialize optimization history
        history = []
        
        # Initialize step size
        step_size = 0.1
        
        # Gradient descent optimization
        for iteration in range(max_iter):
            # Calculate current dynamical dependence
            dd = dynamical_dependence_positive(model, L)
            
            # Store in history
            history.append(dd)
            
            # Calculate gradient by finite differences (simple approach)
            G = np.zeros((n, m))
            eps = 1e-6
            
            for i in range(n):
                for j in range(m):
                    L_perturbed = L.copy()
                    L_perturbed[i, j] += eps
                    L_perturbed = orthonormalize(L_perturbed)
                    dd_perturbed = dynamical_dependence_positive(model, L_perturbed)
                    G[i, j] = (dd_perturbed - dd) / eps
            
            # Project gradient onto tangent space of Stiefel manifold
            G_proj = G - L @ (L.T @ G)
            
            # Check convergence
            if np.linalg.norm(G_proj) < tol:
                if verbose:
                    print(f"  Converged after {iteration} iterations. DD: {dd:.6f}")
                break
            
            # Update projection
            L_new = orthonormalize(L - step_size * G_proj)
            
            # Calculate new dynamical dependence
            dd_new = dynamical_dependence_positive(model, L_new)
            
            # Line search (simple backtracking)
            while dd_new > dd and step_size > tol:
                step_size *= 0.5
                L_new = orthonormalize(L - step_size * G_proj)
                dd_new = dynamical_dependence_positive(model, L_new)
            
            # Update projection
            if dd_new < dd:
                L = L_new
                # Increase step size if successful
                step_size *= 1.2
            else:
                # If no improvement, reduce step size
                step_size *= 0.5
            
            # Check if step size is too small
            if step_size < tol:
                if verbose:
                    print(f"  Step size too small after {iteration} iterations. DD: {dd:.6f}")
                break
        
        # Store history for this run
        all_histories.append(history)
        
        # Check if this run found a better solution
        final_dd = dynamical_dependence_positive(model, L)
        if final_dd < best_dd:
            best_L = L
            best_dd = final_dd
            best_history = history
            best_run = run
    
    if verbose:
        print(f"Best run: {best_run+1}/{n_runs}. Best DD: {best_dd:.6f}")
    
    return {
        'projection': best_L,
        'dynamical_dependence': best_dd,
        'history': best_history,
        'all_histories': all_histories,
        'best_run': best_run
    }


def optimize_across_scales(model, module_sizes, num_restarts=10, max_iter=100, verbose=True):
    """
    Run optimizations for multiple macroscopic scales from 2 up to n-1.
    
    Parameters
    ----------
    model : StateSpaceModel
        The model to analyze
    module_sizes : list
        List of module sizes in the model (for reference only)
    num_restarts : int, optional
        Number of random restarts per scale
    max_iter : int, optional
        Maximum iterations per optimization run
    verbose : bool, optional
        Whether to print progress
        
    Returns
    -------
    dict
        Dictionary containing optimization results for each scale
    """
    n = model.n
    results = {}
    
    # Run optimizations for each macroscopic scale from 2 up to n-1
    for m in range(2, n):
        if verbose:
            print(f"\nOptimizing for macroscopic scale m={m} (of microscopic n={n})...")
        
        # Run optimization with multiple restarts
        opt_result = optimize_dynamical_dependence_positive(
            model,
            m=m,
            n_runs=num_restarts,
            max_iter=max_iter,
            verbose=verbose
        )
        
        # Store results for this scale
        results[m] = opt_result
        
        # Print summary
        if verbose:
            best_dd = opt_result['dynamical_dependence']
            L_random = random_orthonormal(n, m)
            dd_random = dynamical_dependence_positive(model, L_random)
            print(f"  Random projection DD (m={m}): {dd_random:.6f}")
            print(f"  Optimized projection DD (m={m}): {best_dd:.6f}")
            print(f"  Improvement ratio: {dd_random/best_dd:.2f}x")
    
    return results


def plot_optimization_across_scales(model_name, results, figsize=(15, 10)):
    """
    Plot optimization runs for multiple macroscopic scales.
    
    Parameters
    ----------
    model_name : str
        Name of the model for the plot title
    results : dict
        Dictionary containing optimization results for each scale
    figsize : tuple, optional
        Figure size
        
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    # Get the number of scales
    scales = sorted(results.keys())
    n_scales = len(scales)
    
    # Calculate grid dimensions
    n_cols = min(3, n_scales)
    n_rows = (n_scales + n_cols - 1) // n_cols
    
    # Create figure and axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle(f"Optimization Runs for {model_name}", fontsize=16)
    
    # Flatten axes for easier indexing if there's only one row
    if n_rows == 1:
        axes = np.array([axes])
    
    # Flatten axes for easier indexing if there's only one column
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot each scale
    for i, m in enumerate(scales):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        # Get results for this scale
        scale_results = results[m]
        all_histories = scale_results['all_histories']
        best_run = scale_results['best_run']
        
        # Plot all runs
        for j, history in enumerate(all_histories):
            if j == best_run:
                ax.plot(history, 'r-', linewidth=2, alpha=0.8, label='Best run')
            else:
                ax.plot(history, 'b-', linewidth=1, alpha=0.3)
        
        # Set axis labels and title
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Dynamical Dependence')
        ax.set_title(f"Macroscopic Scale m={m}")
        ax.grid(True)
        
        # Use log scale if values span multiple orders of magnitude
        if len(all_histories[best_run]) > 0:
            max_val = max(all_histories[best_run])
            min_val = min(all_histories[best_run])
            if max_val / min_val > 100:
                ax.set_yscale('log')
        
        # Add legend to the first subplot only
        if i == 0:
            ax.legend()
    
    # Hide empty subplots
    for i in range(n_scales, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
    return fig


def main():
    """Demonstrate the use of canonical connectivity patterns."""
    print("Creating canonical state-space models with modular connectivity...\n")
    
    # Create the 9-node model
    print("Creating 9-node model with 3 modules...")
    model_9node = create_canonical_9node_model(rho=0.9, rmii=0.2)
    module_sizes_9 = [2, 3, 4]
    fig1 = visualize_model_with_patterns(
        model_9node, 
        module_sizes_9, 
        "9-Node Model (3 Modules)"
    )
    
    # Create the 16-node model
    print("\nCreating 16-node model with 4 modules...")
    model_16node = create_canonical_16node_model(rho=0.85, rmii=0.1)
    module_sizes_16 = [4, 4, 4, 4]
    fig2 = visualize_model_with_patterns(
        model_16node, 
        module_sizes_16, 
        "16-Node Model (4 Modules)"
    )
    
    # Create the 68-node model (smaller version for quick demonstration)
    print("\nCreating a smaller 20-node model with 4 modules for faster demonstration...")
    # For the demonstration, we'll use a smaller version with 4 modules
    small_module_sizes = [4, 5, 6, 5]
    inter_module_connections = [
        (0, 1), (0, 2),
        (1, 2), (1, 3),
        (2, 3), (2, 0),
        (3, 0)
    ]
    from py_ssdi.connectivity import create_modular_connectivity
    model_small = create_modular_connectivity(
        small_module_sizes,
        inter_module_connections,
        rho=0.8,
        intra_module_density=0.8,
        randomize=True,
        rmii=0.05
    )
    fig3 = visualize_model_with_patterns(
        model_small, 
        small_module_sizes, 
        "20-Node Model (4 Modules)"
    )
    
    # Run optimizations across scales for 9-node model
    print("\n" + "="*80)
    print("Running optimizations across scales for 9-node model...")
    print("="*80)
    results_9node = optimize_across_scales(
        model_9node,
        module_sizes_9,
        num_restarts=10,
        max_iter=100
    )
    
    # Plot optimization runs for 9-node model
    fig4 = plot_optimization_across_scales(
        "9-Node Model (3 Modules)",
        results_9node,
        figsize=(15, 10)
    )
    
    # Run optimizations across scales for 16-node model
    print("\n" + "="*80)
    print("Running optimizations across scales for 16-node model...")
    print("="*80)
    results_16node = optimize_across_scales(
        model_16node,
        module_sizes_16,
        num_restarts=10,
        max_iter=100
    )
    
    # Plot optimization runs for 16-node model
    fig5 = plot_optimization_across_scales(
        "16-Node Model (4 Modules)",
        results_16node,
        figsize=(15, 12)
    )
    
    # Show all figures
    plt.show()


if __name__ == "__main__":
    main() 