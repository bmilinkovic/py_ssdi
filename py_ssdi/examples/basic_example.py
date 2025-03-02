"""
Basic example demonstrating the usage of the py_ssdi package.
"""

import numpy as np
import matplotlib.pyplot as plt

from py_ssdi.models import StateSpaceModel, VARModel
from py_ssdi.metrics import dynamical_dependence, causal_emergence, dynamical_independence_gradient
from py_ssdi.utils import optimize_dynamical_dependence, random_orthonormal, orthonormalize
from py_ssdi.visualization import (
    plot_optimization_history,
    plot_causal_graph,
    plot_projection_comparison,
)


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


def optimize_dynamical_dependence_positive(model, m, method='gradient_descent', 
                                          max_iterations=1000, tolerance=1e-8, 
                                          step_size=0.1, num_restarts=10, 
                                          seed=None, verbose=False):
    """
    Optimize dynamical dependence to find the optimal projection using positive DD measures.
    
    Parameters
    ----------
    model : StateSpaceModel or VARModel
        The model to analyze
    m : int
        Dimension of the projection (m < n)
    method : str, optional
        Optimization method ('gradient_descent' or 'evolutionary')
    max_iterations : int, optional
        Maximum number of iterations
    tolerance : float, optional
        Convergence tolerance
    step_size : float, optional
        Initial step size for gradient descent
    num_restarts : int, optional
        Number of random restarts
    seed : int, optional
        Random seed
    verbose : bool, optional
        Whether to print progress
        
    Returns
    -------
    ndarray
        Optimal projection matrix
    float
        Optimal dynamical dependence value
    list
        Optimization history for all runs
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Get model dimension
    n = model.n
    
    # Initialize best solution
    best_L = None
    best_dd = float('inf')
    
    # Store histories for all runs
    all_histories = []
    
    # Run optimization with multiple restarts
    for run in range(num_restarts):
        if verbose:
            print(f"Run {run+1}/{num_restarts}")
            
        # Initialize random projection
        L = random_orthonormal(n, m)
        
        # Initialize optimization history
        history = []
        
        # Initialize step size
        current_step_size = step_size
        
        # Gradient descent optimization
        for iteration in range(max_iterations):
            # Calculate current dynamical dependence
            dd = dynamical_dependence_positive(model, L)
            
            # Store in history
            history.append(dd)
            
            # Calculate gradient
            G, mG = dynamical_independence_gradient(model, L)
            
            # Check convergence
            if mG < tolerance:
                if verbose:
                    print(f"  Converged after {iteration} iterations. DD: {dd:.6f}")
                break
            
            # Update projection using gradient descent on Stiefel manifold
            L_new = orthonormalize(L - current_step_size * G)
            
            # Calculate new dynamical dependence
            dd_new = dynamical_dependence_positive(model, L_new)
            
            # Line search (simple backtracking)
            while dd_new > dd and current_step_size > tolerance:
                current_step_size *= 0.5
                L_new = orthonormalize(L - current_step_size * G)
                dd_new = dynamical_dependence_positive(model, L_new)
            
            # Update projection
            if dd_new < dd:
                L = L_new
                # Increase step size if successful
                current_step_size *= 1.2
            else:
                # If no improvement, reduce step size
                current_step_size *= 0.5
            
            # Check if step size is too small
            if current_step_size < tolerance:
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
            best_idx = run
    
    if verbose:
        print(f"Best run: {best_idx+1}/{num_restarts}. Best DD: {best_dd:.6f}")
    
    return best_L, best_dd, all_histories, best_idx


def main():
    """Run the basic example."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create a random state-space model (microscopic system)
    n = 5  # microscopic dimension (observation dimension)
    r = 15  # latent state dimension
    rho = 0.9  # spectral radius
    
    print(f"Creating a random state-space model with microscopic dimension n={n}, r={r}, rho={rho}")
    model = StateSpaceModel.create_random(n, r, rho)
    
    # Create a random projection to macroscopic space
    m = 2  # macroscopic dimension (projection dimension), where m < n
    print(f"Selecting macroscopic dimension m={m} (coarse-graining the n={n} dimensional system)")
    L_random = random_orthonormal(n, m)
    
    # Calculate dynamical dependence using both methods for comparison
    dd_random_original = dynamical_dependence(model, L_random)
    dd_random_positive = dynamical_dependence_positive(model, L_random)
    ce_random = causal_emergence(model, L_random)
    
    print(f"Random projection from microscopic (n={n}) to macroscopic (m={m}) space:")
    print(f"  Dynamical Dependence (Original): {dd_random_original:.6f}")
    print(f"  Dynamical Dependence (Positive): {dd_random_positive:.6f}")
    print(f"  Causal Emergence: {ce_random:.6f}")
    
    # Optimize dynamical dependence using positive measure
    print(f"\nOptimizing dynamical dependence for macroscopic dimension m={m}...")
    L_optimal, dd_optimal, all_histories, best_idx = optimize_dynamical_dependence_positive(
        model, m, max_iterations=100, num_restarts=5, verbose=True
    )
    
    # Calculate causal emergence for optimal projection
    ce_optimal = causal_emergence(model, L_optimal)
    
    print(f"Optimal projection from microscopic (n={n}) to macroscopic (m={m}) space:")
    print(f"  Dynamical Dependence (Positive): {dd_optimal:.6f}")
    print(f"  Causal Emergence: {ce_optimal:.6f}")
    
    # Print the min value from history for debugging
    min_history_value = min(all_histories[best_idx])
    print(f"  Min value in optimization history: {min_history_value:.6f}")
    print(f"  Initial value to final value ratio: {max(all_histories[best_idx])/min_history_value:.2f}x")
    
    # Calculate maximum range across all runs for consistent scaling
    from py_ssdi.visualization.plotting import plot_optimization_runs
    
    # Plot all optimization runs with log scale
    fig1 = plot_optimization_runs(all_histories, best_idx, use_log_scale='auto')
    fig1.suptitle(f"Optimization Runs (n={n} → m={m})", fontsize=16)
    
    # Plot just the best run using log scale for consistency
    from py_ssdi.visualization.plotting import plot_optimization_history
    fig2 = plot_optimization_history(all_histories[best_idx], 
                                    title=f"Best Dynamical Dependence Optimization Run (n={n} → m={m})",
                                    positive_dd=True,
                                    use_log_scale='auto')
    
    # Plot causal graph
    fig3 = plot_causal_graph(model, threshold=0.1)
    
    # Calculate DD values for comparison plots for debugging
    dd_random_for_plot = dynamical_dependence_positive(model, L_random)
    dd_optimal_for_plot = dynamical_dependence_positive(model, L_optimal)
    print("\nDD values for projection comparison plot:")
    print(f"  Random projection: {dd_random_for_plot:.6f}")
    print(f"  Optimal projection: {dd_optimal_for_plot:.6f}")
    print(f"  Ratio (Random/Optimal): {dd_random_for_plot/dd_optimal_for_plot:.2f}x")
    print("  Note: Random projections often have very high DD values compared to optimized projections.")
    print("        Using log scale for visualization to handle the large range of values.")
    
    # Compare projections
    projections = [L_random, L_optimal]
    labels = ["Random", "Optimal"]
    fig4 = plot_projection_comparison(model, projections, labels, 
                                     title=f"Projection Comparison (n={n} → m={m})",
                                     use_positive_dd=True,
                                     use_log_scale='auto')
    
    # Try a different macroscopic dimension
    m2 = 3  # alternative macroscopic dimension
    print(f"\nTrying alternative macroscopic dimension m={m2}...")
    L_random2 = random_orthonormal(n, m2)
    
    # Optimize for the new dimension
    L_optimal2, dd_optimal2, all_histories2, best_idx2 = optimize_dynamical_dependence_positive(
        model, m2, max_iterations=100, num_restarts=5, verbose=True
    )
    
    print(f"  Optimal DD for m={m2}: {dd_optimal2:.6f}")
    
    # Calculate DD values for multi-dimension comparison for debugging
    dd_vals = [
        dynamical_dependence_positive(model, L_random),
        dynamical_dependence_positive(model, L_optimal),
        dynamical_dependence_positive(model, L_random2),
        dynamical_dependence_positive(model, L_optimal2)
    ]
    print("\nDD values for dimension comparison plot:")
    print(f"  Random m={m}: {dd_vals[0]:.6f}")
    print(f"  Optimal m={m}: {dd_vals[1]:.6f}")
    print(f"  Random m={m2}: {dd_vals[2]:.6f}")
    print(f"  Optimal m={m2}: {dd_vals[3]:.6f}")
    print(f"  Max/Min ratio: {max(dd_vals)/min(dd_vals):.2f}x")
    
    # Compare different macroscopic dimensions
    all_projections = [L_random, L_optimal, L_random2, L_optimal2]
    all_labels = [f"Random m={m}", f"Optimal m={m}", f"Random m={m2}", f"Optimal m={m2}"]
    fig5 = plot_projection_comparison(model, all_projections, all_labels,
                                     title=f"Comparison of Different Macroscopic Dimensions",
                                     use_positive_dd=True,
                                     use_log_scale='auto')
    
    # Show plots
    plt.show()
    
    # Create a VAR model example
    print("\nCreating a random VAR model...")
    var_model = VARModel.create_random(n, p=3, rho=0.9)
    
    # Calculate metrics for VAR model
    dd_var_original = dynamical_dependence(var_model, L_random)
    dd_var_positive = dynamical_dependence_positive(var_model, L_random)
    ce_var = causal_emergence(var_model, L_random)
    
    print(f"VAR model with random projection from microscopic (n={n}) to macroscopic (m={m}) space:")
    print(f"  Dynamical Dependence (Original): {dd_var_original:.6f}")
    print(f"  Dynamical Dependence (Positive): {dd_var_positive:.6f}")
    print(f"  Causal Emergence: {ce_var:.6f}")
    
    # Plot VAR model causal graph
    fig6 = plot_causal_graph(var_model, threshold=0.1, title="VAR Model Causal Graph")
    plt.show()


if __name__ == "__main__":
    main() 