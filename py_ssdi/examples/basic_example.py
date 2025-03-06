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
    
    # Pre-optimization step using positive measure
    print(f"\nPre-optimizing dynamical dependence for macroscopic dimension m={m}...")
    L_optimal_pre, dd_optimal_pre, all_histories_pre = optimize_dynamical_dependence_positive(
        model, m, max_iterations=100, num_restarts=5, verbose=True
    )
    
    # Calculate causal emergence for pre-optimized projection
    ce_optimal_pre = causal_emergence(model, L_optimal_pre)
    
    print(f"Pre-optimized projection from microscopic (n={n}) to macroscopic (m={m}) space:")
    print(f"  Dynamical Dependence (Positive): {dd_optimal_pre:.6f}")
    print(f"  Causal Emergence: {ce_optimal_pre:.6f}")
    
    # Full optimization step using spectral method
    print(f"\nOptimizing dynamical dependence using spectral method for macroscopic dimension m={m}...")
    L_optimal_spectral, dd_optimal_spectral, all_histories_spectral = optimize_spectral_dynamical_dependence(
        model, m, max_iterations=100, num_restarts=5, verbose=True
    )
    
    # Calculate causal emergence for fully optimized projection
    ce_optimal_spectral = causal_emergence(model, L_optimal_spectral)
    
    print(f"Spectral optimized projection from microscopic (n={n}) to macroscopic (m={m}) space:")
    print(f"  Dynamical Dependence (Spectral): {dd_optimal_spectral:.6f}")
    print(f"  Causal Emergence: {ce_optimal_spectral:.6f}")
    
    # Plot the optimization history for pre-optimization
    fig1 = plot_optimization_history(all_histories_pre, title="Pre-Optimization History")
    
    # Plot the optimization history for spectral optimization
    fig2 = plot_optimization_history(all_histories_spectral, title="Spectral Optimization History")
    
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