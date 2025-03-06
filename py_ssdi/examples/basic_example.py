"""
Basic example demonstrating the usage of the py_ssdi package.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

from py_ssdi.models import StateSpaceModel
from py_ssdi.metrics import (
    dynamical_dependence, causal_emergence,
    preoptimisation_dynamical_dependence,
    optimise_preoptimisation_dynamical_dependence,
    spectral_dynamical_dependence,
    optimise_spectral_dynamical_dependence,
    calculate_cak_sequence,
    random_orthonormal,
    orthonormalise
)
from py_ssdi.visualization import (
    plot_optimization_history,
    plot_optimization_runs,
    plot_causal_graph,
    plot_projection_comparison,
)
from py_ssdi.connectivity import (
    create_canonical_9node_model,
    visualize_connectivity
)

# Add Wes Anderson color palette at the top of the file after imports
WES_ANDERSON_COLORS = [
    '#E16A86',  # Grand Budapest Hotel pink
    '#00A0B0',  # Royal Tenenbaums blue
    '#CC3311',  # Fantastic Mr. Fox orange
    '#009B72',  # Isle of Dogs green
    '#D4AF37',  # Moonrise Kingdom gold
    '#8B4513',  # Darjeeling Limited brown
    '#FF6B6B',  # Life Aquatic red
    '#4A90E2',  # Rushmore blue
    '#F5A623',  # Bottle Rocket yellow
    '#7ED321'   # Asteroid City green
]

def calculate_transfer_function(model, fres=100):
    """Calculate transfer function H from state-space parameters.
    
    Args:
        model: StateSpaceModel instance
        fres: Number of frequency points (default: 100)
        
    Returns:
        H: Transfer function array of shape (n, n, fres+1)
    """
    n = model.n
    r = model.r
    A = model.A
    C = model.C
    K = model.K
    
    # Create identity matrices
    In = np.eye(n)
    Ir = np.eye(r)
    
    # Initialize transfer function array
    h = fres + 1
    H = np.zeros((n, n, h), dtype=complex)
    
    # Calculate frequency points
    w = np.exp(1j * np.pi * np.arange(h) / fres)
    
    # Calculate transfer function for each frequency point
    for k in range(h):
        # H(:,:,k) = In + C*((w(k)*Ir-A)\K)
        # Using scipy.linalg.solve for matrix division
        H[:, :, k] = In + C @ linalg.solve(w[k] * Ir - A, K)
    
    return H

def cluster_hyperplanes(L_matrices, dd_values, tol=1e-4):
    """Cluster hyperplanes based on their distance and dynamical dependence.
    
    Args:
        L_matrices: Array of L matrices
        dd_values: Array of dynamical dependence values
        tol: Distance tolerance for clustering
        
    Returns:
        unique_indices: Indices of unique clusters
        cluster_sizes: Size of each cluster
        cluster_dd: Best DD value for each cluster
    """
    # Convert L_matrices to numpy array if it's a list
    if isinstance(L_matrices, list):
        L_matrices = np.array(L_matrices)
    
    n_runs = len(L_matrices)
    available = np.ones(n_runs, dtype=bool)
    unique_indices = []
    cluster_sizes = []
    cluster_dd = []
    
    # Calculate pairwise distances between L matrices
    dist = np.zeros((n_runs, n_runs))
    for i in range(n_runs):
        for j in range(i+1, n_runs):
            # Frobenius norm of difference
            dist[i,j] = np.linalg.norm(L_matrices[i] - L_matrices[j])
            dist[j,i] = dist[i,j]
    
    # Sort by dynamical dependence (lower is better)
    sorted_idx = np.argsort(dd_values)
    
    k = 0
    for i in sorted_idx:
        if available[i]:
            # New cluster
            unique_indices.append(i)
            cluster_sizes.append(1)
            cluster_dd.append(dd_values[i])
            available[i] = False
            
            # Find similar matrices
            for j in range(n_runs):
                if available[j] and dist[i,j] < tol:
                    available[j] = False
                    cluster_sizes[k] += 1
            
            k += 1
    
    return np.array(unique_indices), np.array(cluster_sizes), np.array(cluster_dd)

def spectral_optimization(H, L_init, max_iter=10000, tol=1e-6, step_size=1e-3):
    """Optimize spectral dynamical dependence using gradient descent."""
    L = L_init.copy()
    m, n = L.shape
    
    # Initialize history
    hist = []
    dhist = []
    
    # Initial evaluation
    dd, Qk, g = spectral_dynamical_dependence(H, L, return_gradient=True)
    sig = np.linalg.norm(g)
    hist.append(L.copy())
    dhist.append([dd, sig, g.copy()])
    
    # Gradient descent
    for i in range(max_iter):
        # Update L
        L_new = L - step_size * g
        
        # Project onto Stiefel manifold using Cholesky decomposition
        # Add small regularization term to ensure positive definiteness
        eps = 1e-10
        try:
            M = L_new @ L_new.T + eps * np.eye(m)  # m x m matrix
            R = np.linalg.cholesky(M)
            L = L_new.T @ np.linalg.inv(R)  # Transpose L_new to match dimensions
        except np.linalg.LinAlgError:
            # Fallback to QR decomposition if Cholesky fails
            Q, R = np.linalg.qr(L_new.T)
            L = Q.T
        
        # Evaluate new point
        dd_new, Qk, g = spectral_dynamical_dependence(H, L, return_gradient=True)
        sig = np.linalg.norm(g)
        
        # Update history
        hist.append(L.copy())
        dhist.append([dd_new, sig, g.copy()])
        
        # Check convergence
        if sig < tol or abs(dd_new - dd) < tol:
            break
        
        dd = dd_new
    
    return dd, L, Qk, sig, i+1, (hist, dhist)

def spectral_dynamical_dependence(H, L, return_gradient=False):
    """Calculate spectral dynamical dependence from transfer function.
    
    Args:
        H: Transfer function array of shape (n, n, h)
        L: Orthonormal projection matrix
        return_gradient: Whether to return gradient information
        
    Returns:
        dd: Dynamical dependence value
        Qk: Q matrices for gradient calculation (if return_gradient=True)
        g: Gradient matrix (if return_gradient=True)
    """
    h = H.shape[2]
    d = np.zeros(h)
    
    # Small regularization term to ensure positive definiteness
    eps = 1e-10
    
    for k in range(h):
        # Qk = H(:,:,k)'*L
        Qk = H[:, :, k].T @ L
        # Calculate residuals covariance matrix (should be identity)
        M = Qk.T @ Qk + eps * np.eye(Qk.shape[1])
        # Calculate log-determinant using Cholesky decomposition
        try:
            R = np.linalg.cholesky(M)
            # Log-determinant should be non-negative as it represents reduction in prediction error
            d[k] = np.sum(np.log(np.abs(np.diag(R))))
        except np.linalg.LinAlgError:
            # If Cholesky fails, try to recover with eigendecomposition
            eigvals = np.linalg.eigvals(M)
            d[k] = np.sum(np.log(np.abs(eigvals)))
    
    # Integrate using trapezoidal rule
    dd = np.sum(d[:-1] + d[1:]) / (h-1)
    
    if return_gradient:
        # Calculate gradient using eigendecomposition
        g = np.zeros_like(L, dtype=complex)
        for k in range(h):
            Qk = H[:, :, k].T @ L
            M = Qk.T @ Qk + eps * np.eye(Qk.shape[1])
            try:
                # Use eigendecomposition for inverse
                eigvals, eigvecs = np.linalg.eigh(M)
                Minv = eigvecs @ np.diag(1/eigvals) @ eigvecs.T
                g += 2 * H[:, :, k] @ Qk @ Minv
            except np.linalg.LinAlgError:
                # If eigendecomposition fails, use pseudo-inverse
                g += 2 * H[:, :, k] @ Qk @ np.linalg.pinv(M)
        
        # Take real part of gradient and ensure it's float64
        g = np.real(g).astype(np.float64)
        return dd, Qk, g
    
    return dd

def analyse_model(model, m, title_prefix="", num_runs=100, max_iterations=10000, H=None):
    """Helper function to analyse a model with both preoptimisation and spectral methods."""
    n = model.n
    print(f"\n{'='*50}")
    print(f"Starting analysis for projection dimension m={m}")
    print(f"{'='*50}")
    
    # Create a random initial projection
    print("\nGenerating random initial projection...")
    L_random = random_orthonormal(n, m)
    
    # Pre-calculate CAK sequence for efficiency
    print(f"\n{title_prefix} - Pre-calculating CAK sequence for optimisation...")
    CAK = calculate_cak_sequence(model)
    print("CAK sequence calculation completed")
    
    # Calculate dynamical dependence using both methods for comparison
    print("\nCalculating initial dynamical dependence values...")
    dd_random_original = dynamical_dependence(model, L_random)
    dd_random_preopt = preoptimisation_dynamical_dependence(model, L_random, CAK)
    try:
        ce_random = causal_emergence(model, L_random)
    except ValueError as e:
        print(f"Warning: Could not calculate initial causal emergence: {e}")
        ce_random = np.nan
    
    print(f"\n{title_prefix} - Random projection from microscopic (n={n}) to macroscopic (m={m}) space:")
    print(f"  Dynamical Dependence (Original): {dd_random_original:.6f}")
    print(f"  Dynamical Dependence (Preoptimisation): {dd_random_preopt:.6f}")
    print(f"  Causal Emergence: {ce_random:.6f}")
    
    # Pre-optimisation step
    print(f"\n{title_prefix} - Starting preoptimisation step for macroscopic dimension m={m}...")
    print(f"  Number of runs: {num_runs}")
    print(f"  Maximum iterations per run: {max_iterations}")
    L_optimal_pre, dd_optimal_pre, all_histories_pre, best_idx_pre = optimise_preoptimisation_dynamical_dependence(
        model, m, max_iterations=max_iterations, num_restarts=num_runs, verbose=True
    )
    print("Preoptimisation step completed")
    
    # Calculate causal emergence for pre-optimised projection
    print("\nCalculating causal emergence for pre-optimised projection...")
    try:
        ce_optimal_pre = causal_emergence(model, L_optimal_pre)
    except ValueError as e:
        print(f"Warning: Could not calculate pre-optimized causal emergence: {e}")
        ce_optimal_pre = np.nan
    
    print(f"\n{title_prefix} - Pre-optimised projection from microscopic (n={n}) to macroscopic (m={m}) space:")
    print(f"  Dynamical Dependence (Preoptimisation): {dd_optimal_pre:.6f}")
    print(f"  Causal Emergence: {ce_optimal_pre:.6f}")
    
    # Use provided transfer function or calculate if not provided
    if H is None:
        print(f"\n{title_prefix} - Calculating transfer function for spectral optimization...")
        H = calculate_transfer_function(model)
        print("Transfer function calculation completed")
    
    # Cluster pre-optimized projections
    print("\nClustering pre-optimized projections...")
    # Extract final L matrices and their DD values from histories
    L_matrices = []
    dd_values = []
    
    # The history structure is a list of lists, where each inner list contains the history for one run
    for run_idx, history in enumerate(all_histories_pre):
        if isinstance(history, tuple) and len(history) == 2:
            hist, dhist = history
            if len(hist) > 0 and isinstance(hist[-1], np.ndarray):
                L = hist[-1]
                if L.shape == (m, n):  # Ensure correct shape
                    # Get the DD value from dhist
                    if len(dhist) > 0 and len(dhist[-1]) > 0:
                        dd = dhist[-1][0]  # First element is the DD value
                        L_matrices.append(L)
                        dd_values.append(dd)
    
    if not L_matrices:
        print("Warning: No valid L matrices found in histories. Using optimal pre-optimized projection.")
        L_matrices = [L_optimal_pre]
        dd_values = [dd_optimal_pre]
    
    L_matrices = np.array(L_matrices)
    dd_values = np.array(dd_values)
    
    # Cluster the matrices
    unique_indices, cluster_sizes, cluster_dd = cluster_hyperplanes(
        L_matrices, dd_values
    )
    print(f"Found {len(unique_indices)} unique clusters")
    
    # Use clustered projections as starting points for spectral optimization
    print(f"\n{title_prefix} - Starting spectral optimisation using clustered projections...")
    print(f"  Number of clusters: {len(unique_indices)}")
    print(f"  Maximum iterations per run: {max_iterations}")
    
    # Initialize arrays for results
    dopt = np.zeros(len(unique_indices))
    Lopt = np.zeros((len(unique_indices), m, n))  # Changed shape to match L matrices
    conv = np.zeros(len(unique_indices))
    iopt = np.zeros(len(unique_indices))
    sopt = np.zeros(len(unique_indices))
    all_histories_spectral = []
    
    # Run optimization for each cluster
    for k, idx in enumerate(unique_indices):
        print(f"\nOptimizing cluster {k+1}/{len(unique_indices)}")
        L0 = L_matrices[idx]  # Use the best L matrix from this cluster
        dd, L, Qk, sig, iters, (hist, dhist) = spectral_optimization(
            H, L0, max_iter=max_iterations, tol=1e-6, step_size=1e-3
        )
        
        dopt[k] = dd
        Lopt[k] = L  # Store L directly without transposing
        conv[k] = iters > 0
        iopt[k] = iters
        sopt[k] = sig
        all_histories_spectral.append((hist, dhist))
        
        print(f"Cluster {k+1}: DD = {dd:.6f}, sig = {sig:.6f}, ", end="")
        if iters > 0:
            print(f"converged({iters})", end="")
        else:
            print("unconverged", end="")
        print(f" in {iters} iterations")
    
    # Find best result
    best_idx = np.argmin(dopt)
    L_optimal_spectral = Lopt[best_idx]  # Get the best L matrix
    dd_optimal_spectral = dopt[best_idx]
    
    # Calculate causal emergence for spectrally optimised projection
    print("\nCalculating causal emergence for spectrally optimised projection...")
    try:
        ce_optimal_spectral = causal_emergence(model, L_optimal_spectral)
    except ValueError as e:
        print(f"Warning: Could not calculate spectrally optimized causal emergence: {e}")
        ce_optimal_spectral = np.nan
    
    print(f"\n{title_prefix} - Spectrally optimised projection from microscopic (n={n}) to macroscopic (m={m}) space:")
    print(f"  Dynamical Dependence (Spectral): {dd_optimal_spectral:.6f}")
    print(f"  Causal Emergence: {ce_optimal_spectral:.6f}")
    
    print(f"\n{'='*50}")
    print(f"Analysis completed for projection dimension m={m}")
    print(f"{'='*50}\n")
    
    return (L_random, L_optimal_pre, L_optimal_spectral, 
            dd_random_original, dd_optimal_pre, dd_optimal_spectral,
            ce_random, ce_optimal_pre, ce_optimal_spectral,
            all_histories_pre, all_histories_spectral)

def create_comprehensive_plots(results, n):
    """Create comprehensive plots showing optimization progress and results."""
    # Extract data for plotting
    m_values = []
    dd_preopt = []
    dd_spec = []
    
    for i, result in enumerate(results):
        m = i + 2  # Since we start from m=2
        m_values.append(m)
        dd_preopt.append(result[4])  # dd_optimal_pre is at index 4
        dd_spec.append(result[5])    # dd_optimal_spectral is at index 5
    
    # Convert lists to numpy arrays
    m_values = np.array(m_values)
    dd_preopt = np.array(dd_preopt)
    dd_spec = np.array(dd_spec)
    
    # Create figure for optimization results
    fig = plt.figure(figsize=(12, 6))
    
    # Plot preoptimization results
    plt.subplot(1, 2, 1)
    plt.plot(m_values, dd_preopt, color=WES_ANDERSON_COLORS[0], label='Pre-optimization')
    plt.xlabel('Macroscopic Dimension (m)')
    plt.ylabel('Dynamical Dependence')
    plt.title('Pre-optimization Results')
    plt.grid(True)
    plt.legend()
    
    # Plot spectral optimization results
    plt.subplot(1, 2, 2)
    plt.plot(m_values, dd_spec, color=WES_ANDERSON_COLORS[1], label='Spectral')
    plt.xlabel('Macroscopic Dimension (m)')
    plt.ylabel('Dynamical Dependence')
    plt.title('Spectral Optimization Results')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    return fig


def create_optimization_history_plots(results, n):
    """Create plots showing optimization history for each macroscopic dimension."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot pre-optimization history
    for i, result in enumerate(results):
        m = i + 2  # Projection dimension starts at 2
        histories = result[9]  # Pre-optimization history
        if len(histories) > 0:  # Only plot if history exists
            # Plot each run's history
            for run_idx, history in enumerate(histories):
                if len(history) > 0:  # Only plot if this run has history
                    # Use alpha to make overlapping lines visible
                    alpha = 0.1 if run_idx > 0 else 1.0  # First run more visible
                    ax1.plot(history, color=WES_ANDERSON_COLORS[i % len(WES_ANDERSON_COLORS)], 
                            alpha=alpha, label=f'm={m}' if run_idx == 0 else None)
    
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Dynamical Dependence')
    ax1.set_title('Pre-optimization History')
    ax1.grid(True)
    ax1.legend()
    
    # Plot spectral optimization history
    for i, result in enumerate(results):
        m = i + 2  # Projection dimension starts at 2
        histories = result[10]  # Spectral optimization history
        if len(histories) > 0:  # Only plot if history exists
            # Plot each run's history
            for run_idx, history in enumerate(histories):
                if len(history) > 0:  # Only plot if this run has history
                    # Use alpha to make overlapping lines visible
                    alpha = 0.1 if run_idx > 0 else 1.0  # First run more visible
                    ax2.plot(history, color=WES_ANDERSON_COLORS[i % len(WES_ANDERSON_COLORS)], 
                            alpha=alpha, label=f'm={m}' if run_idx == 0 else None)
    
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Dynamical Dependence')
    ax2.set_title('Spectral Optimization History')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    return fig


def main():
    """Main function to run the example."""
    # Create and analyze modular model
    n = 9  # Number of nodes
    modular_model = create_canonical_9node_model(rho=0.9, rmii=0.2)
    
    # Calculate transfer function once for all dimensions
    print("\nCalculating transfer function for spectral optimization...")
    H = calculate_transfer_function(modular_model)
    print("Transfer function calculation completed")
    
    # Analyze model for different projection dimensions
    results = []
    for m in range(2, n-1):  # Try different projection dimensions from 2 to n-2
        print(f"\nAnalysing modular model with projection dimension m={m}")
        result = analyse_model(modular_model, m, H=H)  # Pass H to analyse_model
        results.append(result)
    
    # Create and display plots
    fig1 = create_comprehensive_plots(results, n)
    fig2 = create_optimization_history_plots(results, n)
    
    plt.figure(fig1.number)
    plt.show()
    plt.figure(fig2.number)
    plt.show()


if __name__ == "__main__":
    main()