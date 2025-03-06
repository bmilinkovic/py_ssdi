"""
Basic example demonstrating the usage of the py_ssdi package.
"""

import numpy as np
import matplotlib.pyplot as plt

from py_ssdi.models import StateSpaceModel, VARModel
from py_ssdi.metrics import (
    dynamical_dependence, causal_emergence,
    preoptimisation_dynamical_dependence,
    optimise_preoptimisation_dynamical_dependence,
    spectral_dynamical_dependence,
    optimise_spectral_dynamical_dependence,
    calculate_cak_sequence
)
from py_ssdi.utils import random_orthonormal, orthonormalise
from py_ssdi.visualization import (
    plot_optimisation_history,
    plot_causal_graph,
    plot_projection_comparison,
)


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
    
    # Pre-calculate CAK sequence for efficiency
    print("Pre-calculating CAK sequence for optimisation...")
    CAK = calculate_cak_sequence(model)
    
    # Calculate dynamical dependence using both methods for comparison
    dd_random_original = dynamical_dependence(model, L_random)
    dd_random_preopt = preoptimisation_dynamical_dependence(model, L_random, CAK)
    ce_random = causal_emergence(model, L_random)
    
    print(f"Random projection from microscopic (n={n}) to macroscopic (m={m}) space:")
    print(f"  Dynamical Dependence (Original): {dd_random_original:.6f}")
    print(f"  Dynamical Dependence (Preoptimisation): {dd_random_preopt:.6f}")
    print(f"  Causal Emergence: {ce_random:.6f}")
    
    # Pre-optimisation step
    print(f"\nRunning preoptimisation step for macroscopic dimension m={m}...")
    L_optimal_pre, dd_optimal_pre, all_histories_pre, best_idx_pre = optimise_preoptimisation_dynamical_dependence(
        model, m, max_iterations=100, num_restarts=5, verbose=True
    )
    
    # Calculate causal emergence for pre-optimised projection
    ce_optimal_pre = causal_emergence(model, L_optimal_pre)
    
    print(f"Pre-optimised projection from microscopic (n={n}) to macroscopic (m={m}) space:")
    print(f"  Dynamical Dependence (Preoptimisation): {dd_optimal_pre:.6f}")
    print(f"  Causal Emergence: {ce_optimal_pre:.6f}")
    
    # Full optimisation step using spectral method
    print(f"\nRunning spectral optimisation for macroscopic dimension m={m}...")
    L_optimal_spectral, dd_optimal_spectral, all_histories_spectral, best_idx_spectral = optimise_spectral_dynamical_dependence(
        model, m, max_iterations=100, num_restarts=5, verbose=True
    )
    
    # Calculate causal emergence for fully optimised projection
    ce_optimal_spectral = causal_emergence(model, L_optimal_spectral)
    
    print(f"Spectrally optimised projection from microscopic (n={n}) to macroscopic (m={m}) space:")
    print(f"  Dynamical Dependence (Spectral): {dd_optimal_spectral:.6f}")
    print(f"  Causal Emergence: {ce_optimal_spectral:.6f}")
    
    # Plot the optimisation history for pre-optimisation
    fig1 = plot_optimisation_history(all_histories_pre, title="Preoptimisation History")
    
    # Plot the optimisation history for spectral optimisation
    fig2 = plot_optimisation_history(all_histories_spectral, title="Spectral Optimisation History")
    
    # Show plots
    plt.show()
    
    # Create a VAR model example
    print("\nCreating a random VAR model...")
    var_model = VARModel.create_random(n, p=3, rho=0.9)
    
    # Pre-calculate CAK sequence for VAR model
    print("Pre-calculating CAK sequence for VAR model...")
    var_CAK = calculate_cak_sequence(var_model)
    
    # Calculate metrics for VAR model
    dd_var_original = dynamical_dependence(var_model, L_random)
    dd_var_preopt = preoptimisation_dynamical_dependence(var_model, L_random, var_CAK)
    ce_var = causal_emergence(var_model, L_random)
    
    print(f"VAR model with random projection from microscopic (n={n}) to macroscopic (m={m}) space:")
    print(f"  Dynamical Dependence (Original): {dd_var_original:.6f}")
    print(f"  Dynamical Dependence (Preoptimisation): {dd_var_preopt:.6f}")
    print(f"  Causal Emergence: {ce_var:.6f}")
    
    # Plot VAR model causal graph
    fig6 = plot_causal_graph(var_model, threshold=0.1, title="VAR Model Causal Graph")
    plt.show()


if __name__ == "__main__":
    main() 