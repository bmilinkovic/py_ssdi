"""
Basic example demonstrating the usage of the py_ssdi package.
"""

import numpy as np
import matplotlib.pyplot as plt

from py_ssdi.models import StateSpaceModel, VARModel
from py_ssdi.metrics import dynamical_dependence, causal_emergence
from py_ssdi.utils import optimize_dynamical_dependence, random_orthonormal
from py_ssdi.visualization import (
    plot_optimization_history,
    plot_causal_graph,
    plot_projection_comparison,
)


def main():
    """Run the basic example."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create a random state-space model
    n = 5  # observation dimension
    r = 15  # state dimension
    rho = 0.9  # spectral radius
    
    print(f"Creating a random state-space model with n={n}, r={r}, rho={rho}")
    model = StateSpaceModel.create_random(n, r, rho)
    
    # Create a random projection
    m = 2  # projection dimension
    L_random = random_orthonormal(n, m)
    
    # Calculate dynamical dependence and causal emergence for random projection
    dd_random = dynamical_dependence(model, L_random)
    ce_random = causal_emergence(model, L_random)
    
    print(f"Random projection:")
    print(f"  Dynamical Dependence: {dd_random:.6f}")
    print(f"  Causal Emergence: {ce_random:.6f}")
    
    # Optimize dynamical dependence
    print("\nOptimizing dynamical dependence...")
    L_optimal, dd_optimal, history = optimize_dynamical_dependence(
        model, m, max_iterations=100, num_restarts=3
    )
    
    # Calculate causal emergence for optimal projection
    ce_optimal = causal_emergence(model, L_optimal)
    
    print(f"Optimal projection:")
    print(f"  Dynamical Dependence: {dd_optimal:.6f}")
    print(f"  Causal Emergence: {ce_optimal:.6f}")
    
    # Plot optimization history
    fig1 = plot_optimization_history(history)
    
    # Plot causal graph
    fig2 = plot_causal_graph(model, threshold=0.1)
    
    # Compare projections
    projections = [L_random, L_optimal]
    labels = ["Random", "Optimal"]
    fig3 = plot_projection_comparison(model, projections, labels)
    
    # Show plots
    plt.show()
    
    # Create a VAR model example
    print("\nCreating a random VAR model...")
    var_model = VARModel.create_random(n, p=3, rho=0.9)
    
    # Calculate metrics for VAR model
    dd_var = dynamical_dependence(var_model, L_random)
    ce_var = causal_emergence(var_model, L_random)
    
    print(f"VAR model with random projection:")
    print(f"  Dynamical Dependence: {dd_var:.6f}")
    print(f"  Causal Emergence: {ce_var:.6f}")
    
    # Plot VAR model causal graph
    fig4 = plot_causal_graph(var_model, threshold=0.1, title="VAR Model Causal Graph")
    plt.show()


if __name__ == "__main__":
    main() 