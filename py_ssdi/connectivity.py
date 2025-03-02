"""
Connectivity patterns for state-space models.

This module provides functions to create state-space models with different
connectivity patterns, focusing on modular structures with varying degrees
of interconnectivity.
"""

import numpy as np
import scipy.linalg as la
from .models.state_space import StateSpaceModel

def create_modular_connectivity(module_sizes, inter_module_connections=None, rho=0.9, 
                                intra_module_density=1.0, randomize=False, rmii=1):
    """
    Create a state-space model with modular connectivity structure.
    
    Parameters
    ----------
    module_sizes : list of int
        List of sizes for each module
    inter_module_connections : list of tuple, optional
        List of (from_module, to_module) tuples specifying which modules should connect.
        If None, adjacent modules are connected in a chain.
    rho : float, optional
        Spectral radius for stability (< 1), defaults to 0.9
    intra_module_density : float, optional
        Density of connections within modules (0-1), defaults to 1.0 (all-to-all)
    randomize : bool, optional
        If True, add random noise to connection strengths, defaults to False
    rmii : float, optional
        Residuals multiinformation (0 for uncorrelated residuals), defaults to 1
        
    Returns
    -------
    StateSpaceModel
        A state-space model with the specified modular structure
    """
    # Calculate total dimensions
    n = sum(module_sizes)  # Observation dimension
    r = 2 * n              # State dimension (2 times for stable dynamics)
    
    # Initialize matrices
    A = np.zeros((r, r))
    C = np.zeros((n, r))
    K = np.zeros((r, n))
    
    # Track cumulative sizes for indexing
    obs_idx = 0
    state_idx = 0
    module_indices = []
    
    # Create mapping from module index to observation indices
    for i, size in enumerate(module_sizes):
        idx_range = (obs_idx, obs_idx + size)
        module_indices.append(idx_range)
        obs_idx += size
    
    # Create block structure in A matrix (state transition)
    for i, (start, end) in enumerate(module_indices):
        # Double the state space for each module for stable dynamics
        s_start = 2 * start
        s_end = 2 * end
        
        # Create random intra-module connectivity with controlled density
        block_size = 2 * (end - start)
        if intra_module_density >= 1.0:
            # All-to-all connectivity
            block = np.random.randn(block_size, block_size)
        else:
            # Sparse connectivity with given density
            block = np.zeros((block_size, block_size))
            n_connections = int(intra_module_density * block_size * block_size)
            indices = np.random.choice(
                block_size * block_size, 
                size=n_connections, 
                replace=False
            )
            rows, cols = np.unravel_index(indices, (block_size, block_size))
            for row, col in zip(rows, cols):
                block[row, col] = np.random.randn()
        
        # Scale to contribute to desired spectral radius
        A[s_start:s_end, s_start:s_end] = block
    
    # Add inter-module connections
    if inter_module_connections is None:
        # Default: chain connection between adjacent modules
        inter_module_connections = [(i, i+1) for i in range(len(module_sizes)-1)]
    
    for from_mod, to_mod in inter_module_connections:
        from_start, from_end = module_indices[from_mod]
        to_start, to_end = module_indices[to_mod]
        
        # Use a single connection from each source to each target module
        s_from_start = 2 * from_start
        s_from_end = 2 * from_end
        s_to_start = 2 * to_start
        s_to_end = 2 * to_end
        
        # Add a connection from a random node in source to a random node in target
        src_idx = s_from_start + np.random.randint(0, s_from_end - s_from_start)
        tgt_idx = s_to_start + np.random.randint(0, s_to_end - s_to_start)
        A[tgt_idx, src_idx] = np.random.randn()
        
        # Add reverse connection for symmetry if desired
        # A[src_idx, tgt_idx] = np.random.randn()
    
    # Scale A to desired spectral radius
    A_scaled = rho * A / np.max(np.abs(la.eigvals(A)))
    
    # Set up observation matrix C to observe the first half of state variables
    for i in range(n):
        C[i, 2*i] = 1.0  # Each observation sees its corresponding state
    
    # Set up Kalman gain matrix K with reasonable values
    for i in range(n):
        K[2*i, i] = 0.5
        K[2*i+1, i] = 0.2
    
    # Add noise if randomization is requested
    if randomize:
        A_scaled += 0.1 * rho * np.random.randn(*A_scaled.shape) 
        C += 0.1 * np.random.randn(*C.shape)
        K += 0.1 * np.random.randn(*K.shape)
    
    # Create residuals covariance matrix with given multiinformation
    if rmii == 0:
        V = np.eye(n)
    else:
        # Generate a random correlation matrix
        V_raw = np.random.randn(n, n)
        V_raw = V_raw @ V_raw.T
        # Normalize to get correlation matrix (diagonal elements = 1)
        D = np.diag(np.sqrt(np.diag(V_raw)))
        V = np.linalg.inv(D) @ V_raw @ np.linalg.inv(D)
        # Scale to control multiinformation
        if rmii != 1:
            V = (V - np.eye(n)) * rmii + np.eye(n)
    
    return StateSpaceModel(A_scaled, C, K, V)

def create_canonical_9node_model(rho=0.9, rmii=1, randomize=False):
    """
    Create a canonical 9-node model with 3 modules of sizes 2, 3, and 4.
    
    Parameters
    ----------
    rho : float, optional
        Spectral radius for stability (< 1), defaults to 0.9
    rmii : float, optional
        Residuals multiinformation (0 for uncorrelated residuals), defaults to 1
    randomize : bool, optional
        If True, add random noise to connection strengths, defaults to False
        
    Returns
    -------
    StateSpaceModel
        A 9-node state-space model with 3 modules
    """
    module_sizes = [2, 3, 4]
    # Connect module 0 to 1, and 1 to 2
    inter_module_connections = [(0, 1), (1, 2)]
    
    return create_modular_connectivity(
        module_sizes=module_sizes,
        inter_module_connections=inter_module_connections,
        rho=rho,
        intra_module_density=1.0,  # All-to-all within modules
        randomize=randomize,
        rmii=rmii
    )

def create_canonical_16node_model(rho=0.9, rmii=1, randomize=False):
    """
    Create a canonical 16-node model with 4 modules of size 4 each.
    
    Parameters
    ----------
    rho : float, optional
        Spectral radius for stability (< 1), defaults to 0.9
    rmii : float, optional
        Residuals multiinformation (0 for uncorrelated residuals), defaults to 1
    randomize : bool, optional
        If True, add random noise to connection strengths, defaults to False
        
    Returns
    -------
    StateSpaceModel
        A 16-node state-space model with 4 modules
    """
    module_sizes = [4, 4, 4, 4]
    # Create a ring structure: 0->1->2->3->0
    inter_module_connections = [(0, 1), (1, 2), (2, 3), (3, 0)]
    
    return create_modular_connectivity(
        module_sizes=module_sizes,
        inter_module_connections=inter_module_connections,
        rho=rho,
        intra_module_density=1.0,  # All-to-all within modules
        randomize=randomize,
        rmii=rmii
    )

def create_canonical_68node_model(rho=0.9, rmii=1, randomize=False):
    """
    Create a canonical 68-node model with anatomically-inspired modules.
    
    This model has a structure loosely inspired by brain regions:
    - 7 modules representing different cortical areas
    - Varying sizes to represent different anatomical regions
    - Hierarchical connectivity pattern
    
    Parameters
    ----------
    rho : float, optional
        Spectral radius for stability (< 1), defaults to 0.9
    rmii : float, optional
        Residuals multiinformation (0 for uncorrelated residuals), defaults to 1
    randomize : bool, optional
        If True, add random noise to connection strengths, defaults to False
        
    Returns
    -------
    StateSpaceModel
        A 68-node state-space model with anatomically-inspired modules
    """
    # Define modules inspired by brain regions (68 nodes total)
    module_sizes = [8, 10, 12, 14, 8, 8, 8]
    
    # Create a hierarchical connectivity pattern
    # Primary -> Secondary -> Association areas with additional connections
    inter_module_connections = [
        (0, 1), (0, 2),  # Module 0 connects to 1 and 2
        (1, 2), (1, 3),  # Module 1 connects to 2 and 3
        (2, 3), (2, 4),  # Module 2 connects to 3 and 4
        (3, 4), (3, 5),  # Module 3 connects to 4 and 5
        (4, 5), (4, 6),  # Module 4 connects to 5 and 6
        (5, 6), (5, 0),  # Module 5 connects to 6 and 0
        (6, 0), (6, 1)   # Module 6 connects to 0 and 1
    ]
    
    return create_modular_connectivity(
        module_sizes=module_sizes,
        inter_module_connections=inter_module_connections,
        rho=rho,
        intra_module_density=0.8,  # Slightly less dense within modules
        randomize=randomize,
        rmii=rmii
    )

def visualize_connectivity(model, module_sizes=None, figsize=(10, 8)):
    """
    Visualize the connectivity structure of a state-space model.
    
    Parameters
    ----------
    model : StateSpaceModel
        The state-space model to visualize
    module_sizes : list of int, optional
        List of sizes for each module, used to draw module boundaries
    figsize : tuple, optional
        Figure size (width, height) in inches, defaults to (10, 8)
        
    Returns
    -------
    matplotlib.figure.Figure
        The generated figure object
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    
    # Plot A matrix (state transition)
    ax = axs[0, 0]
    im = ax.imshow(model.A, cmap='coolwarm', vmin=-0.5, vmax=0.5)
    ax.set_title('A Matrix (State Transition)')
    
    # Plot C matrix (observation)
    ax = axs[0, 1]
    im = ax.imshow(model.C, cmap='coolwarm', vmin=-0.5, vmax=0.5)
    ax.set_title('C Matrix (Observation)')
    
    # Plot K matrix (Kalman gain)
    ax = axs[1, 0]
    im = ax.imshow(model.K, cmap='coolwarm', vmin=-0.5, vmax=0.5)
    ax.set_title('K Matrix (Kalman Gain)')
    
    # Plot V matrix (residuals covariance)
    ax = axs[1, 1]
    im = ax.imshow(model.V, cmap='coolwarm', vmin=0, vmax=1)
    ax.set_title('V Matrix (Residuals Covariance)')
    
    # Add module boundaries if module_sizes is provided
    if module_sizes is not None:
        n = model.n
        r = model.r
        
        # Cumulative sizes for observation dimension
        obs_cumsum = np.cumsum([0] + module_sizes)
        
        # For state dimension (2x the observation dimension in our setup)
        state_cumsum = np.cumsum([0] + [2*s for s in module_sizes])
        
        # Add rectangles to A matrix plot to show modules
        ax = axs[0, 0]
        for i in range(len(module_sizes)):
            for j in range(len(module_sizes)):
                s_i_start, s_i_end = state_cumsum[i], state_cumsum[i+1]
                s_j_start, s_j_end = state_cumsum[j], state_cumsum[j+1]
                
                if i == j:  # Highlight diagonal modules
                    edgecolor = 'green'
                    linewidth = 2
                else:
                    edgecolor = 'black'
                    linewidth = 1
                
                rect = Rectangle(
                    (s_j_start - 0.5, s_i_start - 0.5),  # Rectangle position
                    s_j_end - s_j_start,                  # Width
                    s_i_end - s_i_start,                  # Height
                    edgecolor=edgecolor,
                    facecolor='none',
                    linewidth=linewidth
                )
                ax.add_patch(rect)
        
        # Add rectangles to C matrix plot
        ax = axs[0, 1]
        for i in range(len(module_sizes)):
            o_i_start, o_i_end = obs_cumsum[i], obs_cumsum[i+1]
            for j in range(len(module_sizes)):
                s_j_start, s_j_end = state_cumsum[j], state_cumsum[j+1]
                
                if i == j:  # Highlight diagonal modules
                    edgecolor = 'green'
                    linewidth = 2
                else:
                    edgecolor = 'black'
                    linewidth = 1
                
                rect = Rectangle(
                    (s_j_start - 0.5, o_i_start - 0.5),  # Rectangle position
                    s_j_end - s_j_start,                  # Width
                    o_i_end - o_i_start,                  # Height
                    edgecolor=edgecolor,
                    facecolor='none',
                    linewidth=linewidth
                )
                ax.add_patch(rect)
        
        # Add rectangles to K matrix plot
        ax = axs[1, 0]
        for i in range(len(module_sizes)):
            s_i_start, s_i_end = state_cumsum[i], state_cumsum[i+1]
            for j in range(len(module_sizes)):
                o_j_start, o_j_end = obs_cumsum[j], obs_cumsum[j+1]
                
                if i == j:  # Highlight diagonal modules
                    edgecolor = 'green'
                    linewidth = 2
                else:
                    edgecolor = 'black'
                    linewidth = 1
                
                rect = Rectangle(
                    (o_j_start - 0.5, s_i_start - 0.5),  # Rectangle position
                    o_j_end - o_j_start,                  # Width
                    s_i_end - s_i_start,                  # Height
                    edgecolor=edgecolor,
                    facecolor='none',
                    linewidth=linewidth
                )
                ax.add_patch(rect)
        
        # Add rectangles to V matrix plot
        ax = axs[1, 1]
        for i in range(len(module_sizes)):
            o_i_start, o_i_end = obs_cumsum[i], obs_cumsum[i+1]
            for j in range(len(module_sizes)):
                o_j_start, o_j_end = obs_cumsum[j], obs_cumsum[j+1]
                
                if i == j:  # Highlight diagonal modules
                    edgecolor = 'green'
                    linewidth = 2
                else:
                    edgecolor = 'black'
                    linewidth = 1
                
                rect = Rectangle(
                    (o_j_start - 0.5, o_i_start - 0.5),  # Rectangle position
                    o_j_end - o_j_start,                  # Width
                    o_i_end - o_i_start,                  # Height
                    edgecolor=edgecolor,
                    facecolor='none',
                    linewidth=linewidth
                )
                ax.add_patch(rect)
    
    plt.tight_layout()
    fig.colorbar(im, ax=axs.ravel().tolist())
    
    return fig 