"""
Optimization utilities for finding optimal projections.
"""

import numpy as np
import scipy.linalg as la


def orthonormalize(L):
    """
    Orthonormalize a matrix using QR decomposition.
    
    Parameters
    ----------
    L : ndarray
        Matrix to orthonormalize
        
    Returns
    -------
    ndarray
        Orthonormalized matrix
    """
    q, r = np.linalg.qr(L)
    return q


def random_orthonormal(n, m, seed=None):
    """
    Generate a random orthonormal matrix.
    
    Parameters
    ----------
    n : int
        Number of rows
    m : int
        Number of columns (m <= n)
    seed : int, optional
        Random seed
        
    Returns
    -------
    ndarray
        Random orthonormal matrix of shape (n, m)
    """
    if seed is not None:
        np.random.seed(seed)
    
    L = np.random.randn(n, m)
    return orthonormalize(L)


def optimize_dynamical_dependence(model, m, method='gradient_descent', 
                                 max_iterations=1000, tolerance=1e-8, 
                                 step_size=0.1, num_restarts=10, seed=None):
    """
    Optimize dynamical dependence to find the optimal projection.
    
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
        
    Returns
    -------
    ndarray
        Optimal projection matrix
    float
        Optimal dynamical dependence value
    list
        Optimization history
    """
    from py_ssdi.metrics.dynamical_independence import (
        dynamical_dependence,
        dynamical_independence_gradient,
    )
    
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)
    
    # Get model dimension
    n = model.n
    
    # Initialize best solution
    best_L = None
    best_dd = float('inf')
    best_history = None
    
    # Run optimization with multiple restarts
    for run in range(num_restarts):
        # Initialize random projection
        L = random_orthonormal(n, m)
        
        # Initialize optimization history
        history = []
        
        # Initialize step size
        current_step_size = step_size
        
        # Gradient descent optimization
        for iteration in range(max_iterations):
            # Calculate current dynamical dependence
            dd = dynamical_dependence(model, L)
            
            # Store in history
            history.append(dd)
            
            # Calculate gradient
            G, mG = dynamical_independence_gradient(model, L)
            
            # Check convergence
            if mG < tolerance:
                break
            
            # Update projection using gradient descent on Stiefel manifold
            L_new = L - current_step_size * G
            L_new = orthonormalize(L_new)
            
            # Calculate new dynamical dependence
            dd_new = dynamical_dependence(model, L_new)
            
            # Line search (simple backtracking)
            while dd_new > dd and current_step_size > tolerance:
                current_step_size *= 0.5
                L_new = L - current_step_size * G
                L_new = orthonormalize(L_new)
                dd_new = dynamical_dependence(model, L_new)
            
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
                break
        
        # Check if this run found a better solution
        final_dd = dynamical_dependence(model, L)
        if final_dd < best_dd:
            best_L = L
            best_dd = final_dd
            best_history = history
    
    return best_L, best_dd, best_history


def cluster_projections(projections, tolerance=1e-6):
    """
    Cluster similar projections based on their subspace angles.
    
    Parameters
    ----------
    projections : list of ndarray
        List of projection matrices
    tolerance : float, optional
        Clustering tolerance
        
    Returns
    -------
    list of list
        Clusters of projection indices
    """
    n = len(projections)
    clusters = []
    used = [False] * n
    
    for i in range(n):
        if used[i]:
            continue
        
        cluster = [i]
        used[i] = True
        
        for j in range(i+1, n):
            if used[j]:
                continue
            
            # Calculate principal angles between subspaces
            L_i = projections[i]
            L_j = projections[j]
            
            # Use SVD to compute principal angles
            u, s, vh = np.linalg.svd(L_i.T @ L_j)
            
            # Check if subspaces are close
            if np.min(s) > 1 - tolerance:
                cluster.append(j)
                used[j] = True
        
        clusters.append(cluster)
    
    return clusters 