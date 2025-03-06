"""
Dynamical independence metrics for state-space and VAR models.
"""

import numpy as np
import scipy.linalg as la
from control import dare


def log_determinant(matrix):
    """
    Compute the logarithm of the determinant of a matrix.
    
    Parameters
    ----------
    matrix : ndarray
        Input matrix
        
    Returns
    -------
    float
        Log-determinant of the matrix
    """
    # Use Cholesky decomposition for numerical stability with positive definite matrices
    try:
        R = la.cholesky(matrix)
        return 2 * np.sum(np.log(np.diag(R)))
    except la.LinAlgError:
        # Fall back to standard determinant if Cholesky fails
        sign, logdet = np.linalg.slogdet(matrix)
        if sign <= 0:
            raise ValueError("Matrix determinant is not positive")
        return logdet


def dynamical_dependence(model, L):
    """
    Calculate dynamical dependence of projection L for a state-space or VAR model.
    
    Parameters
    ----------
    model : StateSpaceModel or VARModel
        The model to analyze
    L : ndarray
        Orthonormal subspace basis (n x m), where m < n
        
    Returns
    -------
    float
        Dynamical dependence value
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
    
    # Calculate residuals covariance matrix V of projected model (solve DARE)
    KKT = K @ K.T
    # Ensure KKT is symmetric
    KKT = (KKT + KKT.T) / 2
    
    LC = L.T @ C
    
    # Solve discrete algebraic Riccati equation
    try:
        # Make sure dimensions are compatible
        r = A.shape[0]  # state dimension
        m = L.shape[1]  # projection dimension
        
        # Check if LC has the right shape for DARE
        if LC.shape != (m, r):
            print(f"Incompatible dimensions: LC shape is {LC.shape}, expected ({m}, {r})")
            return np.nan
            
        # The control.dare function expects:
        # A: r x r, B: r x m, Q: r x r, R: m x m, S: r x m
        # We're passing A, LC.T, KKT, np.eye(m)
        # So B = LC.T should be r x m
        
        # Transpose LC to get the right dimensions for B
        # dare returns X, L, G (not X, L, G, S as we were trying to unpack)
        V, _, _ = dare(A, LC.T, KKT, np.eye(m))
    except Exception as e:
        print(f"DARE failed: {e}")
        return np.nan
    
    # D = log-determinant of residuals covariance matrix V
    try:
        D = log_determinant(V)
    except ValueError:
        return np.nan
    
    return D


def causal_emergence(model, L):
    """
    Calculate causal emergence of projection L for a state-space or VAR model.
    
    Parameters
    ----------
    model : StateSpaceModel or VARModel
        The model to analyze
    L : ndarray
        Orthonormal subspace basis (n x m), where m < n
        
    Returns
    -------
    float
        Causal emergence value (CI - DD)
    """
    # Convert to state-space model if needed
    from py_ssdi.models.var import VARModel
    if isinstance(model, VARModel):
        from py_ssdi.models.state_space import StateSpaceModel
        model = model.to_state_space()
    
    # Extract model parameters
    A, C, K, V = model.A, model.C, model.K, model.V
    
    # Ensure L is orthonormal
    L = np.asarray(L)
    q, r = np.linalg.qr(L)
    L = q
    
    # Get model dimensions
    n = C.shape[0]
    m = L.shape[1]
    r = A.shape[0]
    
    # Get precomputed values if available
    G = model.get_covariance()
    P = model.get_prediction_errors()
    
    # Prepare matrices
    VCHOL = la.cholesky(V, lower=True)
    KV = K @ VCHOL
    KVK = KV @ KV.T
    # Ensure KVK is symmetric
    KVK = (KVK + KVK.T) / 2
    
    LV = L.T @ VCHOL
    LVL = LV @ LV.T
    # Ensure LVL is symmetric
    LVL = (LVL + LVL.T) / 2
    
    # First term - sum of log determinants of per-element prediction errors
    I1 = 0
    for i in range(n):
        I1 += log_determinant(L.T @ (C @ P[:, :, i] @ C.T + V) @ L)
    
    # Second term - log determinant of residuals covariance matrix
    try:
        LC = L.T @ C
        # The control.dare function expects:
        # A: r x r, B: r x m, Q: r x r, R: m x m, S: r x m
        # dare returns X, L, G (not X, L, G, S as we were trying to unpack)
        VR, _, _ = dare(A, LC.T, KVK, LVL, KV @ LV.T)
    except Exception as e:
        print(f"DARE failed in causal_emergence: {e}")
        return np.nan
    
    # Third term - log determinant of projected covariance
    I3 = (n - 1) * log_determinant(L.T @ G @ L)
    
    # Fourth term - log determinant of projected residuals covariance
    I4 = log_determinant(LVL)
    
    # Co-information
    CI = I1 - I4 - I3
    
    # Dynamical Dependence
    DD = log_determinant(VR) - I4
    
    # Causal Emergence
    CE = CI - DD  # = I1 - I2 - I3
    
    return CE


def dynamical_independence_gradient(model, L):
    """
    Calculate gradient of dynamical dependence with respect to projection L.
    
    Parameters
    ----------
    model : StateSpaceModel or VARModel
        The model to analyze
    L : ndarray
        Orthonormal subspace basis (n x m), where m < n
        
    Returns
    -------
    ndarray
        Gradient of dynamical dependence
    float
        Magnitude of gradient
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
    
    # Get CAK sequence
    r = model.r
    n = model.n
    CAK = np.zeros((n, n, r))
    for k in range(r):
        CAK[:, :, k] = C @ np.linalg.matrix_power(A, k) @ K
    
    # Calculate gradient
    P = L @ L.T
    g = np.zeros((n, n))
    for k in range(r):
        Q = CAK[:, :, k]
        QT = Q.T
        g = g + Q @ QT - QT @ P @ Q - Q @ P @ QT
    
    # Project gradient onto tangent space of Stiefel manifold
    G = 2 * g @ L
    G = G - P @ G
    
    # Calculate magnitude of gradient
    mG = np.sqrt(np.sum(G**2))
    
    return G, mG


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
    
    if verbose:
        print(f"Best run: {best_idx+1}/{num_restarts}. Best DD: {best_dd:.6f}")
    
    return best_L, best_dd, all_histories


def spectral_dynamical_dependence(model, L):
    """
    Calculate dynamical dependence using the spectral method with transfer function H.
    
    Parameters
    ----------
    model : StateSpaceModel or VARModel
        The model to analyze
    L : ndarray
        Orthonormal subspace basis (n x m), where m < n
        
    Returns
    -------
    float
        Dynamical dependence value
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
    
    # Calculate transfer function H
    H = C @ np.linalg.inv(np.eye(A.shape[0]) - A) @ K
    
    # Calculate dynamical dependence using spectral method
    D = np.trace(L.T @ H @ H.T @ L)
    
    return D


def optimize_spectral_dynamical_dependence(model, m, method='gradient_descent', 
                                          max_iterations=1000, tolerance=1e-8, 
                                          step_size=0.1, num_restarts=10, 
                                          seed=None, verbose=False):
    """
    Optimize dynamical dependence using the spectral method to find the optimal projection.
    
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
            dd = spectral_dynamical_dependence(model, L)
            
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
            dd_new = spectral_dynamical_dependence(model, L_new)
            
            # Line search (simple backtracking)
            while dd_new > dd and current_step_size > tolerance:
                current_step_size *= 0.5
                L_new = orthonormalize(L - current_step_size * G)
                dd_new = spectral_dynamical_dependence(model, L_new)
            
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
        final_dd = spectral_dynamical_dependence(model, L)
        if final_dd < best_dd:
            best_L = L
            best_dd = final_dd
    
    if verbose:
        print(f"Best run: {best_idx+1}/{num_restarts}. Best DD: {best_dd:.6f}")
    
    return best_L, best_dd, all_histories 