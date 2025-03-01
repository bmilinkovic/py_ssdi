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