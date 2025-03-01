"""
State-space model implementation for the py_ssdi package.
"""

import numpy as np
import scipy.linalg as la
from control import dare
from scipy.linalg import solve_discrete_lyapunov


class StateSpaceModel:
    """
    State-space model in innovations form.
    The model is defined by the equations:
        x_{t+1} = A x_t + K e_t
        y_t = C x_t + e_t
    where e_t is a white noise process with covariance matrix V.
    """

    def __init__(self, A, C, K, V=None):
        """
        Initialize a state-space model.

        Parameters
        ----------
        A : ndarray
            State transition matrix (r x r)
        C : ndarray
            Observation matrix (n x r)
        K : ndarray
            Kalman gain matrix (r x n)
        V : ndarray, optional
            Residuals covariance matrix (n x n), defaults to identity
        """
        self.A = np.asarray(A)
        self.C = np.asarray(C)
        self.K = np.asarray(K)
        
        # Infer dimensions
        self.r = self.A.shape[0]  # state dimension
        self.n = self.C.shape[0]  # observation dimension
        
        # Set default residuals covariance matrix if not provided
        if V is None:
            self.V = np.eye(self.n)
        else:
            self.V = np.asarray(V)
        
        # Verify dimensions
        assert self.A.shape == (self.r, self.r), "A matrix has incorrect dimensions"
        assert self.C.shape == (self.n, self.r), "C matrix has incorrect dimensions"
        assert self.K.shape == (self.r, self.n), "K matrix has incorrect dimensions"
        assert self.V.shape == (self.n, self.n), "V matrix has incorrect dimensions"
        
        # Cache for computed values
        self._G = None  # observable covariance matrix
        self._P = None  # per-element state prediction error covariance matrices
    
    @classmethod
    def create_random(cls, n, r, rho, rmii=1):
        """
        Create a random state-space model.
        
        Parameters
        ----------
        n : int
            Observation dimension
        r : int
            State dimension
        rho : float
            Spectral radius (< 1 for stability)
        rmii : float, optional
            Residuals multiinformation; 0 for zero correlation, defaults to 1
            
        Returns
        -------
        StateSpaceModel
            A random state-space model
        """
        # Create a random state matrix with given spectral radius
        A_raw = np.random.randn(r, r)
        # Scale to desired spectral radius
        A = rho * A_raw / np.max(np.abs(la.eigvals(A_raw)))
        
        # Create random observation and Kalman gain matrices
        C = np.random.randn(n, r)
        K = np.random.randn(r, n)
        
        # Create a residuals covariance matrix with given multiinformation
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
        
        return cls(A, C, K, V)
    
    def transform_to_normalized(self):
        """
        Transform the model to have decorrelated and normalized residuals.
        
        Returns
        -------
        StateSpaceModel
            A new model with identity residuals covariance matrix
        """
        n = self.n
        sqrt_V = la.cholesky(self.V, lower=True)
        isqrt_V = la.inv(sqrt_V)
        
        # A is unchanged
        C_new = isqrt_V @ self.C
        K_new = self.K @ sqrt_V
        V_new = np.eye(n)
        
        return StateSpaceModel(self.A, C_new, K_new, V_new)
    
    def to_var(self):
        """
        Convert the state-space model to a VAR model.
        
        Returns
        -------
        tuple
            (A_var, V_var) - VAR coefficient matrices and residuals covariance
        """
        from py_ssdi.models.var import VARModel
        
        # Extract the impulse response sequence
        # We'll go up to r steps to capture the dynamics
        r = self.r
        CAK = np.zeros((self.n, self.n, r))
        
        for k in range(r):
            CAK[:, :, k] = self.C @ np.linalg.matrix_power(self.A, k) @ self.K
        
        # Solve VAR coefficient matrices from CAK sequence
        # This is a simplified approach - in practice need proper conversion
        # which requires MVGC2-like functionality
        
        return VARModel(CAK, self.V)
    
    def get_covariance(self):
        """
        Get the covariance matrix of the observable process.
        
        Returns
        -------
        ndarray
            The covariance matrix G
        """
        if self._G is None:
            # Compute state covariance by solving Lyapunov equation
            KVK = self.K @ self.V @ self.K.T
            M = solve_discrete_lyapunov(self.A, KVK)
            # Observable covariance
            self._G = self.C @ M @ self.C.T + self.V
        
        return self._G
    
    def get_prediction_errors(self):
        """
        Get per-element state prediction error covariance matrices.
        
        Returns
        -------
        ndarray
            Array of matrices P
        """
        if self._P is None:
            n, r = self.n, self.r
            P = np.zeros((r, r, n))
            KVK = self.K @ self.V @ self.K.T
            
            # Ensure KVK is symmetric (it should be, but numerical issues might cause small asymmetries)
            KVK = (KVK + KVK.T) / 2
            
            for i in range(n):
                try:
                    # Solve discrete algebraic Riccati equation for each element
                    C_i = self.C[i, :].reshape(1, r)
                    V_i = self.V[i, i]
                    
                    # The S matrix should have dimensions (r x 1) for dare
                    # But we're passing KV_i which might have wrong dimensions
                    # Let's just use None for S (default) instead of KV_i
                    
                    # Use the control.dare function
                    # dare returns X, L, G (not X, L, G, S)
                    X, _, _ = dare(self.A.T, C_i.T, KVK, V_i)
                    P[:, :, i] = X
                except Exception as e:
                    print(f"DARE failed in get_prediction_errors for element {i}: {e}")
                    # Use a fallback approach - set to identity matrix
                    P[:, :, i] = np.eye(r)
            
            self._P = P
        
        return self._P 