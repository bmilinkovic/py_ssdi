"""
Vector Autoregressive (VAR) model implementation for the py_ssdi package.
"""

import numpy as np
import scipy.linalg as la


class VARModel:
    """
    Vector Autoregressive (VAR) model.
    The model is defined by the equation:
        y_t = A_1 y_{t-1} + A_2 y_{t-2} + ... + A_p y_{t-p} + e_t
    where e_t is a white noise process with covariance matrix V.
    """

    def __init__(self, A, V=None):
        """
        Initialize a VAR model.

        Parameters
        ----------
        A : ndarray
            VAR coefficient matrices (n x n x p), where p is the VAR order
        V : ndarray, optional
            Residuals covariance matrix (n x n), defaults to identity
        """
        self.A = np.asarray(A)
        
        # Infer dimensions
        self.n = self.A.shape[0]  # variable dimension
        self.p = self.A.shape[2]  # VAR order
        
        # Set default residuals covariance matrix if not provided
        if V is None:
            self.V = np.eye(self.n)
        else:
            self.V = np.asarray(V)
        
        # Verify dimensions
        assert self.A.shape == (self.n, self.n, self.p), "A matrices have incorrect dimensions"
        assert self.V.shape == (self.n, self.n), "V matrix has incorrect dimensions"
    
    @classmethod
    def create_random(cls, n, p, rho, w=1, connectivity=None, rmii=1):
        """
        Create a random VAR model.
        
        Parameters
        ----------
        n : int
            Variable dimension
        p : int
            VAR order
        rho : float
            Spectral radius (< 1 for stability)
        w : float, optional
            VAR coefficients decay parameter, defaults to 1
        connectivity : ndarray, optional
            Connectivity matrix (n x n), defaults to fully connected
        rmii : float, optional
            Residuals multiinformation; 0 for zero correlation, defaults to 1
            
        Returns
        -------
        VARModel
            A random VAR model
        """
        # Default connectivity is fully connected
        if connectivity is None:
            connectivity = np.ones((n, n))
            
        # Generate VAR coefficient matrices with decay
        A = np.zeros((n, n, p))
        for k in range(p):
            A_k = np.random.randn(n, n) * connectivity
            # Apply decay to higher order coefficients
            A_k = A_k * np.exp(-w * k)
            A[:, :, k] = A_k
        
        # Ensure stability by scaling the coefficients
        # First form the companion matrix
        companion = np.zeros((n * p, n * p))
        companion[:n, :] = A.reshape(n, n * p)
        companion[n:, :-n] = np.eye(n * (p - 1))
        
        # Get the maximum eigenvalue magnitude
        max_eig = np.max(np.abs(la.eigvals(companion)))
        
        # Scale to ensure stability with the desired spectral radius
        if max_eig > 0:
            # Use a safety factor to ensure we're below the target spectral radius
            safety_factor = 0.95
            scaling = safety_factor * rho / max_eig
            A = A * scaling
        else:
            # If max eigenvalue is 0, scale conservatively
            A = A * (rho * 0.5)
        
        # Verify stability after scaling
        companion = np.zeros((n * p, n * p))
        companion[:n, :] = A.reshape(n, n * p)
        companion[n:, :-n] = np.eye(n * (p - 1))
        max_eig = np.max(np.abs(la.eigvals(companion)))
        
        # If still not stable, force stability with a more aggressive scaling
        if max_eig >= 1.0:
            scaling = 0.9 / max_eig  # More aggressive scaling to ensure stability
            A = A * scaling
            
            # Final verification
            companion = np.zeros((n * p, n * p))
            companion[:n, :] = A.reshape(n, n * p)
            companion[n:, :-n] = np.eye(n * (p - 1))
            max_eig = np.max(np.abs(la.eigvals(companion)))
            
            # If still unstable (very unlikely at this point), use a very conservative scaling
            if max_eig >= 1.0:
                A = A * 0.5  # Halve all coefficients as a last resort
        
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
                
        return cls(A, V)
    
    def transform_to_normalized(self):
        """
        Transform the model to have decorrelated and normalized residuals.
        
        Returns
        -------
        VARModel
            A new model with identity residuals covariance matrix
        """
        n = self.n
        p = self.p
        
        sqrt_V = la.cholesky(self.V, lower=True)
        isqrt_V = la.inv(sqrt_V)
        
        # Transform VAR coefficients
        A_new = np.zeros_like(self.A)
        for k in range(p):
            A_new[:, :, k] = isqrt_V @ self.A[:, :, k] @ sqrt_V
        
        return VARModel(A_new, np.eye(n))
    
    def to_state_space(self):
        """
        Convert the VAR model to a state-space model.
        
        Returns
        -------
        StateSpaceModel
            An equivalent state-space model
        """
        from py_ssdi.models.state_space import StateSpaceModel
        
        n = self.n
        p = self.p
        
        # Create state transition matrix A
        A = np.zeros((n * p, n * p))
        A[:n, :] = self.A.reshape(n, n * p)
        A[n:, :-n] = np.eye(n * (p - 1))
        
        # Create observation matrix C
        C = np.zeros((n, n * p))
        C[:, :n] = np.eye(n)
        
        # Create Kalman gain matrix K
        K = np.zeros((n * p, n))
        K[:n, :] = np.eye(n)
        
        return StateSpaceModel(A, C, K, self.V)
    
    def var_to_cak(self):
        """
        Convert VAR coefficients to CAK sequence.
        
        Returns
        -------
        ndarray
            CAK sequence (n x n x p)
        """
        # For VAR models, the CAK sequence is simply the coefficients
        return self.A.copy()
    
    def compute_autocovariance(self, lags=None):
        """
        Compute autocovariance sequence for the VAR model.
        
        Parameters
        ----------
        lags : int, optional
            Number of lags to compute, defaults to VAR order
            
        Returns
        -------
        ndarray
            Autocovariance sequence (n x n x (lags+1))
        """
        if lags is None:
            lags = self.p
            
        # This is a simplified implementation that would need to be expanded
        # for proper VAR autocovariance computation
        
        # Convert to state-space and use the state-space autocovariance
        ss_model = self.to_state_space()
        G = ss_model.get_covariance()
        
        # Initialize autocovariance sequence
        gamma = np.zeros((self.n, self.n, lags + 1))
        gamma[:, :, 0] = G  # Lag 0 is just the covariance
        
        # Compute remaining lags using the VAR recursion
        for k in range(1, lags + 1):
            gamma_k = np.zeros((self.n, self.n))
            for j in range(min(k, self.p)):
                gamma_k += self.A[:, :, j] @ gamma[:, :, k - j - 1]
            gamma[:, :, k] = gamma_k
            
        return gamma 