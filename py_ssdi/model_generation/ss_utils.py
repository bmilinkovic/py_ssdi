# ss_utils.py
import numpy as np
from numpy.linalg import cholesky, inv, matrix_power, slogdet
import scipy.linalg as la
from py_ssdi.model_generation.var_utils import nextpow2, specnorm
from typing import Optional



def ss_check_fres(A, C, K, V, fres: int) -> float:
    """
    ∣ ∫₀^π log det S(ω) dω  −  log det V ∣   where
    S(ω) = H(ω) V H(ω)ᴴ  and  H = ss2trfun(A,C,K,ω).
    """
    omega = np.linspace(0.0, np.pi, fres + 1)      # 0 … π  (fres+1 points)
    H = ss2trfun(A, C, K, omega)                   # (n,n,fres+1)

    logd = np.empty_like(omega)
    for k in range(len(omega)):
        S = H[:, :, k] @ V @ H[:, :, k].conj().T
        _, logdet_S = slogdet(S)
        logd[k] = logdet_S

    integral = np.trapz(logd, dx=np.pi / fres) / np.pi    # divide by π
    _, logdet_V = slogdet(V)
    return abs(integral - logdet_V)


# ------------------------------------------------------------
# main routine
# ------------------------------------------------------------
def ss2fres(
    A: np.ndarray,
    C: np.ndarray,
    K: np.ndarray,
    V: Optional[np.ndarray] = None,
    fast: bool = False,
    siparms: tuple[float, int, int] = (1e-12, 6, 14),
):
    """
    Choose an appropriate frequency resolution (power-of-two length) for
    state-space spectral integrals.

    Parameters
    ----------
    A, C, K : ndarray
        Innovations-form SS model matrices.
    V : ndarray or None
        Residual covariance; identity is used if None.
    fast : bool, default False
        Use heuristic based on autocovariance decay (identical to MATLAB's
        `fastm` branch).  Accurate branch integrates until tolerance met.
    siparms : (tol, minpow2, maxpow2)
        * tol      – allowable absolute error in logdet spectral integral.
        * minpow2  – minimum grid exponent (length = 2**minpow2).
        * maxpow2  – maximum grid exponent.

    Returns
    -------
    fres : int
        Chosen frequency resolution (power-of-two length, e.g. 64, 128, …).
    ierr : float
        Integral error (accurate mode) or current error (fast mode).
    """
    if V is None:
        V = np.eye(C.shape[0])

    tol, minpow2, maxpow2 = siparms

    if fast:
        # heuristic: use the larger spectral norm of A and A-K C
        rho = max(specnorm(A), specnorm(A - K @ C))
        frpow2 = nextpow2(np.log(np.finfo(float).eps) / np.log(rho))

        frpow2 = min(max(frpow2, minpow2), maxpow2)
        fres = 2**frpow2

        ierr = ss_check_fres(A, C, K, V, fres)
        return fres, ierr

    # accurate branch
    ierr = None
    for frpow2 in range(minpow2, maxpow2 + 1):
        fres_candidate = 2**frpow2
        ierr_candidate = ss_check_fres(A, C, K, V, fres_candidate)
        if ierr_candidate <= tol:
            fres, ierr = fres_candidate, ierr_candidate
            break
    else:
        print(f"[warning] ss2fres: need > 2^{maxpow2}; using 2^{maxpow2}")
        fres = 2**maxpow2
        ierr = ss_check_fres(A, C, K, V, fres)

    return fres, ierr


def corr_rand(n, seed=None):
    if seed is not None:
        np.random.seed(seed)
    X = np.random.randn(n, n)
    S = np.dot(X, X.T)
    D_inv = np.diag(1.0 / np.sqrt(np.diag(S)))
    R = D_inv @ S @ D_inv
    R = (R + R.T) / 2.0
    return R


def iss_rand(n, r, spectral_norm=0.95, seed=None):
    """
    Generate random state-space (A0, C0, K0) with hidden dimension r and observation dimension n,
    and random residual covariance V (n x n).
    Ensure A0 is stable with spectral radius < spectral_norm.
    """
    if seed is not None:
        np.random.seed(seed)
    # Generate random residual covariance V
    V = corr_rand(n, seed=seed)
    # Generate random A0 (r x r)
    A0 = np.random.randn(r, r)
    # Scale to have spectral radius < spectral_norm
    eigs = la.eigvals(A0)
    max_eig = np.max(np.abs(eigs))
    if max_eig >= spectral_norm:
        A0 = A0 * (spectral_norm / (max_eig + 1e-8))
    # Generate random C0 (n x r) and K0 (r x n)
    C0 = np.random.randn(n, r)
    K0 = np.random.randn(r, n)
    return A0, C0, K0, V


def transform_ss(A0, C0, K0, V):
    """
    Decorrelate and normalize SS residuals using Cholesky factor of V.
    Returns (A, C, K, V_norm) where V_norm = identity.
    """
    # Cholesky lower-triangular factor of V
    C_chol = cholesky(V)
    C_chol_inv = inv(C_chol)
    # Transform
    A = A0.copy()
    C = C_chol_inv @ C0
    K = K0 @ C_chol_inv
    V_norm = np.eye(V.shape[0])
    return A, C, K, V_norm


def iss_to_CAK(A, C, K):
    """
    Convert a (A, C, K) innovations‐form SS into a CAK tensor of shape (n, n, r),
    exactly as MATLAB's iss2cak does.

    Parameters
    ----------
    A : ndarray, shape (r, r)
        State‐transition matrix.
    C : ndarray, shape (n, r)
        Observation matrix.
    K : ndarray, shape (r, n)
        Kalman gain matrix.

    Returns
    -------
    CAK : ndarray, shape (n, n, r)
        CAK[:,:,k] = C @ (A**k) @ K   for k = 0..r-1.
        This matches MATLAB's:
           CAK(:,:,1) = C*K   (i.e. k=0),
           CAK(:,:,2) = C*A*K (i.e. k=1),
           …,
           CAK(:,:,r) = C*A^(r-1)*K.
    """
    r = A.shape[0]
    n = C.shape[0]

    CAK = np.zeros((n, n, r))
    for k in range(r):
        # A^0 = eye(r), A^1 = A, A^2 = A@A, …, A^(r-1)
        A_pow = matrix_power(A, k)
        CAK[:, :, k] = C @ A_pow @ K

    return CAK


def ss_to_pwcgc(A, C, K, V):
    """
    Compute pairwise-conditional Granger causality from SS model.
    Not implemented: placeholder for future implementation.
    """
    raise NotImplementedError("ss_to_pwcgc is not yet implemented")

def ss2trfun(A: np.ndarray,
             C: np.ndarray,
             K: np.ndarray,
             fres: np.ndarray) -> np.ndarray:
    """
    Compute the transfer‐function tensor H for an innovations‐form SS model
    over frequencies 0..π.

    Parameters
    ----------
    A    : ndarray, shape (r, r)
           State‐transition matrix.
    C    : ndarray, shape (n, r)
           Observation matrix.
    K    : ndarray, shape (r, n)
           Kalman‐gain matrix.
    fres : ndarray
           Array of frequencies in [0, π] at which to evaluate H.

    Returns
    -------
    H : ndarray, shape (n, n, len(fres)), dtype complex
        H[:,:,k] = I_n + C @ ((w[k] I_r - A)^{-1} @ K),
        where w[k] = exp(i fres[k]).
    """
    n, r = C.shape
    h = len(fres)

    # Pre‐allocate complex array
    H = np.zeros((n, n, h), dtype=complex)

    # Identity matrices
    In = np.eye(n, dtype=complex)
    Ir = np.eye(r, dtype=complex)

    # Frequencies w = exp(i fres)
    w = np.exp(1j * fres)

    # Loop over frequencies
    for k in range(h):
        # Solve (w[k] I_r - A) X = K  →  X = (w[k] I_r - A)^{-1} @ K
        X = np.linalg.solve(w[k] * Ir - A, K)
        # Build H slice
        H[:, :, k] = In + C @ X

    return H

