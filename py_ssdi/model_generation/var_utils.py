# var_utils.py
import numpy as np
from numpy.linalg import cholesky, inv, eigvals, svd, slogdet, solve
import scipy.linalg as la



def var2trfun(A_list, fres: int):
    """
    VAR transfer-function H(:,:,k) for k = 0…fres (ω = 0…π).
    A_list = [A1 … Ap] with shape (n,n) each.
    """
    n = A_list[0].shape[0]
    I = np.eye(n)
    # Build polynomial [I, -A1, -A2, …, -Ap, 0 … 0] length 2*fres
    coeffs = [I] + [-A for A in A_list] + [np.zeros_like(I)]*(2*fres - len(A_list))
    Af = np.fft.fft(np.stack(coeffs, axis=2), axis=2)
    H = np.empty((n, n, fres+1), dtype=complex)
    for k in range(fres+1):          # 0 … π
        H[:, :, k] = np.linalg.solve(Af[:, :, k], I)
    return H

def corr_rand(n, seed=None):
    """
    Generate a random n x n positive-definite correlation matrix.
    """
    if seed is not None:
        np.random.seed(seed)
    # Sample a random matrix
    X = np.random.randn(n, n)
    # Form covariance-like matrix
    S = np.dot(X, X.T)
    # Convert to correlation matrix
    D_inv = np.diag(1.0 / np.sqrt(np.diag(S)))
    R = D_inv @ S @ D_inv
    # Ensure symmetry
    R = (R + R.T) / 2.0
    return R


def var_rand(connectivity, n, r, spectral_norm=0.95, decay=0.0, seed=None):
    """
    A “MATLAB-style” var_rand that exactly mirrors specnorm→var_decay logic.
    
    connectivity:  (nxnxr) binary mask array
    n:             number of variables
    r:             model order (number of lags)
    spectral_norm: the target spectral radius (|rho| < 1 for stability)
    decay:         if >0, apply an exponential-decay factor exp(-decay*sqrt(r)) before specnorm
    seed:          optional RNG seed
    """
    if seed is not None:
        np.random.seed(seed)

    # 1) Build raw masked lags with optional exponential‐decay factor
    A_list = []
    for k in range(r):
        mask = connectivity[:, :, k]
        A_k = np.random.randn(n, n) * mask
        if decay != 0.0:
            A_k = np.exp(-decay * np.sqrt(r)) * A_k
        A_list.append(A_k)

    # 2) Build companion matrix from these A_list
    comp = np.zeros((n*r, n*r))
    for k in range(r):
        comp[0:n, k*n:(k+1)*n] = A_list[k]
    for i in range(1, r):
        comp[i*n:(i+1)*n, (i-1)*n:i*n] = np.eye(n)

    # 3) Compute current spectral radius
    eigs = eigvals(comp)
    lambda_max = np.max(np.abs(eigs))

    # 4) If it already meets the target, return as is
    if lambda_max <= spectral_norm:
        return A_list

    # 5) Otherwise, compute alpha so that alpha^k A_list[k] yields |new rho| = spectral_norm
    alpha = (spectral_norm / (lambda_max + 1e-12)) ** (1.0 / r)

    # 6) Rescale each lag by alpha^k
    A_list_new = []
    for k in range(r):
        scale_factor = alpha ** (k + 1)  # MATLAB’s var_decay uses k starting at 1
        A_list_new.append(A_list[k] * scale_factor)

    return A_list_new




def nextpow2(x: float) -> int:
    """Return the smallest integer n such that 2**n >= x."""
    return int(np.ceil(np.log2(x)))


def specnorm(A_list) -> float:
    """Spectral norm (largest singular value) of concatenated VAR block matrix."""
    # stack VAR coefficient matrices horizontally then compute σ_max
    A_big = np.hstack(A_list)
    return svd(A_big, compute_uv=False)[0]


def var_check_fres(A_list, V, fres: int) -> float:
    """
    Integral error ϵ = |∫ log det S(ω) dω - log det V|.
    Implements the check used in Lionel Barnett's MATLAB code.
    """
    r = A_list[0].shape[0]
    # FFT frequencies ω_k = 2πk / fres  (we need 0..π)
    z = np.exp(1j * np.pi * np.arange(fres + 1) / fres)   # 0 .. π
    logd = np.empty_like(z, dtype=float)

    # build polynomial A(z⁻¹) = I - Σ A_p z^{-p}
    I = np.eye(r, dtype=complex)
    for k, zk in enumerate(z):
        Az = I.copy()
        for p, Ap in enumerate(A_list, start=1):
            Az -= Ap * zk**(-p)
        # innovations spectral matrix S = (Az)^{-1} * V * (Azᵀ)^{-1}
        Az_inv = np.linalg.inv(Az)
        S = Az_inv @ V @ Az_inv.conj().T
        sign, logdet_S = slogdet(S)
        logd[k] = logdet_S

    # trapezoidal integral over 0..π, divide by π
    integral = np.trapz(logd, dx=np.pi / fres) / np.pi
    sign, logdet_V = slogdet(V)
    return abs(integral - logdet_V)


def var2fres(A_list, V=None, fast=False, siparms=(1e-12, 6, 14)):
    """
    Choose an appropriate frequency resolution for VAR spectral integrals.

    Parameters
    ----------
    A_list : list[ndarray]
        List of VAR coefficient matrices A_p, each shape (n,n).
    V : ndarray or None
        Residual covariance (n,n).  If None, identity is assumed.
    fast : bool, default False
        If True, use the quick heuristic based on spectral norm.
    siparms : tuple (tol, minpow2, maxpow2)
        * tol      - allowable error in logdet spectral integral.
        * minpow2  - minimum grid length 2**minpow2.
        * maxpow2  - maximum grid length 2**maxpow2.

    Returns
    -------
    fres : int
        Power-of-two grid length (e.g. 64, 128, …; **not** the power itself).
    ierr : float
        Integral error (only in accurate mode; `None` in fast mode).
    """
    if V is None:
        V = np.eye(A_list[0].shape[0])

    tol, minpow2, maxpow2 = siparms

    if fast:
        # heuristic based on autocorrelation decay
        rho = specnorm(A_list)
        frpow2 = nextpow2(np.log(np.finfo(float).eps) / np.log(rho))
        frpow2 = min(max(frpow2, minpow2), maxpow2)
        fres = 2**frpow2
        ierr = var_check_fres(A_list, V, fres) if tol is not None else None
        return fres, ierr

    # accurate branch: grow grid until integral matches logdet V within tol
    ierr = None
    for frpow2 in range(minpow2, maxpow2 + 1):
        fres_candidate = 2**frpow2
        ierr_candidate = var_check_fres(A_list, V, fres_candidate)
        if ierr_candidate <= tol:
            ierr = ierr_candidate
            fres = fres_candidate
            break
    else:
        # failed to meet tolerance; default to max grid
        print(f"[warning] resolution > 2^{maxpow2} required - using 2^{maxpow2}")
        fres = 2**maxpow2
        ierr = var_check_fres(A_list, V, fres)

    return fres, ierr


def transform_var(A_list, V):
    """
    Decorrelate (and normalize) VAR residuals so that V_new = I.
    Mirrors MATLAB's [A,V] = transform_var(A,V) exactly.
    """
    # 1) Get the LOWER-triangular factor L so that V = L @ L.T
    L = cholesky(V)   # <— explicitly ask for lower‐triangular

    # 2) Compute its inverse
    L_inv = inv(L)

    # 3) Apply A_new = L_inv @ A_p @ L  for each lag
    A_list_transformed = []
    for A_p in A_list:
        A_p_new = L_inv @ A_p @ L
        A_list_transformed.append(A_p_new)

    # 4) Residual covariance is now the identity
    V_new = np.eye(V.shape[0])
    return A_list_transformed, V_new

def var_to_ss(A_list, V):
    """
    Convert innovations-form VAR to state-space companion form (A, C, K).
    A_list: list of p covariance-normalized A matrices (each nxn)
    V: residual covariance (nxn), assumed identity (after transform_var).

    Implements:
      [A, C, K] = var_to_ss(VARA, V) where VARA is nxnxp, and C = [A1 ... Ap],
      A = [C; eye((p-1)*n), zeros((p-1)*n, n)],
      K = [eye(n); zeros((p-1)*n, n)].
    """
    n = A_list[0].shape[0]
    p = len(A_list)
    # Build C as [A1, A2, ..., Ap]
    C_mat = np.hstack(A_list)  # shape (n, n*p)
    # Compute companion A_ss of size (n*p) x (n*p)
    A_ss = np.zeros((n * p, n * p))
    # Top row: all lag matrices concatenated
    A_ss[0:n, :] = C_mat
    # Subdiagonal identity blocks
    for i in range(1, p):
        A_ss[i * n:(i + 1) * n, (i - 1) * n:i * n] = np.eye(n)
    # Build K: [I_n; zeros((p-1)*n, n)]
    pn1 = (p - 1) * n
    K_ss = np.vstack([np.eye(n), np.zeros((pn1, n))])
    # Return companion A, block C, and Kalman gain K
    return A_ss, C_mat, K_ss


def var_to_pwcgc(A_list, V):
    """
    Compute pairwise-conditional Granger causality from a VAR model.
    Not implemented: placeholder for future implementation.
    """
    raise NotImplementedError("var_to_pwcgc is not yet implemented")


