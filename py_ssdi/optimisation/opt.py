# File: py_ssdi/optimisation/opt.py

import time
import numpy as np
from numpy.linalg import svd
from scipy.linalg import cholesky
from py_ssdi.preoptimisation.preoptim import orthonormalise

def trfun2dd(L: np.ndarray, H: np.ndarray) -> float:
    """
    Calculate spectral dynamical dependence D for subspace L and transfer function H.
    """
    # H: (n, n, h), L: (n, m)
    h = H.shape[2]
    d = np.zeros(h, dtype=complex)
    for k in range(h):
        Hk = H[:, :, k]
        Qk = Hk.conj().T @ L            # (n, m)
        # Compute (log-determinant)/2 of Qk'Qk
        try:
            R = cholesky(Qk.conj().T @ Qk, lower=False)
            d[k] = np.sum(np.log(np.diag(R)))
        except np.linalg.LinAlgError:
            # safer: bypass SVD and use eigs on the Hermitian product
            # lam = np.linalg.eigvalsh(Qk.conj().T @ Qk)   # all ≥ 0
            # d[k] = 0.5 * np.sum(np.log(lam))             # ½·logdet
            # # If Cholesky fails, use SVD instead
            _, s, _ = svd(Qk, full_matrices=False)
            d[k] = np.sum(np.log(s))
    # trapezoidal integration over frequencies
    return np.real(np.sum(d[:-1] + d[1:]) / (h - 1))

def trfun2ddgrad(L: np.ndarray, H: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Calculate gradient G and magnitude mG of spectral dynamical dependence for L and H.
    """
    n, m = L.shape
    h = H.shape[2]
    g_tensor = np.zeros((n, m, h), dtype=complex)
    for k in range(h):
        Hk = H[:, :, k]
        HLk = Hk.conj().T @ L                   # (n, m)
        # (Hk * HLk) @ inv(HLk' HLk)
        M = HLk.conj().T @ HLk
        try:
            invM = np.linalg.inv(M)
            g_tensor[:, :, k] = (Hk @ HLk) @ invM
        except np.linalg.LinAlgError:
            # If matrix is singular, use pseudoinverse
            invM = np.linalg.pinv(M)
            g_tensor[:, :, k] = (Hk @ HLk) @ invM
    # integrate with trapezoidal rule
    G = np.real(np.sum(g_tensor[:, :, :-1] + g_tensor[:, :, 1:], axis=2) / (h - 1)) - 2 * L
    mG = np.linalg.norm(G)
    return G, mG

def opt_gd1_dds(H, L0, maxiters, gdsig0, gdls, tol, hist):
    """
    Variant 1 gradient descent for spectral DD.
    """
    # gain/loss
    if np.isscalar(gdls):
        ifac, nfac = gdls, 1.0/gdls
    else:
        ifac, nfac = gdls[0], gdls[1]
    # tolerances
    if np.isscalar(tol):
        stol = dtol = gtol = tol
    else:
        stol, dtol, gtol = tol

    # initial
    L = L0.copy()
    G, g = trfun2ddgrad(L, H)
    dd = trfun2dd(L, H)
    sig = gdsig0
    dhist = [] if not hist else np.zeros((maxiters, 3))
    if hist:
        dhist[0] = [dd, sig, g]

    converged, iters = 0, 1
    for it in range(1, maxiters):
        Ltry = orthonormalise(L - sig*(G/g))
        ddtry = trfun2dd(Ltry, H)
        if ddtry < dd:
            L = Ltry
            G, g = trfun2ddgrad(L, H)
            dd = ddtry
            sig *= ifac
        else:
            sig *= nfac
        if hist:
            dhist[it] = [dd, sig, g]
        if sig < stol:
            converged, iters = 1, it+1
            break
        elif dd < dtol:
            converged, iters = 2, it+1
            break
        elif g < gtol:
            converged, iters = 3, it+1
            break
    if hist:
        dhist = dhist[:iters]
    return dd, L, converged, sig, iters, dhist

def opt_gd2_dds(H, L0, maxiters, gdsig0, gdls, tol, hist):
    """
    Variant 2 gradient descent (always take the step, then evaluate DD).
    """
    # gain / loss factors
    if np.isscalar(gdls):
        ifac, nfac = gdls, 1.0 / gdls
    else:
        ifac, nfac = gdls

    # tolerances
    if np.isscalar(tol):
        stol = dtol = gtol = tol
    else:
        stol, dtol, gtol = tol

    L = L0.copy()
    G, g = trfun2ddgrad(L, H)      # gradient + magnitude
    dd   = trfun2dd(L, H)
    sig  = gdsig0

    dhist = None
    if hist:
        dhist = np.zeros((maxiters, 3))
        dhist[0] = [dd, sig, g]

    converged, iters = 0, 1
    for it in range(1, maxiters):
        # Always step along the current gradient
        L = orthonormalise(L - sig * (G / g))

        # New gradient and DD at the *moved* position
        G, g = trfun2ddgrad(L, H)
        ddnew = trfun2dd(L, H)

        # 1+1 ES step-size control
        if ddnew < dd:
            dd  = ddnew
            sig = ifac * sig
        else:
            sig = nfac * sig

        if hist:
            dhist[it] = [dd, sig, g]

        # convergence tests
        if   sig < stol: converged = 1; iters = it + 1; break
        elif dd  < dtol: converged = 2; iters = it + 1; break
        elif g   < gtol: converged = 3; iters = it + 1; break

    if hist:
        dhist = dhist[:iters]
    return dd, L, converged, sig, iters, dhist


def opt_gd_dds_mruns(
    H, L0, niters, gdsig0, gdls, gdtol, hist=False, parallel=False, variant=2
):
    """
    Multiple‐run optimisation for spectral DD.
    """
    nruns = L0.shape[2]
    dopt = np.zeros(nruns)
    Lopt = np.zeros_like(L0)
    conv = np.zeros(nruns, dtype=int)
    sopt = np.zeros(nruns)
    iopt = np.zeros(nruns, dtype=int)
    cput = np.zeros(nruns)
    ohist = [None]*nruns if hist else []

    print(f"\nStarting {nruns} optimization runs with {niters} max iterations each")
    
    for k in range(nruns):
        t0 = time.perf_counter()
        if variant == 1:
            d, Lf, cv, sf, itc, hst = opt_gd1_dds(H, L0[:,:,k], niters, gdsig0, gdls, gdtol, hist)
        elif variant == 2:
            d, Lf, cv, sf, itc, hst = opt_gd2_dds(H, L0[:,:,k], niters, gdsig0, gdls, gdtol, hist)
        else:
            raise ValueError("variant must be 1 or 2")
        t1 = time.perf_counter()
        elapsed_cpu = t1 - t0
        
        dopt[k] = d
        Lopt[:,:,k] = Lf
        conv[k] = cv
        sopt[k] = sf
        iopt[k] = itc
        cput[k] = elapsed_cpu
        if hist:
            ohist[k] = hst
            
        # Print progress line
        status = (
            f"GD/DD({variant}) serial run {k+1:3d} of {nruns:3d} : "
            f"dd = {d:.4e} : "
            f"sig = {sf:.4e} : "
            f"{'converged('+str(cv)+')' if cv>0 else 'unconverged'} "
            f"in {itc:4d} iter : CPU secs = {elapsed_cpu:6.2f}"
        )
        print(status)

    # sort by dopt ascending
    order = np.argsort(dopt)
    dopt = dopt[order]
    Lopt = Lopt[:,:,order]
    conv = conv[order]
    sopt = sopt[order]
    iopt = iopt[order]
    cput = cput[order]
    if hist:
        ohist = [ohist[i] for i in order]
    return dopt, Lopt, conv, iopt, sopt, cput, ohist

# End of opt.py
