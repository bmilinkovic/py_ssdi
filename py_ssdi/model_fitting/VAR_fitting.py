# py_ssdi/model_fitting/VAR_fitting.py
"""
Utilities to (1) select the VAR model order and (2) estimate the VAR model
for multi-trial, multivariate time–series data.

Data format
-----------
X : ndarray, shape (n, m, N)
    * n  – #channels
    * m  – #time points per trial
    * N  – #trials

Public API
----------
select_order(X, momax, regmode='OLS', alpha=(0.01, 0.05), verbose=True)
    → dict with AIC/BIC/HQC/LRT curves and the selected order.

fit_var(X, p, regmode='OLS')
    → A, V, E   (coefficients tensor, residual covariance, residuals)

Both functions de-mean the data channel-wise before processing.
"""

from __future__ import annotations
import numpy as np
from numpy.linalg import eigvals, cholesky
from scipy.stats import norm, f
from typing import Tuple

from statsmodels.tsa.api import VAR


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _demean(X: np.ndarray) -> np.ndarray:
    """Channel-wise temporal de-meaning; keeps original shape."""
    mu = X.mean(axis=1, keepdims=True)
    return X - mu


def _var_info(A: np.ndarray, V: np.ndarray) -> dict:
    """Stability + covariance checks (Lionel's `var_info`)."""
    p = A.shape[2]
    n = A.shape[0]
    # companion matrix eigenvalues
    companion_rows = np.hstack(A)            # n × np
    if p > 1:
        bottom = np.eye(n * (p - 1))
        bottom = np.pad(bottom, ((n, 0), (0, n)), "constant")
        companion = np.vstack((companion_rows, bottom))
    else:
        companion = companion_rows
    rho_max = np.abs(eigvals(companion)).max()
    try:
        cholesky(V)          # symmetric PD test
        cov_ok = True
    except np.linalg.LinAlgError:
        cov_ok = False
    return {"rho_max": rho_max, "cov_ok": cov_ok, "error": (rho_max >= 1) or (not cov_ok)}


# ----------------------------------------------------------------------
# 1) model-order selection
# ----------------------------------------------------------------------
def select_order(
    X: np.ndarray,
    momax: int,
    regmode: str = "OLS",
    alpha: Tuple[float, float] = (0.01, 0.05),
    verbose: bool = True,
) -> dict:
    """
    Replicates `tsdata_to_varmo` and `moselect` (without plots).

    Returns
    -------
    dict with keys:
        'aic','bic','hqc','lrt_pval'      – arrays length momax
        'opt_aic','opt_bic','opt_hqc','opt_lrt' – selected orders
    """
    X = _demean(X)
    n, m, N = X.shape
    # concatenate trials → shape (m*N, n)
    Xcat = X.transpose(1, 2, 0).reshape(m * N, n)

    aic = np.full(momax, np.nan)
    bic = np.full(momax, np.nan)
    hqc = np.full(momax, np.nan)
    llf = np.full(momax, np.nan)       # log-likelihood
    df  = np.full(momax, np.nan)       # #free params
    eff_n = (m - np.arange(1, momax + 1)) * N   # effective obs per order

    for p in range(1, momax + 1):
        model = VAR(Xcat)
        res   = model.fit(p, trend="nc")   # no constant (demeaned)
        llf[p - 1] = res.llf
        df[p - 1]  = res.params.size
        aic[p - 1] = res.aic
        bic[p - 1] = res.bic
        hqc[p - 1] = res.hqic

    # sequential likelihood-ratio test (Lütkepohl)
    lambda_stat = 2 * np.diff(eff_n * llf)           # length momax-1
    dof = n * n
    df2 = eff_n[1:] - n * np.arange(1, momax) - 1
    lrt_p = np.ones(momax)
    lrt_p[1:] = 1 - f.cdf(lambda_stat / dof, dof, df2)
    lralpha = alpha[0] / momax
    opt_lrt = 0
    for k in range(momax, 0, -1):
        if lrt_p[k - 1] < lralpha:
            opt_lrt = k
            break

    opt_aic = int(np.nanargmin(aic) + 1)
    opt_bic = int(np.nanargmin(bic) + 1)
    opt_hqc = int(np.nanargmin(hqc) + 1)

    if verbose:
        print("\nBest model orders")
        print("-----------------")
        print(f"AIC : {opt_aic:2d}")
        print(f"BIC : {opt_bic:2d}")
        print(f"HQC : {opt_hqc:2d}")
        print(f"LRT : {opt_lrt:2d}\n")

    return dict(
        aic=aic, bic=bic, hqc=hqc, lrt_pval=lrt_p,
        opt_aic=opt_aic, opt_bic=opt_bic, opt_hqc=opt_hqc, opt_lrt=opt_lrt,
    )


# ----------------------------------------------------------------------
# 2) VAR estimation
# ----------------------------------------------------------------------
def fit_var(
    X: np.ndarray,
    p: int,
    regmode: str = "OLS",
    *,
    return_resid: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Replicates `tsdata_to_var`.  Always fits via QR/OLS (statsmodels)
    because Morf's LWR is uncommon in Python toolkits.

    Returns
    -------
    A : ndarray, shape (n,n,p)
        Coefficient matrices (A[:,:,k] is k-lag matrix).
    V : ndarray, shape (n,n)
        Residual covariance (unbiased estimator).
    E : ndarray, shape (n, m-p, N)  (only if return_resid=True)
        Residual time series aligned with original data (first p lags dropped).
    """
    X = _demean(X)
    n, m, N = X.shape
    assert p < m, "model order too large for time series length"

    Xcat = X.transpose(1, 2, 0).reshape(m * N, n)
    model = VAR(Xcat)
    res   = model.fit(p, trend="nc")      # no constant

    A = res.coefs.swapaxes(0, 1)          # statsmodels: (p,n,n) → (n,n,p)
    V = res.resid_cov

    # stability & covariance check
    info = _var_info(A, V)
    if info["error"]:
        raise RuntimeError("VAR estimation failed stability / PD check")

    if return_resid:
        resid = res.resid.T                # (n, m*N - p)
        resid = resid.reshape(n, m - p, N)
        return A, V, resid
    return A, V, None
