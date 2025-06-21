# preoptimise.py
import os
import pickle
import numpy as np
from numpy.random import RandomState
from py_ssdi.optimisation.opt import orthonormalise

from .preoptim import (
    Lcluster,
    opt_gd_ddx_mruns,
    gmetrics,
    itransform_subspace
)


def run_preoptimisation(model_pickle_path,
                        nrunsp=100,
                        niters_p=10000,
                        init_step=1.0,
                        tol=1e-8,
                        ctol=1e-3,
                        compute_history=False,
                        parallel=False):
    """
    Orchestrate the pre-optimisation procedure given a saved SS model.

    Parameters
    ----------
    model_pickle_path : str
        Path to .pkl containing model parameters.
    nrunsp : int
        Number of pre-optimisation restarts.
    niters_p : int
        Max iterations for gradient descent.
    init_step : float
        Initial step size.
    tol : float or sequence of float
        Convergence tolerance(s). If float, used for [stol, dtol, gtol].
    ctol : float
        Clustering tolerance for Lcluster (not used in preoptimisation).
    compute_history : bool
        Whether to record and return optimisation history.
    parallel : bool
        Whether to parallelize runs.

    Returns
    -------
    out_path : str
        Path where pre-optimisation results are saved.
    """

    # 1) Prepare output directory
    this_dir    = os.path.dirname(__file__)
    results_dir = os.path.join(this_dir, "results", "preoptimisation_results")
    os.makedirs(results_dir, exist_ok=True)

    # 2) Load model parameters
    with open(model_pickle_path, 'rb') as f:
        model_params = pickle.load(f)

    # 3) Build CAK and residual covariance V
    if 'CAK' in model_params:
        CAK = model_params['CAK']
        V   = model_params.get('V0', np.eye(CAK.shape[0]))
    else:
        A_ss = model_params['A_ss']
        C_ss = model_params['C_ss']
        K_ss = model_params['K_ss']
        n    = C_ss.shape[0]
        r    = A_ss.shape[0]

        CAK = np.zeros((n, n, r))
        CAK[:, :, 0] = C_ss @ K_ss
        for k in range(2, r+1):
            Ak = np.linalg.matrix_power(A_ss, k)
            CAK[:, :, k-1] = C_ss @ Ak @ K_ss

        V = model_params.get('V_identity', np.eye(n))

    n_obs, _, _ = CAK.shape

    # 4) Prepare container for results
    preopt_results = {
        'model_pickle': model_pickle_path,
        'nrunsp':       nrunsp,
        'results':     {}
    }

    # 5) Set up reproducible RNG
    rng = RandomState(0)

    # 6) Loop over macroscopic scales m = 2, …, n_obs-2
    for m in range(2, n_obs-1):
        print(f"Starting pre-optimisation for m={m}")

        # 6a) Initialize random orthonormal bases L0
        L0 = np.empty((n_obs, m, nrunsp))
        for j in range(nrunsp):
            X = rng.standard_normal((n_obs, m))
            L0[:, :, j] = orthonormalise(X)      # SVD-based, matches MATLAB

        # 6b) Run the gradient-descent preoptimiser
        #     Pass through niters_p, init_step, tol, compute_history, parallel
        #     gdtol must be a length-3 sequence
        if isinstance(tol, (int, float)):
            gdtol = [tol, tol, tol]
        else:
            gdtol = list(tol)

        doptp, Lp_norm, convp, ioptp, soptp, cputp, histp = opt_gd_ddx_mruns(
            CAK,
            L0,
            niters=niters_p,
            gdsig0=init_step,
            gdls=1.2,
            gdtol=gdtol,
            hist=compute_history,
            parallel=parallel,
            variant=1
        )

        # 6c) Undo the normalization of residuals
        Lp_undec = itransform_subspace(Lp_norm, V)

        # 6d) Compute the preoptimisation similarity matrix
        goptp = gmetrics(Lp_undec)

        # 6e) Store everything for this m
        preopt_results['results'][f'm={m}'] = {
            'doptp':    doptp,
            'Lp_norm':  Lp_norm,
            'Lp_undec': Lp_undec,
            'convp':    convp,
            'ioptp':    ioptp,
            'soptp':    soptp,
            'cputp':    cputp,
            'histp':    histp,
            'goptp':    goptp,
            'V':        V
        }

        print(f"Completed m={m}: Lp_norm={Lp_norm.shape}, goptp={goptp.shape}")

    # 7) Dump to disk
    base      = os.path.splitext(os.path.basename(model_pickle_path))[0]
    out_fname = f"preopt_{base}.pkl"
    out_path  = os.path.join(results_dir, out_fname)

    with open(out_path, 'wb') as f:
        pickle.dump(preopt_results, f)

    print(f"\nPre-optimisation results saved to: {out_path}")
    return out_path
