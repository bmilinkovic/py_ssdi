#!/usr/bin/env python3
# File: run_optimisation.py

"""
run_optimisation.py

Final SSDI optimisation function that mirrors the MATLAB opt_gd_dds pipeline.
"""

import os
import pickle
import time
import numpy as np
from typing import Union, Tuple

from py_ssdi.preoptimisation.preoptim import Lcluster, itransform_subspace, gmetrics
from py_ssdi.model_generation.ss_utils import ss2trfun
from py_ssdi.model_generation.var_utils import var2trfun
from py_ssdi.optimisation.opt import opt_gd_dds_mruns



# ────────────────────────────────────────────────────────────
def run_optimisation(
    model_path: str,
    preopt_path: str,
    mdim: int,
    fres: Union[int, np.ndarray],
    *,
    niters_o: int = 10_000,
    init_step: float = 0.1,
    tol: Union[float, Tuple[float, float, float]] = 1e-10, 
    ctol: float = 1e-6,
    compute_history: bool = True,
    parallel: bool = False,
) -> dict:
    """
    Optimise dynamical dependence for one macroscopic dimension.
    `fres` may be either an int (Lionel style) or an explicit 1-D array of ω.
    """
    # ── load model pickle ───────────────────────────────────
    with open(model_path, "rb") as f:
        md = pickle.load(f)

    model_kind = md["model_type"]            # "VAR" or "SS"

    # SS parameters are always present (VAR has been converted already)
    A_ss, C_ss, K_ss = md["A_ss"], md["C_ss"], md["K_ss"]
    V0 = md.get("V_identity", np.eye(C_ss.shape[0]))

    # ── load pre-optim for this m ───────────────────────────
    with open(preopt_path, "rb") as f:
        pp = pickle.load(f)
    pre = pp["results"][f"m={mdim}"]

    Lp_norm, doptp, goptp = pre["Lp_norm"], pre["doptp"], pre["goptp"]
    n, m_saved, _ = Lp_norm.shape
    assert m_saved == mdim

    # ── cluster pre-optima ──────────────────────────────────
    uidx, _, _ = Lcluster(goptp, ctol, doptp,
                          gpterm=None, gpscale=None,
                          gpfsize=None, gpplot=False)
    reps = np.unique(uidx)
    reps = reps[np.argsort(doptp[reps])]     # best DD first
    L0o  = Lp_norm[:, :, reps]               # (n, mdim, nrunso)

    # ── build transfer function H ───────────────────────────
    if isinstance(fres, (np.ndarray, list)):
        omega = np.asarray(fres, dtype=float)
        fres_idx = len(omega) - 1
    else:           # scalar given
        fres_idx = int(fres)
        omega    = np.linspace(0.0, np.pi, fres_idx + 1)

    if model_kind == "VAR":
        A_dec_tensor = md["A_var_decor"] # (n,n,r_var)
        A_dec_list   = [A_dec_tensor[:, :, p] for p in range(A_dec_tensor.shape[2])]
        H = var2trfun(A_dec_list, fres_idx)
    else:  # "SS"
        H = ss2trfun(A_ss, C_ss, K_ss, omega)

    # ── optimisation parameters ─────────────────────────────
    gdtol = (tol, tol, tol) if np.isscalar(tol) else tuple(tol)
    if parallel:
        print("  [warning] parallel=True is not yet implemented – running serially")

    t0 = time.perf_counter()
    dopto, Lo, convp, iopto, sopto, cputo, ohisto = opt_gd_dds_mruns(
        H,
        L0o,
        niters=niters_o,
        gdsig0=init_step,
        gdls=2.0,
        gdtol=gdtol,
        hist=compute_history,
        parallel=False,
        variant=2,
    )
    elapsed = time.perf_counter() - t0

    # ── post-process ────────────────────────────────────────
    Lopto = itransform_subspace(Lo, V0)
    gopto = gmetrics(Lopto)

    print("\nOptimal DD (sorted):", dopto)
    print(f"Simulation time : {elapsed:.3f} s")
    print(f"CPU secs / run  : {np.mean(cputo):.4f} ± {np.std(cputo):.4f}")

    return {
        "dopto": dopto,
        "Lo": Lo,
        "Lopto": Lopto,
        "convp": convp,
        "iopto": iopto,
        "sopto": sopto,
        "cputo": cputo,
        "ohisto": ohisto,
        "gopto": gopto,
        "fres": fres_idx if isinstance(fres, int) else omega,
        "elapsed_time": elapsed,
        "mdim": mdim,
        "model_path": model_path,
        "preopt_path": preopt_path,
    }
