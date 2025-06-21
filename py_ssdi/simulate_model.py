#!/usr/bin/env python3
"""
simulate_model.py

Defines a single function simulate_model() that can be called to generate either:
  • A VAR(n, r_var) model (using a named connectivity), or
  • A “pure” SS model with hidden dimension = 3 × n_obs.

All generated parameters are saved (as a pickle) under:
    py_ssdi/results/model_parameters/

At the end of the call, simulate_model() also prints out the name and shape of each saved array.
"""

import os
import pickle
import numpy as np

# Import our utilities
from py_ssdi.model_generation.connectivity_utils import load_connectivity
from py_ssdi.model_generation.var_utils         import corr_rand, var_rand, transform_var, var_to_ss, var2fres
from py_ssdi.model_generation.ss_utils          import iss_rand, transform_ss, iss_to_CAK, ss2fres

def simulate_model(model_type: str, **kwargs):
    """
    Simulate a model and save its parameters under py_ssdi/results/model_parameters/.

    Parameters
    ----------
    model_type : str
        Either "VAR" or "SS" (case-insensitive).
        - "VAR": simulate a VAR(n, r_var) using a named connectivity.
        - "SS" : simulate a direct SS with hidden state dimension = 3 * n_obs.

    For model_type == "VAR", kwargs must include:
        n                : int, number of observed variables.
        r_var            : int, VAR order (number of lags).
        connectivity_name: str, e.g. "tnet9x" or "tnet9d".
        seed             : int (optional; default=0)
        spectral_norm    : float (optional; default=0.95)
        decay            : float (optional; default=1.0)

    For model_type == "SS", kwargs must include:
        n_obs            : int, observation‐dimension.
        seed             : int (optional; default=0)
        spectral_norm    : float (optional; default=0.95)

        (hidden‐state dimension r_state is set internally to 3*n_obs.)

    Returns
    -------
    out_path : str
        Full path of the pickle file into which model parameters were saved.
    """

    model_type = model_type.strip().upper()
    # Locate the “results/model_parameters/” folder under py_ssdi/
    this_dir   = os.path.dirname(__file__)
    results_dir = os.path.join(this_dir, "results", "model_parameters")
    os.makedirs(results_dir, exist_ok=True)

    params = {}  # Dictionary to collect everything to save

    if model_type == "VAR":
        # Required args for VAR
        n                = kwargs["n"]
        r_var            = kwargs["r_var"]
        connectivity_name= kwargs["connectivity_name"]
        seed             = kwargs.get("seed", 0)
        spectral_norm    = kwargs.get("spectral_norm", 0.95)
        decay            = kwargs.get("decay", 1.0)

        # 1) Load connectivity
        Cmat = load_connectivity(connectivity_name)
        # Stack into (n, n, r_var)
        connectivity = np.stack([Cmat for _ in range(r_var)], axis=2)

        # 2) Random residual‐covariance
        V = corr_rand(n, seed=seed)

        # 3) Generate VAR coefficient matrices A_list = [A1, …, A_r_var]
        A_original = var_rand(connectivity, n, r_var,
                          spectral_norm=spectral_norm,
                          decay=decay,
                          seed=seed)

        # 4) Decorrelate VAR residuals
        A_original_decor, V_identity = transform_var(A_original, V)

        # 5) Convert to innovations SS conversion
        A_ss, C_ss, K_ss = var_to_ss(A_original_decor, V_identity)

        # 6) choose frequency resolution
        fres, ierr = var2fres(A_original_decor, V_identity, fast=False)

        # 6) Stack A_list and A_list_decor into 3D tensors
        A_var_cor       = np.stack(A_original,       axis=2)  # shape = (n, n, r_var)
        A_var_decor = np.stack(A_original_decor, axis=2)

        # 7) Package everything into the params dict
        params["model_type"]       = "VAR"
        params["n"]                = n
        params["r_var"]            = r_var
        params["connectivity_name"]= connectivity_name
        params["seed"]             = seed
        params["spectral_norm"]    = spectral_norm
        params["decay"]            = decay
        params["fres"]             = fres
        params["ierr"]             = ierr

        params["connectivity"]     = connectivity    # (n, n, r_var)
        params["V"]                = V               # (n, n)
        params["A_var_cor"]            = A_var_cor          # (n, n, r_var)
        params["V_identity"]       = V_identity      # (n, n), identity
        params["A_var_decor"]      = A_var_decor     # (n, n, r_var)

        params["A_ss"]             = A_ss            # (n*r_var, n*r_var)
        params["C_ss"]             = C_ss            # (n, n*r_var)
        params["K_ss"]             = K_ss            # (n*r_var, n)

        # Filename: e.g. “VAR_n9_r2_tnet9x.pkl”
        fname = f"VAR_n{n}_r{r_var}_{connectivity_name}.pkl"

    elif model_type == "SS":
        # Required args for pure SS
        n_obs         = kwargs["n_obs"]
        seed          = kwargs.get("seed", 0)
        spectral_norm = kwargs.get("spectral_norm", 0.95)
        r_state       = 3 * n_obs

        # 1) Generate random SS (A0, C0, K0, V0)
        A0, C0, K0, V0 = iss_rand(n_obs, r_state,
                                 spectral_norm=spectral_norm,
                                 seed=seed)

        # 2) Decorrelate SS residuals
        A_decor, C_decor, K_decor, V_norm = transform_ss(A0, C0, K0, V0)

        # 3) pick frequency resolution
        fres, ierr = ss2fres(A_decor, C_decor, K_decor, V_norm, fast=False)

        # 4) Build CAK tensor
        CAK = iss_to_CAK(A_decor, C_decor, K_decor)  # shape = (n_obs, n_obs, r_state)

        # 4) Package everything
        params["model_type"] = "SS"
        params["n_obs"]      = n_obs
        params["r_state"]    = r_state
        params["seed"]       = seed
        params["spectral_norm"] = spectral_norm
        params["fres"]          = fres
        params["ierr"]          = ierr

        params["A0"]   = A0    # (r_state, r_state)
        params["C0"]   = C0    # (n_obs, r_state)
        params["K0"]   = K0    # (r_state, n_obs)
        params["V0"]   = V0    # (n_obs, n_obs)

        params["V_norm"] = V_norm  # (n_obs, n_obs), identity
        params["A_decor"]  = A_decor   # (r_state, r_state)
        params["C_decor"]  = C_decor   # (n_obs, r_state)
        params["K_decor"]  = K_decor   # (r_state, n_obs)

        params["CAK"]    = CAK     # (n_obs, n_obs, r_state)

        # Filename: e.g. “SS_n5_r15.pkl”
        fname = f"SS_n{n_obs}_r{r_state}.pkl"

    else:
        raise ValueError("model_type must be 'VAR' or 'SS' (case-insensitive).")

    # Save `params` as a pickle
    out_path = os.path.join(results_dir, fname)
    with open(out_path, "wb") as f:
        pickle.dump(params, f)

    # Print out names and shapes of everything we saved
    print(f"\nModel parameters saved to: {out_path}\n")
    for key, val in params.items():
        if isinstance(val, np.ndarray):
            print(f"  • {key}: shape = {val.shape}")
        else:
            print(f"  • {key}: {type(val).__name__}  (value = {val})")

    return out_path
