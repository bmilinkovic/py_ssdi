#!/usr/bin/env python3
"""
debug_pipeline_short.py

After you’ve run:
    model_path = simulate_model("VAR", n=9, r_var=2, connectivity_name="tnet9x", ...)
this script reloads that pickle, computes CAK, and prints norms at every step.
"""

import pickle
import numpy as np
from numpy.linalg import norm, eigvals
import os

def print_norms(name, M):
    """
    Print Frobenius norm, spectral norm, and max‐abs entry of M.
    Handles both 2D and 3D arrays (for CAK).
    """
    if M.ndim == 2:
        fn = norm(M, 'fro')
    else:
        # For 3D, just treat as flattened
        fn = norm(M.ravel(), 2)

    # Spectral norm (2‐norm) only makes sense for 2D. For 3D, take max spectral among slices.
    if M.ndim == 2:
        sn = norm(M, 2)
    else:
        # compute spectral norm of each frontal slice, then take the maximum
        sn = max(norm(M[:, :, k], 2) for k in range(M.shape[2]))

    ma = np.max(np.abs(M))
    print(f"{name:20s}  shape={M.shape}  •  ‖·‖_F={fn:.4e}  •  ‖·‖₂={sn:.4e}  •  max|·|={ma:.4e}")

def main():
    # 1) Point to your saved VAR→SS model
    model_path = "py_ssdi/results/model_parameters/VAR_n9_r2_tnet9x.pkl"
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Cannot find model pickle at {model_path}")

    # 2) Reload the pickle
    with open(model_path, 'rb') as f:
        mp = pickle.load(f)

    print("\n=== Inspecting VAR→SS model parameters ===\n")

    # 3) Raw VAR lags, if present
    if 'A_var' in mp:
        A_var = mp['A_var']  # (n, n, r_var)
        n, _, r_var = A_var.shape
        print(f"Raw VAR A_var (n={n}, r_var={r_var}):")
        for k in range(r_var):
            print_norms(f"  A_var[:,:, {k}]", A_var[:, :, k])
    else:
        print("No 'A_var' found in pickle—skipping raw VAR check.")

    # 4) Decorrelated VAR lags
    if 'A_var_decor' in mp:
        A_var_decor = mp['A_var_decor']
        n, _, r_var = A_var_decor.shape
        print(f"\nDecorrelated VAR A_var_decor (n={n}, r_var={r_var}):")
        for k in range(r_var):
            print_norms(f"  A_var_decor[:,:, {k}]", A_var_decor[:, :, k])
    else:
        print("No 'A_var_decor' found—skip transform_var check.")

    # 5) Companion‐form SS: A_ss, C_ss, K_ss
    if all(key in mp for key in ('A_ss', 'C_ss', 'K_ss')):
        A_ss = mp['A_ss']
        C_ss = mp['C_ss']
        K_ss = mp['K_ss']
        print("\nCompanion‐form SS matrices:")
        print_norms("  A_ss", A_ss)
        print_norms("  C_ss", C_ss)
        print_norms("  K_ss", K_ss)
        # Spectral radius of A_ss:
        eigs = eigvals(A_ss)
        rho = np.max(np.abs(eigs))
        print(f"  Spectral radius of A_ss = {rho:.4e}")
    else:
        print("A_ss / C_ss / K_ss not found—perhaps this wasn’t a VAR‐based model.")

    # 6) Reconstruct CAK exactly as `iss_to_CAK` would
    print("\nReconstructing CAK from (A_ss, C_ss, K_ss)…")
    n = mp['C_ss'].shape[0]
    r_full = mp['A_ss'].shape[0]
    CAK = np.zeros((n, n, r_full))
    for k in range(r_full):
        CAK[:, :, k] = C_ss @ np.linalg.matrix_power(A_ss, k) @ K_ss

    print_norms("  CAK (full tensor)", CAK)
    print("  Per‐slice CAK norms:")
    for k in range(r_full):
        print_norms(f"    CAK[:,:, {k}]", CAK[:, :, k])

    print("\n=== End of diagnostics ===\n")

if __name__ == '__main__':
    main()

