# compare_cak2ddxgrad.py

import numpy as np
import scipy.io
from py_ssdi.preoptimisation.preoptim import cak2ddxgrad

# 1) Load Python‐generated test data
data_in = scipy.io.loadmat('debug_grad.mat')
L_py   = data_in['L_py']    # (5,2)
CAK_py = data_in['CAK_py']  # (5,5,3)

# 2) Compute Python’s (G_py, mG_py)
G_py, mG_py = cak2ddxgrad(L_py, CAK_py)

# 3) Load MATLAB’s outputs
data_out = scipy.io.loadmat('/Users/borjan/code/dmt_project/ssdi-1/debug_grad_results.mat')
G_matlab  = data_out['G_matlab']   # (5,2)
mG_matlab = data_out['mG_matlab'][0,0]  # scalar in a 1×1 array

# 4) Compare shapes
print("G_py shape      =", G_py.shape)
print("G_matlab shape  =", G_matlab.shape)
print("mG_py           =", mG_py)
print("mG_matlab       =", mG_matlab)

# 5) Compute differences
diff_G = np.linalg.norm(G_py - G_matlab)
diff_mG = abs(mG_py - mG_matlab)
print(f"||G_py - G_matlab||_F = {diff_G:.6e}")
print(f"|mG_py - mG_matlab|   = {diff_mG:.6e}")

# 6) If they do not match exactly, print small excerpts
if diff_G > 1e-12 or diff_mG > 1e-12:
    print("\nG_py:\n", G_py)
    print("\nG_matlab:\n", G_matlab)
    print(f"\nmG_py = {mG_py}, mG_matlab = {mG_matlab}")
else:
    print("\nG and mG match MATLAB exactly.")
