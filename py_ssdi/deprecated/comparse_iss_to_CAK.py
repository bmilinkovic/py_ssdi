# compare_iss_to_CAK.py
import numpy as np
import scipy.io
from py_ssdi.model_generation.ss_utils import iss_to_CAK

# 1) Load from debug_iss.mat (Python’s A0_py, C0_py, K0_py)
data_in = scipy.io.loadmat('debug_iss.mat')
A0_py = data_in['A0_py']  # shape (r, r)
C0_py = data_in['C0_py']  # shape (n, r)
K0_py = data_in['K0_py']  # shape (r, n)

# 2) Compute Python’s CAK
CAK_py = iss_to_CAK(A0_py, C0_py, K0_py)

# 3) Load MATLAB’s CAK
data_mat = scipy.io.loadmat('/Users/borjan/code/dmt_project/ssdi-1/debug_iss_results.mat')
CAK_matlab = data_mat['CAK_matlab']  # shape (n, n, r)

# 4) Compare shapes and entries
print("CAK_py shape    =", CAK_py.shape)
print("CAK_matlab shape=", CAK_matlab.shape)

diff = np.linalg.norm(CAK_py - CAK_matlab)
print(f"||CAK_py - CAK_matlab||_F = {diff:.6e}")

# Display small slices if not identical
if diff < 1e-12:
    print("CAK matches MATLAB exactly.")
else:
    print("Max abs‐difference per element:", np.max(np.abs(CAK_py - CAK_matlab)))
    # Optionally print both side by side for k=0..r-1
    r = CAK_py.shape[2]
    for k in range(r):
        print(f"\nCAK_py[:,:,{k}]:\n", CAK_py[:, :, k])
        print(f"CAK_matlab[:,:,{k}]:\n", CAK_matlab[:, :, k])
