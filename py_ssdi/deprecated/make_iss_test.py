# make_iss_test.py
import numpy as np
import scipy.io
from numpy.linalg import svd

# Fix the seed so Python's randn matches its own future calls
np.random.seed(2025)

# Choose small dimensions: n = 4, r = 3
n = 4
r = 3

# 1) Random A0 (r×r), C0 (n×r), K0 (r×n), and random positive‐definite V0 (n×n)
A0 = np.random.randn(r, r)
# Force A0 to be stable-ish (not strictly necessary for CAK test, but closer to real pipeline):
eigs = np.linalg.eigvals(A0)
max_eig = np.max(np.abs(eigs))
if max_eig > 1:
    A0 = A0 / (max_eig + 1e-8)

C0 = np.random.randn(n, r)
K0 = np.random.randn(r, n)
X = np.random.randn(n, n)
V0 = X @ X.T  # positive‐definite

# Save them to a .mat
scipy.io.savemat('debug_iss.mat', {
    'A0_py': A0,
    'C0_py': C0,
    'K0_py': K0
})
print("Wrote debug_iss.mat (A0_py, C0_py, K0_py).")
