# make_grad_test.py

import numpy as np
import scipy.io
from numpy.linalg import svd

# 1) Fix NumPy’s RNG
np.random.seed(0)

# 2) Choose dimensions: n = 5, m = 2, r = 3
n = 5
m = 2
r = 3

# 3) Construct a random orthonormal L (n×m)
X = np.random.randn(n, m)
U, _, _ = svd(X, full_matrices=False)
L_py = U[:, :m]  # shape (n, m)

# 4) Construct a random CAK tensor (n×n×r)
CAK_py = np.random.randn(n, n, r)

# 5) Save L_py and CAK_py to debug_grad.mat
scipy.io.savemat('debug_grad.mat', {
    'L_py':   L_py,
    'CAK_py': CAK_py
})
print("Wrote debug_grad.mat with L_py and CAK_py.")
