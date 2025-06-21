import numpy as np
import scipy.io
from numpy.linalg import svd

# 1) Fix NumPy’s RNG
np.random.seed(0)

# 2) Create CAK_py = randn(3,3,2) and X = randn(3,2)
CAK_py = np.random.randn(3, 3, 2)
X      = np.random.randn(3, 2)

# 3) Orthonormalise X exactly as MATLAB’s [U,~,~]=svd(X,0); L=U(:,1:2)
U, _, _ = svd(X, full_matrices=False)
L_py    = U[:, :2]

# 4) Save these two arrays to a .mat file
scipy.io.savemat('debug_cak.mat', {
    'CAK_py': CAK_py,
    'L_py' : L_py
})
print("Wrote debug_cak.mat with CAK_py and L_py.")
