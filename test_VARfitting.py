import numpy as np
from py_ssdi.model_fitting.VAR_fitting import select_order, fit_var

# X: shape (n, m, N) = (channels, timepoints, trials)
X = np.load("my_timeseries.npy")

# 1) choose model order
sel = select_order(X, momax=20, regmode="OLS")
p_opt = sel["opt_bic"]     # e.g. pick BIC

# 2) estimate VAR(p_opt)
A, V, E = fit_var(X, p_opt, return_resid=True)
print("coeff tensor shape =", A.shape)   # (n,n,p_opt)
print("resid cov          =", V)