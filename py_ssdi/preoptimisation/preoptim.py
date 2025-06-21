# preoptim.py
import time
import numpy as np
from numpy.linalg import svd, eig
from scipy.linalg import cholesky, inv, subspace_angles


def Lcluster(gmat, ctol, dvals, gpterm=None, gpscale=None, gpfsize=None, gpplot=False):
    """
    Cluster pre-optimized subspaces based on pairwise distance matrix gmat.
    Select cluster medoids based on smallest objective values in dvals.

    Parameters
    ----------
    gmat : ndarray, shape (N, N)
        Pairwise distance matrix between N subspaces.
    ctol : float
        Clustering tolerance: maximum distance within a cluster.
    dvals : ndarray, shape (N,)
        Objective values for each subspace (used to pick medoid).
    gpterm, gpscale, gpfsize, gpplot : unused placeholders for plotting.

    Returns
    -------
    uidx : ndarray, shape (N,)
        For each index i, uidx[i] = index of cluster medoid.
    cluster_labels : ndarray, shape (N,)
        Label of cluster for each i (optional, can return None).
    nrunso : int
        Number of unique clusters.
    """
    N = gmat.shape[0]
    uidx = np.arange(N)
    cluster_labels = -np.ones(N, dtype=int)
    
    # Simple greedy clustering: assign each to a cluster where max distance to cluster medoid <= ctol
    medoids = []
    labels = np.full(N, -1, dtype=int)
    cluster_id = 0
    for i in range(N):
        if labels[i] != -1:
            continue
        # Start new cluster with i as medoid initially
        current_medoid = i
        labels[i] = cluster_id
        medoids.append(current_medoid)
        # assign j to this cluster if distance to medoid <= ctol
        for j in range(i+1, N):
            if labels[j] == -1 and gmat[current_medoid, j] <= ctol:
                labels[j] = cluster_id
        cluster_id += 1
    medoids = np.array(medoids)
    nrunso = len(medoids)
    # For each cluster, pick true medoid as point with minimum dvals
    for cid in range(nrunso):
        members = np.where(labels == cid)[0]
        best = members[np.argmin(dvals[members])]
        for m in members:
            uidx[m] = best
    return uidx, labels, nrunso


def cak2ddx(L, CAK):
    """
    Calculate proxy dynamical dependence D for subspace basis L and CAK tensor.

    Parameters
    ----------
    L   : ndarray, shape (n, m)
          Orthonormal subspace basis (columns orthonormal).
    CAK : ndarray, shape (n, n, r)
          Sequence of n×n matrices (CA^{k-1}K) for k = 1..r.

    Returns
    -------
    D : float
        Proxy dynamical dependence: sum_k [‖L' * CAK[:,:,k]‖_F^2 - ‖(L' * CAK[:,:,k] * L)‖_F^2].
    """
    n, m = L.shape
    _, _, r = CAK.shape
    D = 0.0
    Lt = L.T  # shape (m, n)

    for k in range(r):
        Q = CAK[:, :, k]            # shape (n, n)
        LQ = Lt @ Q                 # shape (m, n)
        LQL = LQ @ L                # shape (m, m)
        D1k = np.sum(LQ**2)         # Frobenius norm squared of L'Q
        D2k = np.sum(LQL**2)        # Frobenius norm squared of L'Q L
        D += (D1k - D2k)
    return D


def cak2ddxgrad(L, CAK):
    """
    Calculate gradient G and its magnitude mG of proxy dynamical dependence for L and CAK.

    Parameters
    ----------
    L   : ndarray, shape (n, m)
          Orthonormal subspace basis (columns orthonormal).
    CAK : ndarray, shape (n, n, r)
          Sequence of n×n matrices (CA^{k-1}K) for k = 1..r.

    Returns
    -------
    G  : ndarray, shape (n, m)
         Gradient matrix (on Stiefel manifold) of proxy DI w.r.t. L.
    mG : float
         Magnitude (Frobenius norm) of G.
    """
    n, m = L.shape
    _, _, r = CAK.shape

    # P = L L'
    P = L @ L.T  # shape (n, n)

    # Compute intermediate g matrix (n×n)
    g = np.zeros((n, n))
    for k in range(r):
        Q = CAK[:, :, k]     # shape (n, n)
        QT = Q.T             # shape (n, n)
        # g += Q Q' - Q' P Q - Q P Q'
        g += (Q @ QT) - (QT @ P @ Q) - (Q @ P @ QT)

    # Compute raw gradient G = 2 g L
    G = 2.0 * (g @ L)        # shape (n, m)

    # Project onto tangent space of Stiefel (Grassmannian):
    # G ← G - P G
    G = G - (P @ G)

    # Magnitude of gradient (Frobenius norm)
    mG = np.sqrt(np.sum(G**2))
    return G, mG




def orthonormalise(X: np.ndarray, return_complement: bool = False):
    """
    Orthonormalise the columns of X.

    Parameters
    ----------
    X : ndarray (n, m)
        Input matrix.
    return_complement : bool, default=False
        If True, also return an orthonormal basis for the orthogonal complement.

    Returns
    -------
    L : ndarray (n, r)
        Orthonormal basis for span(X) with L.conj().T @ L = I_r.
        Here r = min(rank(X), m).
    M : ndarray (n, n-r), optional
        Orthonormal basis for the orthogonal complement (returned
        only if `return_complement=True`).
    """
    # full_matrices=False gives the economy-size U of shape (n, r)
    U, _, _ = np.linalg.svd(X, full_matrices=return_complement)

    m = X.shape[1]        # original column count
    L = U[:, :m]          # matches MATLAB's L = U(:,1:m)

    if not return_complement:
        return L
    else:
        n = X.shape[0]
        M = U[:, m:n]     # remaining columns form complement
        return L, M



# ——— Updated opt_gd_ddx_mruns with per‐run printing ———
def opt_gd_ddx_mruns(
    CAK,
    L0,
    niters=10000,
    gdsig0=1.0,
    gdls=1.0,
    gdtol=1e-8,
    hist=False,
    parallel=False,
    variant=1
):
    """
    Runs proxy‐DI gradient‐descent (pre‐optimisation) over multiple restarts,
    printing a formatted status line for each run.

    Parameters
    ----------
    CAK      : ndarray, shape (n, n, r)
        The decorrelated, normalized SS parameters from iss_to_CAK().
    L0       : ndarray, shape (n, m, nruns)
        Initial orthonormal bases for each of the nruns restarts.
    niters   : int
        Maximum GD iterations.
    gdsig0   : float
        Initial step size.
    gdls     : float or tuple of two floats
        Step‐size multipliers: if scalar f, then nfac = 1/f. If two‐vector [f, nfac].
    gdtol    : float or array‐like of length 2 or 3
        Tolerances [stol, dtol, gtol].
    hist     : bool
        If True, record history [dd, sigma, grad_norm] per iteration.
    parallel : bool
        Ignored (no parallelism in this Python port).
    variant  : int (1 or 2)
        Selects MATLAB's opt_gd1_ddx (variant=1) or opt_gd2_ddx (variant=2).

    Returns
    -------
    dopt   : ndarray, shape (nruns,)
    Lopt   : ndarray, shape (n, m, nruns)
    conv   : ndarray, shape (nruns,)
    iopt   : ndarray, shape (nruns,)
    sopt   : ndarray, shape (nruns,)
    cput   : ndarray, shape (nruns,)
    ohist  : list of length nruns (or empty list if hist=False)
    """

    n, m, nruns = L0.shape
    dopt = np.zeros(nruns)
    Lopt = np.zeros((n, m, nruns))
    conv = np.zeros(nruns, dtype=int)
    iopt = np.zeros(nruns, dtype=int)
    sopt = np.zeros(nruns)
    cput = np.zeros(nruns)
    ohist = [None] * nruns if hist else []

    # Parse step-size multipliers
    if np.isscalar(gdls):
        ifac = gdls
        nfac = 1.0 / gdls
    else:
        ifac = gdls[0]
        nfac = gdls[1]

    # Parse tolerances
    if np.isscalar(gdtol):
        stol = gdtol
        dtol = gdtol
        gtol = gdtol / 10.0
    elif len(gdtol) == 2:
        stol = gdtol[0]
        dtol = gdtol[1]
        gtol = dtol / 10.0
    else:
        stol, dtol, gtol = gdtol

    # Main loop over each restart (serial)
    for k in range(nruns):
        start_cpu = time.process_time()

        L = L0[:, :, k].copy()
        sigma = gdsig0
        d_current = cak2ddx(L, CAK)
        G, gmag = cak2ddxgrad(L, CAK)

        if hist:
            dhist = np.zeros((niters, 3))
            dhist[0, :] = [d_current, sigma, gmag]
        else:
            dhist = None

        converged = 0

        for iters in range(2, niters + 1):
            if variant == 1:
                # Variant 1 (MATLAB opt_gd1_ddx)
                Ltry = orthonormalise(L - sigma * (G / (gmag + 1e-16)))
                dtry = cak2ddx(Ltry, CAK)
                if dtry < d_current:
                    L = Ltry
                    G, gmag = cak2ddxgrad(L, CAK)
                    d_current = dtry
                    sigma *= ifac
                else:
                    sigma *= nfac

            else:
                # Variant 2 (MATLAB opt_gd2_ddx)
                L = orthonormalise(L - sigma * (G / (gmag + 1e-16)))
                G, gmag = cak2ddxgrad(L, CAK)
                dnew = cak2ddx(L, CAK)
                if dnew < d_current:
                    d_current = dnew
                    sigma *= ifac
                else:
                    sigma *= nfac

            if hist:
                dhist[iters - 1, :] = [d_current, sigma, gmag]

            # Check convergence
            if sigma < stol:
                converged = 1
                break
            elif d_current < dtol:
                converged = 2
                break
            elif gmag < gtol:
                converged = 3
                break

        if hist and dhist is not None:
            dhist = dhist[:iters, :]

        elapsed_cpu = time.process_time() - start_cpu

        # Save results
        dopt[k] = d_current
        Lopt[:, :, k] = L
        conv[k] = converged
        iopt[k] = iters
        sopt[k] = sigma
        cput[k] = elapsed_cpu
        if hist:
            ohist[k] = dhist

        # Print a nice progress line:
        # e.g. "GD/ES(2) serial run  76 of 100 : dd=7.7392e-02 : sig=7.4506e-09 : converged(1) in  156 iterations : CPU secs = 0.01"
        status = (
            f"GD/ES({variant}) serial run {k+1:3d} of {nruns:3d} : "
            f"dd = {d_current: .4e} : "
            f"sig = {sigma: .4e} : "
            f"{'converged('+str(converged)+')' if converged>0 else 'unconverged'} "
            f"in {iters:4d} iter : CPU secs = {elapsed_cpu:6.2f}"
        )
        print(status)

    # Sort by ascending dopt
    sidx = np.argsort(dopt)
    dopt = dopt[sidx]
    Lopt = Lopt[:, :, sidx]
    conv = conv[sidx]
    iopt = iopt[sidx]
    sopt = sopt[sidx]
    cput = cput[sidx]
    if hist:
        ohist = [ohist[i] for i in sidx]

    return dopt, Lopt, conv, iopt, sopt, cput, ohist


def itransform_subspace(L_norm, V):
    """
    Undo the earlier normalisation for each slice of L_norm:
      For each j, L_undec(:,:,j) = C @ L_norm(:,:,j), where C = chol(V)
    Inputs:
      L_norm : ndarray, shape (n, m, N)
      V      : ndarray, shape (n, n) (original correlated residual-covariance)
    Returns:
      L_undec : ndarray, shape (n, m, N)
    """
    n, m, N = L_norm.shape
    C = cholesky(V, lower=True)
    L_undec = np.zeros_like(L_norm)
    for j in range(N):
        L_undec[:, :, j] = C @ L_norm[:, :, j]
    return L_undec


def gmetrics(Lset):
    """
    Compute pairwise subspace distances for Lset (shape (n, m, N)).
    Returns a (N, N) matrix g, where g[i, j] is the normalized Grassmann distance between subspaces i and j.
    The distances are normalized to [0,1] by dividing by the maximum possible Grassmann distance (π/2 * √m).
    """
    n, m, N = Lset.shape
    g = np.zeros((N, N))
    max_dist = (np.pi/2) * np.sqrt(m)  # Maximum possible Grassmann distance
    
    for i in range(N):
        Ui = Lset[:, :, i]
        for j in range(i+1, N):
            Uj = Lset[:, :, j]
            # Compute principal angles via SVD of Ui^T Uj
            _, s, _ = svd(Ui.T @ Uj)
            # Grassmann distance = sqrt(sum(arccos(s)^2))
            angles = np.arccos(np.clip(s, -1.0, 1.0))
            g[i, j] = np.linalg.norm(angles) / max_dist  # Normalize by maximum possible distance
            g[j, i] = g[i, j]
    return g
