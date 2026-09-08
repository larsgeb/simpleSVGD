import numpy as _numpy


def _pairwise_sq_dists(theta):
    """Pairwise squared Euclidean distances between rows of theta.

    Computed via the Gram-matrix identity ``|a-b|^2 = |a|^2 + |b|^2 - 2 a.b``
    instead of scipy's pdist/squareform, which always upcasts to float64
    regardless of input dtype -- this version preserves theta's dtype.
    """
    sq_norms = _numpy.sum(theta * theta, axis=1)
    sq_dist = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (theta @ theta.T)
    # Floating-point cancellation can push near-zero (e.g. diagonal) entries
    # slightly negative.
    _numpy.maximum(sq_dist, 0, out=sq_dist)
    return sq_dist


def rbf_kernel(theta, h=-1):
    """Radial basis function kernel."""
    pairwise_dists = _pairwise_sq_dists(theta)
    if h < 0:  # if h < 0, using median trick
        h = _numpy.median(pairwise_dists)
        h = _numpy.sqrt(0.5 * h / _numpy.log(theta.shape[0] + 1))
        # np.log(int) always returns float64; without this cast, dividing
        # the (possibly float32) median by it would silently upcast h and
        # everything computed from it.
        h = theta.dtype.type(h)

    # Guard against zero bandwidth (all particles identical)
    if h < 1e-30:
        n = theta.shape[0]
        return _numpy.ones((n, n), dtype=theta.dtype), _numpy.zeros_like(theta)

    # compute the rbf kernel
    Kxy = _numpy.exp(-pairwise_dists / h ** 2 / 2)

    dxkxy = -_numpy.matmul(Kxy, theta)
    sumkxy = _numpy.sum(Kxy, axis=1)
    for i in range(theta.shape[1]):
        dxkxy[:, i] = dxkxy[:, i] + _numpy.multiply(theta[:, i], sumkxy)
    dxkxy = dxkxy / (h ** 2)
    return (Kxy, dxkxy)


def rbf_kernel_normalized(theta, h=-1):
    """RBF kernel with per-dimension normalization for high-dimensional spaces.

    In high dimensions (d >> 1), the standard median heuristic produces
    bandwidth h^2 ~ O(d), which causes the repulsive gradient per dimension
    to scale as O(1/d) while the attractive gradient stays O(1).  This kills
    particle diversity as d grows.

    This variant normalizes each dimension to unit variance before computing
    pairwise distances and the bandwidth, then maps the kernel gradient back
    to the original space.  The effective bandwidth is dimension-independent,
    preserving repulsion in spaces with thousands of dimensions (e.g. FWI
    parameter vectors).
    """
    n_particles, n_dims = theta.shape

    # Per-dimension normalization
    std = _numpy.std(theta, axis=0)
    # Avoid division by zero for constant dimensions
    std = _numpy.where(std < 1e-30, 1.0, std)
    theta_n = theta / std  # normalized particles

    # Compute kernel in normalized space
    pairwise_dists = _pairwise_sq_dists(theta_n)
    if h < 0:
        h = _numpy.median(pairwise_dists)
        h = _numpy.sqrt(0.5 * h / _numpy.log(n_particles + 1))
        h = theta.dtype.type(h)

    if h < 1e-30:
        return _numpy.ones((n_particles, n_particles), dtype=theta.dtype), _numpy.zeros_like(theta)

    Kxy = _numpy.exp(-pairwise_dists / h ** 2 / 2)

    # Kernel gradient in normalized space
    dxkxy_n = -_numpy.matmul(Kxy, theta_n)
    sumkxy = _numpy.sum(Kxy, axis=1)
    for i in range(n_dims):
        dxkxy_n[:, i] = dxkxy_n[:, i] + _numpy.multiply(theta_n[:, i], sumkxy)
    dxkxy_n = dxkxy_n / (h ** 2)

    # Map gradient back to original space: d/dx_k = (1/std_k) * d/dx_n_k
    dxkxy = dxkxy_n / std

    return (Kxy, dxkxy)
