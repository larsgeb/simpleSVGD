import numpy as _numpy
from scipy.spatial.distance import pdist as _pdist, squareform as _squareform


def rbf_kernel(theta, h=-1):
    """Radial basis function kernel."""
    sq_dist = _pdist(theta)
    pairwise_dists = _squareform(sq_dist) ** 2
    if h < 0:  # if h < 0, using median trick
        h = _numpy.median(pairwise_dists)
        h = _numpy.sqrt(0.5 * h / _numpy.log(theta.shape[0] + 1))

    # Guard against zero bandwidth (all particles identical)
    if h < 1e-30:
        n = theta.shape[0]
        return _numpy.ones((n, n)), _numpy.zeros_like(theta)

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
    sq_dist = _pdist(theta_n)
    pairwise_dists = _squareform(sq_dist) ** 2
    if h < 0:
        h = _numpy.median(pairwise_dists)
        h = _numpy.sqrt(0.5 * h / _numpy.log(n_particles + 1))

    if h < 1e-30:
        return _numpy.ones((n_particles, n_particles)), _numpy.zeros_like(theta)

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
