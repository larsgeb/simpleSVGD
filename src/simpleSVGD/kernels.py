"""RBF kernels (standard and per-dimension normalized) used by update()."""

import numpy as _numpy
import numpy.typing as _npt

from ._typing import FloatDType

# Below this, particles are treated as coincident/constant: further division
# by bandwidth or per-dimension std risks blowing up to inf/nan for no
# numerical reason.
_DEGENERATE_SCALE_THRESHOLD = 1e-30


def _pairwise_sq_dists(theta: _npt.NDArray[FloatDType]) -> _npt.NDArray[FloatDType]:
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


def rbf_kernel(
    theta: _npt.NDArray[FloatDType], h: float = -1
) -> tuple[_npt.NDArray[FloatDType], _npt.NDArray[FloatDType]]:
    """Radial basis function kernel."""
    pairwise_dists = _pairwise_sq_dists(theta)
    bandwidth: FloatDType | float
    if h < 0:  # if h < 0, using median trick
        bandwidth = _numpy.median(pairwise_dists)
        bandwidth = _numpy.sqrt(0.5 * bandwidth / _numpy.log(theta.shape[0] + 1))
        # np.log(int) always returns float64; without this cast, dividing
        # the (possibly float32) median by it would silently upcast bandwidth
        # and everything computed from it.
        bandwidth = theta.dtype.type(bandwidth)
    else:
        bandwidth = h

    # Guard against zero bandwidth (all particles identical)
    if bandwidth < _DEGENERATE_SCALE_THRESHOLD:
        n = theta.shape[0]
        return _numpy.ones((n, n), dtype=theta.dtype), _numpy.zeros_like(theta)

    # compute the rbf kernel
    Kxy = _numpy.exp(-pairwise_dists / bandwidth ** 2 / 2)

    dxkxy = -_numpy.matmul(Kxy, theta)
    sumkxy = _numpy.sum(Kxy, axis=1)
    for i in range(theta.shape[1]):
        dxkxy[:, i] = dxkxy[:, i] + _numpy.multiply(theta[:, i], sumkxy)
    dxkxy = dxkxy / (bandwidth ** 2)
    return (Kxy, dxkxy)


def rbf_kernel_normalized(
    theta: _npt.NDArray[FloatDType], h: float = -1
) -> tuple[_npt.NDArray[FloatDType], _npt.NDArray[FloatDType]]:
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
    std = _numpy.where(std < _DEGENERATE_SCALE_THRESHOLD, 1.0, std)
    theta_n = theta / std  # normalized particles

    # Compute kernel in normalized space
    pairwise_dists = _pairwise_sq_dists(theta_n)
    bandwidth: FloatDType | float
    if h < 0:
        bandwidth = _numpy.median(pairwise_dists)
        bandwidth = _numpy.sqrt(0.5 * bandwidth / _numpy.log(n_particles + 1))
        bandwidth = theta.dtype.type(bandwidth)
    else:
        bandwidth = h

    if bandwidth < _DEGENERATE_SCALE_THRESHOLD:
        return _numpy.ones((n_particles, n_particles), dtype=theta.dtype), _numpy.zeros_like(theta)

    Kxy = _numpy.exp(-pairwise_dists / bandwidth ** 2 / 2)

    # Kernel gradient in normalized space
    dxkxy_n = -_numpy.matmul(Kxy, theta_n)
    sumkxy = _numpy.sum(Kxy, axis=1)
    for i in range(n_dims):
        dxkxy_n[:, i] = dxkxy_n[:, i] + _numpy.multiply(theta_n[:, i], sumkxy)
    dxkxy_n = dxkxy_n / (bandwidth ** 2)

    # Map gradient back to original space: d/dx_k = (1/std_k) * d/dx_n_k
    dxkxy = dxkxy_n / std

    return (Kxy, dxkxy)
