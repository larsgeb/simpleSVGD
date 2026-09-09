"""RBF kernels (standard, per-dimension normalized, and mass-weighted) used by update()."""

from typing import Any

import numpy as np
import numpy.typing as npt

from ._typing import FloatDType, KernelFn

# Below this, particles are treated as coincident/constant: further division
# by bandwidth or per-dimension std risks blowing up to inf/nan for no
# numerical reason.
_DEGENERATE_SCALE_THRESHOLD = 1e-30


def _pairwise_sq_dists(particles: npt.NDArray[FloatDType]) -> npt.NDArray[FloatDType]:
    """Pairwise squared Euclidean distances between particle rows.

    Computed via the Gram-matrix identity ``|a-b|^2 = |a|^2 + |b|^2 - 2 a.b``
    instead of scipy's pdist/squareform, which always upcasts to float64
    regardless of input dtype -- this version preserves the input dtype.
    """
    sq_norms = np.sum(particles * particles, axis=1)
    sq_dist = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (particles @ particles.T)
    # Floating-point cancellation can push near-zero (e.g. diagonal) entries
    # slightly negative.
    np.maximum(sq_dist, 0, out=sq_dist)
    return sq_dist


def rbf_kernel(
    particles: npt.NDArray[FloatDType], h: float = -1
) -> tuple[npt.NDArray[FloatDType], npt.NDArray[FloatDType]]:
    """Radial basis function kernel."""
    pairwise_dists = _pairwise_sq_dists(particles)
    bandwidth: FloatDType | float
    if h < 0:  # if h < 0, using median trick
        bandwidth = np.median(pairwise_dists)
        bandwidth = np.sqrt(0.5 * bandwidth / np.log(particles.shape[0] + 1))
        # np.log(int) always returns float64; without this cast, dividing
        # the (possibly float32) median by it would silently upcast bandwidth
        # and everything computed from it.
        bandwidth = particles.dtype.type(bandwidth)
    else:
        bandwidth = h

    # Guard against zero bandwidth (all particles identical)
    if bandwidth < _DEGENERATE_SCALE_THRESHOLD:
        n = particles.shape[0]
        return np.ones((n, n), dtype=particles.dtype), np.zeros_like(particles)

    # compute the rbf kernel
    kernel_matrix = np.exp(-pairwise_dists / bandwidth ** 2 / 2)

    kernel_grad = -np.matmul(kernel_matrix, particles)
    kernel_row_sums = np.sum(kernel_matrix, axis=1)
    for i in range(particles.shape[1]):
        kernel_grad[:, i] = kernel_grad[:, i] + np.multiply(particles[:, i], kernel_row_sums)
    kernel_grad = kernel_grad / (bandwidth ** 2)
    return (kernel_matrix, kernel_grad)


def rbf_kernel_normalized(
    particles: npt.NDArray[FloatDType], h: float = -1
) -> tuple[npt.NDArray[FloatDType], npt.NDArray[FloatDType]]:
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
    n_particles, n_dims = particles.shape

    # Per-dimension normalization
    std = np.std(particles, axis=0)
    # Avoid division by zero for constant dimensions
    std = np.where(std < _DEGENERATE_SCALE_THRESHOLD, 1.0, std)
    particles_normalized = particles / std

    # Compute kernel in normalized space
    pairwise_dists = _pairwise_sq_dists(particles_normalized)
    bandwidth: FloatDType | float
    if h < 0:
        bandwidth = np.median(pairwise_dists)
        bandwidth = np.sqrt(0.5 * bandwidth / np.log(n_particles + 1))
        bandwidth = particles.dtype.type(bandwidth)
    else:
        bandwidth = h

    if bandwidth < _DEGENERATE_SCALE_THRESHOLD:
        return np.ones((n_particles, n_particles), dtype=particles.dtype), np.zeros_like(particles)

    kernel_matrix = np.exp(-pairwise_dists / bandwidth ** 2 / 2)

    # Kernel gradient in normalized space
    kernel_grad_normalized = -np.matmul(kernel_matrix, particles_normalized)
    kernel_row_sums = np.sum(kernel_matrix, axis=1)
    for i in range(n_dims):
        kernel_grad_normalized[:, i] = kernel_grad_normalized[:, i] + np.multiply(
            particles_normalized[:, i], kernel_row_sums
        )
    kernel_grad_normalized = kernel_grad_normalized / (bandwidth ** 2)

    # Map gradient back to original space: d/dx_k = (1/std_k) * d/dx_n_k
    kernel_grad = kernel_grad_normalized / std

    return (kernel_matrix, kernel_grad)


def make_mass_weighted_kernel(weights: npt.NDArray[np.floating[Any]]) -> "KernelFn[FloatDType]":
    """Build a discretization-invariant RBF kernel weighted by ``weights``.

    For particles representing a field discretized on a mesh or grid (e.g. a
    velocity/density model in FWI), the plain Euclidean RBF kernel implicitly
    treats every nodal degree of freedom as equally important -- refining the
    mesh adds more (correlated) coordinates to the same distance computation,
    changing the effective repulsion per physical degree of freedom purely as
    an artifact of resolution.

    This kernel instead measures distance in the mass-weighted inner product
    ``<a, b>_M = sum_i weights[i] * a[i] * b[i]``, i.e. ``h(x, y)^2 = (x-y)^T
    diag(weights) (x-y)``. Passing cell volumes (or quadrature weights) as
    ``weights`` makes this inner product a Riemann-sum approximation of the
    continuum L2 inner product, so it (and the resulting SVGD dynamics)
    converges to a fixed, resolution-independent quantity as the mesh is
    refined, rather than growing with the number of nodes. ``weights=1``
    everywhere recovers the plain :func:`rbf_kernel`.

    Parameters
    ----------
    weights : np.ndarray
        Per-dimension weights (e.g. mesh cell volumes), shape ``(n_dims,)``.
        Must be non-negative.

    Returns
    -------
    callable
        A kernel function usable as ``SVGDConfig(kernel=...)``.

    """
    weights = np.asarray(weights)

    def mass_weighted_kernel(
        particles: npt.NDArray[FloatDType], h: float = -1
    ) -> tuple[npt.NDArray[FloatDType], npt.NDArray[FloatDType]]:
        w = weights.astype(particles.dtype)
        sqrt_w = np.sqrt(w)
        particles_scaled = particles * sqrt_w
        pairwise_dists = _pairwise_sq_dists(particles_scaled)
        bandwidth: FloatDType | float
        if h < 0:
            bandwidth = np.median(pairwise_dists)
            bandwidth = np.sqrt(0.5 * bandwidth / np.log(particles.shape[0] + 1))
            bandwidth = particles.dtype.type(bandwidth)
        else:
            bandwidth = h

        if bandwidth < _DEGENERATE_SCALE_THRESHOLD:
            n = particles.shape[0]
            return np.ones((n, n), dtype=particles.dtype), np.zeros_like(particles)

        kernel_matrix = np.exp(-pairwise_dists / bandwidth ** 2 / 2)

        # Same combination as rbf_kernel's un-weighted gradient, then mapped
        # through M=diag(weights): d/dx_i k(x_i,x_j) picks up a factor of
        # weights (the M-quadratic form's gradient is M(x-y), not (x-y)).
        kernel_grad = -np.matmul(kernel_matrix, particles)
        kernel_row_sums = np.sum(kernel_matrix, axis=1)
        for i in range(particles.shape[1]):
            kernel_grad[:, i] = kernel_grad[:, i] + np.multiply(particles[:, i], kernel_row_sums)
        kernel_grad = kernel_grad * w / (bandwidth ** 2)
        return (kernel_matrix, kernel_grad)

    return mass_weighted_kernel
