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
