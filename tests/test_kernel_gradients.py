"""Closed-form checks on the kernel gradients.

These pin the gradient each kernel returns to its analytic definition rather
than to whatever the implementation happens to compute. The gradient
accumulation is vectorized across dimensions, which is worth several times the
throughput at FWI-scale ``d`` but is also exactly the kind of rewrite that can
silently transpose or broadcast wrong -- these tests are the guard on that.
"""

import numpy as np
import pytest

from simplesvgd import make_mass_weighted_kernel
from simplesvgd.kernels import rbf_kernel, rbf_kernel_normalized


def _reference_rbf_grad(kernel_matrix, particles, bandwidth):
    """sum_j k(x_i, x_j) * (x_i - x_j) / h^2, built one particle at a time.

    Deliberately the slow, obvious double loop: it mirrors the definition
    rather than the implementation, so agreement is real evidence.
    """
    n = particles.shape[0]
    out = np.zeros_like(particles)
    for i in range(n):
        for j in range(n):
            out[i] += kernel_matrix[i, j] * (particles[i] - particles[j])
    return out / bandwidth**2


class TestRBFGradientClosedForm:
    @pytest.mark.parametrize("n_dims", [1, 3, 64])
    def test_matches_definition(self, n_dims):
        rng = np.random.default_rng(0)
        particles = rng.normal(size=(12, n_dims))
        bandwidth = 1.7

        kernel_matrix, kernel_grad = rbf_kernel(particles, bandwidth)
        expected = _reference_rbf_grad(kernel_matrix, particles, bandwidth)

        np.testing.assert_allclose(kernel_grad, expected, rtol=1e-10, atol=1e-12)

    def test_normalized_matches_definition_in_normalized_space(self):
        """rbf_normalized computes the gradient against per-dimension
        standardized particles, then maps it back by dividing by the same
        per-dimension scale.
        """
        rng = np.random.default_rng(1)
        particles = rng.normal(size=(12, 40)) * rng.uniform(0.5, 20.0, size=40)
        bandwidth = 2.3

        kernel_matrix, kernel_grad = rbf_kernel_normalized(particles, bandwidth)

        std = np.std(particles, axis=0)
        normalized = particles / std
        expected = _reference_rbf_grad(kernel_matrix, normalized, bandwidth) / std

        np.testing.assert_allclose(kernel_grad, expected, rtol=1e-10, atol=1e-12)

    def test_mass_weighted_matches_definition(self):
        """The mass-weighted gradient is the plain combination mapped through
        M = diag(weights), since d/dx of the M-quadratic form is M(x-y).
        """
        rng = np.random.default_rng(2)
        particles = rng.normal(size=(12, 30))
        weights = rng.uniform(0.1, 3.0, size=30)
        bandwidth = 1.1

        kernel_matrix, kernel_grad = make_mass_weighted_kernel(weights)(particles, bandwidth)
        expected = _reference_rbf_grad(kernel_matrix, particles, bandwidth) * weights

        np.testing.assert_allclose(kernel_grad, expected, rtol=1e-10, atol=1e-12)


class TestRBFGradientProperties:
    def test_gradient_sums_to_zero_over_particles(self):
        """Every pair contributes (x_i - x_j) to i and (x_j - x_i) to j, so the
        repulsive term moves particles apart without translating the ensemble.
        """
        rng = np.random.default_rng(3)
        particles = rng.normal(size=(20, 50))

        _, kernel_grad = rbf_kernel(particles, -1)

        np.testing.assert_allclose(kernel_grad.sum(axis=0), 0.0, atol=1e-10)

    def test_identical_particles_have_zero_gradient(self):
        particles = np.tile(np.arange(8.0), (5, 1))

        _, kernel_grad = rbf_kernel(particles, 1.0)

        np.testing.assert_allclose(kernel_grad, 0.0, atol=1e-12)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_preserves_dtype(self, dtype):
        rng = np.random.default_rng(4)
        particles = rng.normal(size=(10, 128)).astype(dtype)

        _, kernel_grad = rbf_kernel(particles, -1)

        assert kernel_grad.dtype == dtype
