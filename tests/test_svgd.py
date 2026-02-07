"""Distribution recovery tests for SVGD."""

import numpy as np
import pytest

import simpleSVGD


class Test1DGaussian:
    """SVGD should recover N(3, 4) (mean=3, std=2)."""

    def setup_method(self):
        self.true_mean = 3.0
        self.true_var = 4.0

        def grad_fn(x):
            return (x - self.true_mean) / self.true_var

        self.grad_fn = grad_fn

    def test_mean_and_std(self):
        np.random.seed(42)
        x0 = np.random.normal(0, 3, (200, 1))
        state = simpleSVGD.update(
            x0, self.grad_fn, n_iter=500, stepsize=0.5, disable_progressbar=True
        )
        p = state.particles
        assert abs(np.mean(p) - self.true_mean) < 0.4
        assert abs(np.std(p) - np.sqrt(self.true_var)) < 0.5


class Test2DIsotropicGaussian:
    """SVGD should recover N([1,2], I)."""

    def setup_method(self):
        self.true_mean = np.array([1.0, 2.0])

        def grad_fn(x):
            return x - self.true_mean

        self.grad_fn = grad_fn

    def test_mean_and_cov(self):
        np.random.seed(42)
        x0 = np.random.normal(0, 3, (300, 2))
        state = simpleSVGD.update(
            x0, self.grad_fn, n_iter=800, stepsize=0.3, disable_progressbar=True
        )
        p = state.particles
        sample_mean = np.mean(p, axis=0)
        sample_cov = np.cov(p.T)

        np.testing.assert_allclose(sample_mean, self.true_mean, atol=0.4)
        np.testing.assert_allclose(np.diag(sample_cov), [1.0, 1.0], atol=0.4)


class Test2DAnisotropicGaussian:
    """SVGD should recover N([0,0], [[4,1],[1,1]])."""

    def setup_method(self):
        self.true_mean = np.array([0.0, 0.0])
        self.true_cov = np.array([[4.0, 1.0], [1.0, 1.0]])
        self.cov_inv = np.linalg.inv(self.true_cov)

        def grad_fn(x):
            return (x - self.true_mean) @ self.cov_inv

        self.grad_fn = grad_fn

    def test_mean_and_cov(self):
        np.random.seed(42)
        x0 = np.random.normal(0, 3, (500, 2))
        state = simpleSVGD.update(
            x0, self.grad_fn, n_iter=1500, stepsize=0.3, disable_progressbar=True
        )
        p = state.particles
        sample_mean = np.mean(p, axis=0)
        sample_cov = np.cov(p.T)

        np.testing.assert_allclose(sample_mean, self.true_mean, atol=0.4)
        np.testing.assert_allclose(sample_cov, self.true_cov, atol=0.8)


class TestBananaDistribution:
    """SVGD on banana-shaped posterior: p(x1) = N(0, s1^2), p(x2|x1) = N(x1^2, s2^2)."""

    def setup_method(self):
        self.s1 = 2.0
        self.s2 = 0.5

        def grad_fn(x):
            x1, x2 = x[:, 0:1], x[:, 1:2]
            # grad -log p = grad( x1^2/(2*s1^2) + (x2 - x1^2)^2/(2*s2^2) )
            dx1 = x1 / self.s1**2 + 2 * x1 * (x1**2 - x2) / self.s2**2
            dx2 = (x2 - x1**2) / self.s2**2
            return np.hstack([dx1, dx2])

        self.grad_fn = grad_fn

    def test_marginal_x1(self):
        np.random.seed(42)
        x0 = np.random.normal(0, 2, (500, 2))
        state = simpleSVGD.update(
            x0, self.grad_fn, n_iter=2000, stepsize=0.1, disable_progressbar=True
        )
        p = state.particles
        # x1 marginal should be N(0, s1^2=4)
        assert abs(np.mean(p[:, 0])) < 0.5
        assert abs(np.std(p[:, 0]) - self.s1) < 0.8

    def test_conditional_structure(self):
        np.random.seed(42)
        x0 = np.random.normal(0, 2, (500, 2))
        state = simpleSVGD.update(
            x0, self.grad_fn, n_iter=2000, stepsize=0.1, disable_progressbar=True
        )
        p = state.particles
        # x2 - x1^2 should have mean ~0 and std ~s2
        residual = p[:, 1] - p[:, 0] ** 2
        assert abs(np.mean(residual)) < 0.5
        assert abs(np.std(residual) - self.s2) < 0.5


class TestGaussianMixture:
    """Bimodal: 0.5*N([-3,0], I) + 0.5*N([3,0], I)."""

    def setup_method(self):
        self.mu1 = np.array([-3.0, 0.0])
        self.mu2 = np.array([3.0, 0.0])

        def grad_fn(x):
            # grad of -log(0.5*N1 + 0.5*N2) via log-sum-exp
            d1 = x - self.mu1
            d2 = x - self.mu2
            log_p1 = -0.5 * np.sum(d1**2, axis=1, keepdims=True)
            log_p2 = -0.5 * np.sum(d2**2, axis=1, keepdims=True)
            # Softmax weights
            max_log = np.maximum(log_p1, log_p2)
            w1 = np.exp(log_p1 - max_log)
            w2 = np.exp(log_p2 - max_log)
            w_sum = w1 + w2
            w1 /= w_sum
            w2 /= w_sum
            # Gradient of -log p = weighted sum of individual gradients
            return w1 * d1 + w2 * d2

        self.grad_fn = grad_fn

    def test_bimodality(self):
        np.random.seed(42)
        x0 = np.random.normal(0, 4, (500, 2))
        state = simpleSVGD.update(
            x0, self.grad_fn, n_iter=1500, stepsize=0.3, disable_progressbar=True
        )
        p = state.particles
        # Check both modes are populated
        left = np.sum(p[:, 0] < 0)
        right = np.sum(p[:, 0] >= 0)
        frac_left = left / len(p)
        assert 0.2 < frac_left < 0.8, f"Mode balance: {frac_left:.2f}"

    def test_mode_locations(self):
        np.random.seed(42)
        x0 = np.random.normal(0, 4, (500, 2))
        state = simpleSVGD.update(
            x0, self.grad_fn, n_iter=1500, stepsize=0.3, disable_progressbar=True
        )
        p = state.particles
        left_particles = p[p[:, 0] < 0]
        right_particles = p[p[:, 0] >= 0]
        if len(left_particles) > 10:
            assert abs(np.mean(left_particles[:, 0]) - (-3.0)) < 1.0
        if len(right_particles) > 10:
            assert abs(np.mean(right_particles[:, 0]) - 3.0) < 1.0
