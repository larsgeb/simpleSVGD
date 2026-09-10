"""Tests for mean-field Stein Variational Newton (SVN) preconditioning."""

import numpy as np
import pytest

import simplesvgd
from simplesvgd import SVGDConfig, SVNConfig
from simplesvgd.svn import svn_direction
from simplesvgd.update import _resolve_step_schedule


def _dense_cg_reference(matrix, b, cg_iters):
    """Independent dense CG per row of b, as a ground truth for _batched_cg."""
    n, d = b.shape
    out = np.zeros_like(b)
    for i in range(n):
        v = np.zeros(d, dtype=b.dtype)
        r = b[i].copy()
        p = r.copy()
        rs_old = r @ r
        for _ in range(cg_iters):
            ap = matrix @ p
            pap = p @ ap
            alpha = rs_old / pap
            v = v + alpha * p
            r = r - alpha * ap
            rs_new = r @ r
            beta = rs_new / rs_old
            p = r + beta * p
            rs_old = rs_new
        out[i] = v
    return out


class TestSVNDirection:
    def test_matches_dense_cg_reference(self):
        rng = np.random.default_rng(0)
        d = 6
        n = 4
        random_matrix = rng.normal(size=(d, d))
        spd_matrix = random_matrix.T @ random_matrix + np.eye(d)

        def hvp(_particles, vectors):
            return vectors @ spd_matrix.T

        particles = rng.normal(size=(n, d))
        phi = rng.normal(size=(n, d))

        got = svn_direction(hvp, particles, phi, SVNConfig(cg_iters=50, damping=0.0))
        expected = _dense_cg_reference(spd_matrix, phi, cg_iters=50)
        np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-8)

    def test_recovers_exact_newton_step_for_known_hessian(self):
        """For a target with a known, constant Hessian A, mean-field SVN's
        shared operator IS A exactly (evaluated Hessian doesn't depend on
        particle position), so solving A @ v = phi should give the true
        Newton direction A^{-1} @ phi to CG's convergence tolerance.
        """
        rng = np.random.default_rng(1)
        d = 5
        n = 8
        random_matrix = rng.normal(size=(d, d))
        spd_matrix = random_matrix.T @ random_matrix + 2 * np.eye(d)
        inv_matrix = np.linalg.inv(spd_matrix)

        def hvp(_particles, vectors):
            return vectors @ spd_matrix.T

        particles = rng.normal(size=(n, d)) * 3
        phi = rng.normal(size=(n, d))

        got = svn_direction(hvp, particles, phi, SVNConfig(cg_iters=100, damping=0.0))
        expected = phi @ inv_matrix.T
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-7)

    def test_damping_regularizes_singular_operator(self):
        """A singular (here, all-zero) Hessian-vector product would make an
        undamped CG solve divide by zero; damping keeps it well-posed and
        recovers the damped-identity solution v = phi / damping.
        """
        rng = np.random.default_rng(2)
        d = 4
        n = 3

        def zero_hvp(_particles, vectors):
            return np.zeros_like(vectors)

        particles = rng.normal(size=(n, d))
        phi = rng.normal(size=(n, d))
        damping = 0.5

        got = svn_direction(zero_hvp, particles, phi, SVNConfig(cg_iters=20, damping=damping))
        np.testing.assert_allclose(got, phi / damping, rtol=1e-6)

    def test_preserves_dtype(self):
        rng = np.random.default_rng(3)
        d, n = 4, 3

        def hvp(_particles, vectors):
            return vectors * 2.0

        particles = rng.normal(size=(n, d)).astype(np.float32)
        phi = rng.normal(size=(n, d)).astype(np.float32)

        got = svn_direction(hvp, particles, phi, SVNConfig())
        assert got.dtype == np.float32


class TestSVNIntegration:
    def test_requires_hessian_vector_product(self):
        rng = np.random.default_rng(4)
        x0 = rng.normal(size=(5, 2))

        def grad_fn(x):
            return x

        with pytest.raises(ValueError, match="hessian_vector_product"):
            simplesvgd.update(
                x0, grad_fn, SVGDConfig(n_iter=3, preconditioner="svn", disable_progressbar=True)
            )

    def test_rejects_adagrad_step_schedule(self):
        rng = np.random.default_rng(5)
        x0 = rng.normal(size=(5, 2))

        def grad_fn(x):
            return x

        def hvp(_particles, vectors):
            return vectors

        with pytest.raises(ValueError, match="adagrad"):
            simplesvgd.update(
                x0,
                grad_fn,
                SVGDConfig(
                    n_iter=3,
                    preconditioner="svn",
                    step_schedule="adagrad",
                    hessian_vector_product=hvp,
                    disable_progressbar=True,
                ),
            )

    def test_converges_on_gaussian_target(self):
        """A Gaussian target has a constant Hessian (the precision matrix),
        the easiest possible case for mean-field SVN's shared-operator
        approximation -- the particle mean should converge close to the
        target mean.
        """
        rng = np.random.default_rng(6)
        d = 5
        true_mean = rng.normal(size=d) * 3
        precision = np.eye(d) * 2.0

        def grad_fn(x):
            return (x - true_mean) @ precision.T

        def hvp(_particles, vectors):
            return vectors @ precision.T

        x0 = rng.normal(0, 1, (40, d))
        state = simplesvgd.update(
            x0,
            grad_fn,
            SVGDConfig(
                n_iter=100,
                stepsize=1.0,
                preconditioner="svn",
                hessian_vector_product=hvp,
                svn=SVNConfig(cg_iters=15, damping=1e-6),
                disable_progressbar=True,
            ),
        )
        final_mean = state.particles.mean(axis=0)
        assert np.linalg.norm(final_mean - true_mean) < 0.5

    def test_default_step_schedule_is_constant(self):
        assert _resolve_step_schedule(None, "svn") == "constant"
