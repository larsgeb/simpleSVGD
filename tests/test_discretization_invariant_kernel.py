"""Tests for the mass-weighted (discretization-invariant) kernel."""

import numpy as np

import simplesvgd
from simplesvgd import SVGDConfig, make_mass_weighted_kernel
from simplesvgd.kernels import rbf_kernel


class TestMassWeightedKernelBasics:
    """Sanity checks on the kernel's shape and its relationship to plain RBF."""

    def test_uniform_weights_matches_plain_rbf(self):
        """Uniform weights are a constant rescale of distance, absorbed by the
        auto (median-heuristic) bandwidth -- so a uniform-weight mass kernel
        must reduce to exactly the plain RBF kernel."""
        rng = np.random.default_rng(0)
        particles = rng.normal(size=(15, 6))
        weights = np.full(6, 0.3)
        mass_kernel = make_mass_weighted_kernel(weights)

        k_matrix_mass, k_grad_mass = mass_kernel(particles, -1)
        k_matrix_rbf, k_grad_rbf = rbf_kernel(particles, -1)

        np.testing.assert_allclose(k_matrix_mass, k_matrix_rbf, rtol=1e-10)
        np.testing.assert_allclose(k_grad_mass, k_grad_rbf, rtol=1e-10)

    def test_usable_as_update_kernel(self):
        rng = np.random.default_rng(0)
        x0 = rng.normal(size=(10, 4))
        weights = np.array([1.0, 2.0, 0.5, 3.0])
        kfn = make_mass_weighted_kernel(weights)

        def grad_fn(particles):
            return particles

        state = simplesvgd.update(
            x0, grad_fn, SVGDConfig(n_iter=5, stepsize=0.1, kernel=kfn, disable_progressbar=True)
        )
        assert state.particles.shape == (10, 4)

    def test_degenerate_zero_bandwidth_returns_no_repulsion(self):
        particles = np.ones((5, 3))  # all particles coincident -> zero spread
        kfn = make_mass_weighted_kernel(np.array([1.0, 1.0, 1.0]))
        k_matrix, k_grad = kfn(particles, -1)
        np.testing.assert_array_equal(k_matrix, np.ones((5, 5)))
        np.testing.assert_array_equal(k_grad, np.zeros((5, 3)))


class TestDiscretizationInvarianceUnderNonUniformMesh:
    """Posterior stability under a non-uniformly refined mesh.

    Two physical regions of equal length share one domain, but are
    discretized with different node counts (``n1`` fine, ``n2`` coarse) --
    exactly the situation the issue describes: a locally-refined mesh (e.g.
    denser near receivers in FWI) shouldn't change how much physical
    variance SVGD recovers in the coarser region.

    The target is a discretized white-noise-like field: node ``i`` has
    precision ``dx_i / sigma**2``, so the *region integral* (a Riemann sum
    approximating a continuum quantity) has variance ``sigma**2 * region
    length`` independent of how many nodes discretize it -- the
    discretization-invariant statistic this test checks.

    With the plain isotropic RBF kernel, the auto (median-heuristic)
    bandwidth is calibrated to the whole particle vector's distances, which
    are dominated by whichever region has more nodes -- so the coarser
    region's much-larger per-node deviations barely move the total distance,
    and its particles collapse to near-zero spread (verified empirically:
    across 6 seeds, the coarser region's recovered integral std never
    exceeds 0.02). The mass-weighted kernel corrects this by weighting each
    node's contribution to distance by its cell width, restoring meaningful
    (if not fully converged) spread in the coarse region: never below 0.08
    in the same 6 seeds -- a 5x+ margin over the RBF floor.
    """

    SIGMA_TRUE = 1.0
    L_TOTAL = 1.0
    N1 = 30
    N2 = 20
    N_ITER = 800
    STEPSIZE = 0.5
    N_PARTICLES = 40

    def _dx(self):
        dx1 = (self.L_TOTAL / 2) / self.N1
        dx2 = (self.L_TOTAL / 2) / self.N2
        return np.concatenate([np.full(self.N1, dx1), np.full(self.N2, dx2)])

    def _run(self, kernel, seed):
        dx = self._dx()
        n_nodes = self.N1 + self.N2

        def grad_fn(particles):
            return dx * particles / self.SIGMA_TRUE**2

        rng = np.random.default_rng(seed)
        x0 = rng.normal(0, 1.0, size=(self.N_PARTICLES, n_nodes)) / np.sqrt(dx)

        kfn = make_mass_weighted_kernel(dx) if kernel == "mass" else kernel

        state = simplesvgd.update(
            x0,
            grad_fn,
            SVGDConfig(
                n_iter=self.N_ITER, stepsize=self.STEPSIZE, kernel=kfn, disable_progressbar=True
            ),
        )
        p = state.particles
        region1_integral = np.sum(p[:, : self.N1] * dx[: self.N1], axis=1)
        region2_integral = np.sum(p[:, self.N1 :] * dx[self.N1 :], axis=1)
        return min(np.std(region1_integral), np.std(region2_integral))

    def test_mass_weighted_kernel_preserves_coarse_region_variance(self):
        min_stds = [self._run("mass", seed) for seed in range(6)]
        assert all(s > 0.05 for s in min_stds), (
            f"mass-weighted kernel should keep both regions' recovered variance "
            f"well above collapse, got {min_stds}"
        )

    def test_isotropic_rbf_kernel_collapses_the_coarse_region(self):
        """Regression/contrast case: the existing isotropic kernel is *not*
        stable under a non-uniformly refined mesh -- the coarser region's
        variance collapses towards zero."""
        min_stds = [self._run("rbf", seed) for seed in range(6)]
        assert all(s < 0.03 for s in min_stds), (
            f"expected the isotropic kernel to under-repel the coarser region, "
            f"got {min_stds}"
        )
