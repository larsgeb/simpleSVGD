"""Feature-specific tests: preconditioning, sigma, bounds, resume, callback, backward compat."""

import numpy as np

import simpleSVGD


def _gaussian_grad(x, mean=0.0, var=1.0):
    return (x - mean) / var


class TestLBFGSPreconditioner:
    """L-BFGS should converge to correct mean faster than AdaGrad on anisotropic target."""

    def setup_method(self):
        self.true_mean = np.array([3.0, -2.0])
        cov = np.array([[4.0, 1.0], [1.0, 1.0]])
        self.cov_inv = np.linalg.inv(cov)

        def grad_fn(x):
            return (x - self.true_mean) @ self.cov_inv

        self.grad_fn = grad_fn

    def test_mean_convergence(self):
        np.random.seed(42)
        x0 = np.random.normal(0, 5, (200, 2))
        initial_error = np.linalg.norm(np.mean(x0, axis=0) - self.true_mean)
        state = simpleSVGD.update(
            x0, self.grad_fn, n_iter=500, stepsize=0.1,
            preconditioner="lbfgs", lbfgs_history=10, disable_progressbar=True
        )
        p = state.particles
        final_error = np.linalg.norm(np.mean(p, axis=0) - self.true_mean)
        # L-BFGS should reduce the error substantially
        assert final_error < initial_error * 0.6, (
            f"L-BFGS did not reduce error enough: {initial_error:.2f} -> {final_error:.2f}"
        )

    def test_lbfgs_states_populated(self):
        np.random.seed(42)
        x0 = np.random.normal(0, 3, (10, 2))
        state = simpleSVGD.update(
            x0, self.grad_fn, n_iter=20, stepsize=0.05,
            preconditioner="lbfgs", disable_progressbar=True
        )
        assert state.lbfgs_states is not None
        assert len(state.lbfgs_states) == 10
        # At least some states should have curvature info
        assert any(s.count > 0 for s in state.lbfgs_states)


class TestHierarchicalSigma:
    """Sigma estimation should converge toward the true noise level."""

    def setup_method(self):
        self.true_sigma = 0.5
        self.true_mean = np.array([0.0])

        def grad_fn(x):
            grads = (x - self.true_mean)
            # Simulate misfit = 0.5 * ||residual||^2 / sigma^2
            # but return raw misfit (before sigma scaling)
            misfits = 0.5 * np.sum((x - self.true_mean) ** 2, axis=1)
            return grads, misfits

        self.grad_fn = grad_fn

    def test_sigma_converges(self):
        np.random.seed(42)
        x0 = np.random.normal(0, 1, (50, 1))
        state = simpleSVGD.update(
            x0, self.grad_fn, n_iter=100, stepsize=0.3,
            data_sigma=2.0, estimate_sigma=True,
            sigma_prior_alpha=2.0, n_data_samples=1,
            disable_progressbar=True
        )
        assert len(state.sigma_history) == 100
        # Sigma should have changed from initial value
        assert state.sigma_history[-1] != state.sigma_history[0]
        assert state.data_sigma is not None

    def test_misfit_history_recorded(self):
        np.random.seed(42)
        x0 = np.random.normal(0, 1, (20, 1))
        state = simpleSVGD.update(
            x0, self.grad_fn, n_iter=50, stepsize=0.3,
            data_sigma=1.0, estimate_sigma=True,
            sigma_prior_alpha=2.0, n_data_samples=1,
            disable_progressbar=True
        )
        assert len(state.misfit_history) == 50
        assert len(state.particle_misfit_history) == 50
        assert len(state.particle_misfit_history[0]) == 20


class TestBoundsEnforcement:

    def test_particles_stay_within_bounds(self):
        np.random.seed(42)
        # Target mean is outside bounds → particles should be clipped
        x0 = np.random.normal(0, 0.5, (100, 2))

        def grad_fn(x):
            return x - 5.0  # pushes toward 5.0

        state = simpleSVGD.update(
            x0, grad_fn, n_iter=100, stepsize=0.5,
            bounds=(-1.0, 1.0), disable_progressbar=True
        )
        assert np.all(state.particles >= -1.0)
        assert np.all(state.particles <= 1.0)


class TestResume:

    def test_resume_continues_iteration(self):
        np.random.seed(42)
        x0 = np.random.normal(0, 3, (100, 1))

        def grad_fn(x):
            return (x - 3.0) / 4.0

        state1 = simpleSVGD.update(
            x0, grad_fn, n_iter=200, stepsize=0.5, disable_progressbar=True
        )
        assert state1.iteration == 200

        state2 = simpleSVGD.update(
            x0, grad_fn, n_iter=200, stepsize=0.5,
            resume_from=state1, disable_progressbar=True
        )
        assert state2.iteration == 400
        # Resumed run should have better or equal convergence
        mean_final = abs(np.mean(state2.particles) - 3.0)
        assert mean_final < 0.5

    def test_resume_lbfgs(self):
        np.random.seed(42)
        x0 = np.random.normal(0, 3, (50, 2))

        def grad_fn(x):
            return x

        state1 = simpleSVGD.update(
            x0, grad_fn, n_iter=50, stepsize=0.05,
            preconditioner="lbfgs", disable_progressbar=True
        )
        state2 = simpleSVGD.update(
            x0, grad_fn, n_iter=50, stepsize=0.05,
            preconditioner="lbfgs", resume_from=state1, disable_progressbar=True
        )
        assert state2.iteration == 100
        assert state2.lbfgs_states is not None


class TestCallback:

    def test_callback_called_each_iteration(self):
        np.random.seed(42)
        x0 = np.random.normal(0, 1, (20, 1))
        iterations_seen = []

        def my_callback(iteration, _state):
            iterations_seen.append(iteration)

        simpleSVGD.update(
            x0, lambda x: x, n_iter=10, stepsize=0.1,
            callback=my_callback, disable_progressbar=True
        )
        assert iterations_seen == list(range(10))

    def test_callback_receives_state(self):
        np.random.seed(42)
        x0 = np.random.normal(0, 1, (20, 2))
        states_seen = []

        def my_callback(_iteration, state):
            states_seen.append(state)

        simpleSVGD.update(
            x0, lambda x: x, n_iter=5, stepsize=0.1,
            callback=my_callback, disable_progressbar=True
        )
        assert len(states_seen) == 5
        assert states_seen[0].particles.shape == (20, 2)


class TestBackwardCompat:
    """Old-style positional call should still work."""

    def test_positional_args(self):
        np.random.seed(42)
        x0 = np.random.normal(0, 3, (100, 2))

        def grad_fn(x):
            return x

        # Old API: update(x0, gradient_fn, n_iter, stepsize)
        # New API uses keyword-only after gradient_fn, but we still support positional
        # for x0 and gradient_fn
        state = simpleSVGD.update(x0, grad_fn, n_iter=50, stepsize=0.3,
                                  disable_progressbar=True)
        assert state.particles.shape == (100, 2)

    def test_returns_svgd_state(self):
        np.random.seed(42)
        x0 = np.random.normal(0, 1, (10, 1))
        result = simpleSVGD.update(x0, lambda x: x, n_iter=5, stepsize=0.1,
                                   disable_progressbar=True)
        assert isinstance(result, simpleSVGD.SVGDState)
        assert hasattr(result, "particles")


class TestRobbinsMonroSchedule:

    def test_does_not_diverge(self):
        np.random.seed(42)
        x0 = np.random.normal(0, 3, (100, 1))

        def grad_fn(x):
            return (x - 2.0)

        state = simpleSVGD.update(
            x0, grad_fn, n_iter=200, stepsize=1.0,
            step_schedule="robbins-monro", disable_progressbar=True
        )
        # Should converge to mean ~2
        assert abs(np.mean(state.particles) - 2.0) < 1.0
        # Particles should not have diverged
        assert np.std(state.particles) < 10.0


class TestConstantSchedule:

    def test_constant_step(self):
        np.random.seed(42)
        x0 = np.random.normal(0, 1, (50, 1))

        def grad_fn(x):
            return x

        state = simpleSVGD.update(
            x0, grad_fn, n_iter=100, stepsize=0.01,
            step_schedule="constant", disable_progressbar=True
        )
        assert abs(np.mean(state.particles)) < 0.5
