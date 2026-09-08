"""Input-validation and error-path tests: the ValueErrors update() raises on bad config."""

import numpy as np
import pytest

import simplesvgd
from simplesvgd import SigmaConfig, SVGDConfig
from simplesvgd.kernels import rbf_kernel


def _identity_grad(x):
    return x


class TestNoneInputs:
    def test_none_x0_raises(self):
        with pytest.raises(ValueError, match="cannot be None"):
            simplesvgd.update(None, _identity_grad)  # ty: ignore[invalid-argument-type]

    def test_none_gradient_fn_raises(self):
        rng = np.random.default_rng(0)
        x0 = rng.normal(size=(5, 2))
        with pytest.raises(ValueError, match="cannot be None"):
            simplesvgd.update(x0, None)  # ty: ignore[invalid-argument-type]


class TestUnknownKernel:
    def test_unknown_kernel_string_raises(self):
        rng = np.random.default_rng(0)
        x0 = rng.normal(size=(5, 2))
        with pytest.raises(ValueError, match="Unknown kernel"):
            simplesvgd.update(
                x0,
                _identity_grad,
                SVGDConfig(n_iter=1, kernel="not-a-kernel", disable_progressbar=True),
            )


class TestUnknownStepSchedule:
    def test_unknown_step_schedule_raises(self):
        rng = np.random.default_rng(0)
        x0 = rng.normal(size=(5, 2))
        # n_iter=2: the offending branch is only reached while computing a
        # displacement, which the last iteration skips.
        with pytest.raises(ValueError, match="Unknown step_schedule"):
            simplesvgd.update(
                x0,
                _identity_grad,
                SVGDConfig(n_iter=2, step_schedule="not-a-schedule", disable_progressbar=True),
            )


class TestSigmaRequirements:
    def test_estimate_sigma_without_n_data_samples_raises(self):
        rng = np.random.default_rng(0)
        x0 = rng.normal(size=(5, 2))
        with pytest.raises(ValueError, match="n_data_samples is required"):
            simplesvgd.update(
                x0,
                _identity_grad,
                SVGDConfig(
                    n_iter=1,
                    sigma=SigmaConfig(value=1.0, estimate=True),
                    disable_progressbar=True,
                ),
            )

    def test_estimate_sigma_without_data_sigma_raises(self):
        rng = np.random.default_rng(0)
        x0 = rng.normal(size=(5, 2))
        with pytest.raises(ValueError, match="data_sigma is required"):
            simplesvgd.update(
                x0,
                _identity_grad,
                SVGDConfig(
                    n_iter=1,
                    sigma=SigmaConfig(estimate=True, n_data_samples=1),
                    disable_progressbar=True,
                ),
            )

    def test_explicit_prior_beta_overrides_computed_default(self):
        """An explicit sigma.prior_beta should be used as-is, not recomputed."""
        rng = np.random.default_rng(0)
        x0 = rng.normal(size=(20, 1))

        def grad_fn(x):
            misfits = np.sum(x**2, axis=1)
            return x, misfits

        def run(prior_beta):
            return simplesvgd.update(
                x0,
                grad_fn,
                SVGDConfig(
                    n_iter=1,
                    stepsize=0.1,
                    sigma=SigmaConfig(
                        value=1.0, estimate=True, n_data_samples=1, prior_beta=prior_beta
                    ),
                    disable_progressbar=True,
                ),
            )

        state_default = run(None)
        state_explicit = run(1000.0)
        assert state_explicit.sigma_history[0] != state_default.sigma_history[0]


class TestConfigDefaults:
    def test_omitting_config_uses_svgdconfig_defaults(self):
        rng = np.random.default_rng(0)
        x0 = rng.normal(size=(5, 2))
        # Called with no config at all -- exercises update()'s `config is
        # None -> SVGDConfig()` fallback, not a caller-constructed default.
        state = simplesvgd.update(x0, _identity_grad)
        assert state.particles.shape == (5, 2)
        assert state.iteration == SVGDConfig().n_iter


class TestCustomKernel:
    def test_custom_kernel_callable_is_used(self):
        calls = []

        def custom_kernel(particles, bandwidth):
            calls.append(particles.shape)
            return rbf_kernel(particles, bandwidth)

        rng = np.random.default_rng(0)
        x0 = rng.normal(size=(10, 2))
        state = simplesvgd.update(
            x0,
            _identity_grad,
            SVGDConfig(n_iter=5, stepsize=0.1, kernel=custom_kernel, disable_progressbar=True),
        )
        assert state.particles.shape == (10, 2)
        # kernel_fn is called once per iteration except the last (which only
        # records state, without computing a further displacement).
        assert len(calls) == 4


class TestResumeSwitchesToLBFGS:
    def test_resuming_a_non_lbfgs_run_with_lbfgs_now_enabled_initializes_fresh_state(self):
        """resume_from without lbfgs_states + preconditioner="lbfgs" now: build fresh states."""
        rng = np.random.default_rng(0)
        x0 = rng.normal(size=(10, 2))

        state1 = simplesvgd.update(
            x0, _identity_grad, SVGDConfig(n_iter=5, stepsize=0.1, disable_progressbar=True)
        )
        assert state1.lbfgs_states is None

        state2 = simplesvgd.update(
            x0,
            _identity_grad,
            SVGDConfig(
                n_iter=5,
                stepsize=0.1,
                preconditioner="lbfgs",
                resume_from=state1,
                disable_progressbar=True,
            ),
        )
        assert state2.lbfgs_states is not None
        assert len(state2.lbfgs_states) == 10


class TestKeyboardInterrupt:
    def test_callback_interrupt_returns_gracefully(self):
        """A KeyboardInterrupt raised mid-run (e.g. from a callback) shouldn't propagate."""
        rng = np.random.default_rng(0)
        x0 = rng.normal(size=(10, 2))

        def callback(iteration, _state):
            if iteration == 3:
                raise KeyboardInterrupt

        state = simplesvgd.update(
            x0,
            _identity_grad,
            SVGDConfig(n_iter=100, stepsize=0.1, callback=callback, disable_progressbar=True),
        )
        assert state.particles.shape == (10, 2)
