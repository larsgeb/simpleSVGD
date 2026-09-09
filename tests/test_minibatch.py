"""Tests for minibatched (stochastic) gradient evaluation via ``minibatch_sampler``."""

import importlib

import numpy as np
import pytest

import simplesvgd
from simplesvgd import SigmaConfig, SVGDConfig


def _make_gaussian_data(rng, n_data, theta_true, sigma_true):
    return rng.normal(theta_true, sigma_true, size=n_data)


def _sum_grad_fn(data):
    """Returns raw sums (not means) of residuals/squared-residuals over the given indices.

    Mirrors the additive structure of a real multi-source FWI misfit, where
    the gradient and misfit are sums over data samples -- the shape the
    unbiased-rescale in ``update()`` is designed for.
    """

    def grad_fn(particles, batch_indices=None):
        idx = batch_indices if batch_indices is not None else np.arange(data.shape[0])
        d = data[idx]
        theta = particles[:, 0]
        resid = theta[:, None] - d[None, :]
        grads = np.sum(resid, axis=1, keepdims=True)
        misfits = np.sum(resid**2, axis=1)
        return grads, misfits

    return grad_fn


class TestMinibatchSampling:
    """Basic wiring: minibatch_sampler is called and drives gradient_fn's second argument."""

    def test_gradient_fn_receives_batch_indices(self):
        rng = np.random.default_rng(0)
        data = _make_gaussian_data(rng, n_data=50, theta_true=0.0, sigma_true=1.0)
        seen_batches = []

        def grad_fn(particles, batch_indices):
            seen_batches.append(np.asarray(batch_indices).copy())
            d = data[batch_indices]
            theta = particles[:, 0]
            resid = theta[:, None] - d[None, :]
            return np.sum(resid, axis=1, keepdims=True)

        def sampler(iteration):
            return np.arange(iteration % 5, iteration % 5 + 10)

        x0 = rng.normal(0, 1, size=(5, 1))
        simplesvgd.update(
            x0,
            grad_fn,
            SVGDConfig(
                n_iter=6,
                stepsize=0.01,
                sigma=SigmaConfig(n_data_samples=50),
                minibatch_sampler=sampler,
                disable_progressbar=True,
            ),
        )

        assert len(seen_batches) == 6
        for iteration, batch in enumerate(seen_batches):
            np.testing.assert_array_equal(batch, np.arange(iteration % 5, iteration % 5 + 10))

    def test_no_minibatch_sampler_calls_single_arg_gradient_fn(self):
        rng = np.random.default_rng(0)
        calls = []

        def grad_fn(particles):
            calls.append(particles.shape)
            return particles

        x0 = rng.normal(0, 1, size=(4, 2))
        config = SVGDConfig(n_iter=3, stepsize=0.01, disable_progressbar=True)
        simplesvgd.update(x0, grad_fn, config)
        assert len(calls) == 3

    def test_minibatch_sampler_requires_n_data_samples(self):
        rng = np.random.default_rng(0)
        x0 = rng.normal(0, 1, size=(4, 1))

        def grad_fn(particles, _batch_indices):
            return particles

        with pytest.raises(ValueError, match="n_data_samples"):
            simplesvgd.update(
                x0,
                grad_fn,
                SVGDConfig(
                    n_iter=1,
                    minibatch_sampler=lambda _iteration: np.array([0, 1]),
                    disable_progressbar=True,
                ),
            )


class TestMinibatchSigmaCalibration:
    """Sigma estimation under minibatching should stay calibrated, not just run without error.

    A full-batch run gives a reference ``data_sigma``. Minibatch runs -- which
    see only a subsample's raw (unscaled) gradient/misfit sums each
    iteration -- are rescaled by ``n_data_samples / len(batch_indices)`` so
    the hierarchical sigma posterior sees full-dataset-equivalent quantities.
    If that rescale were missing, the posterior would systematically
    underestimate sigma by roughly ``sqrt(batch_size / n_data_samples)``.
    """

    N_DATA = 200
    SIGMA_TRUE = 1.5
    BATCH_SIZE = 20
    N_ITER = 150
    STEPSIZE = 0.05

    def _run_full(self, x0, grad_fn):
        state = simplesvgd.update(
            x0.copy(),
            grad_fn,
            SVGDConfig(
                n_iter=self.N_ITER,
                stepsize=self.STEPSIZE,
                sigma=SigmaConfig(
                    value=1.0, estimate=True, prior_alpha=2.0, n_data_samples=self.N_DATA
                ),
                disable_progressbar=True,
            ),
        )
        assert state.data_sigma is not None
        return state.data_sigma

    def _run_minibatch(self, x0, grad_fn, sampler_seed):
        sampler_rng = np.random.default_rng(sampler_seed)

        def sampler(_iteration):
            return sampler_rng.choice(self.N_DATA, size=self.BATCH_SIZE, replace=False)

        state = simplesvgd.update(
            x0.copy(),
            grad_fn,
            SVGDConfig(
                n_iter=self.N_ITER,
                stepsize=self.STEPSIZE,
                sigma=SigmaConfig(
                    value=1.0, estimate=True, prior_alpha=2.0, n_data_samples=self.N_DATA
                ),
                minibatch_sampler=sampler,
                disable_progressbar=True,
            ),
        )
        assert state.data_sigma is not None
        return state.data_sigma

    def test_minibatch_sigma_matches_full_batch(self):
        rng = np.random.default_rng(123)
        data = _make_gaussian_data(rng, self.N_DATA, theta_true=0.0, sigma_true=self.SIGMA_TRUE)
        grad_fn = _sum_grad_fn(data)

        x0_rng = np.random.default_rng(0)
        x0 = x0_rng.normal(0, 2, size=(30, 1))

        full_sigma = self._run_full(x0, grad_fn)
        mini_sigmas = [self._run_minibatch(x0, grad_fn, seed) for seed in range(8)]
        mean_mini_sigma = float(np.mean(mini_sigmas))

        # A broken/missing rescale would underestimate sigma by roughly
        # sqrt(BATCH_SIZE / N_DATA) = sqrt(20/200) ~= 0.32 -- well outside
        # this tolerance around the full-batch reference.
        assert mean_mini_sigma == pytest.approx(full_sigma, rel=0.3)

    def test_broken_rescale_would_fail_calibration(self, monkeypatch):
        """Guards the test above against a no-op rescale silently agreeing."""
        # `simplesvgd/__init__.py` does `from .update import update`, which
        # shadows the `update` submodule attribute on the `simplesvgd`
        # package with the function -- so `import simplesvgd.update as x`
        # (attribute access on the already-imported package) binds `x` to
        # that function, not the module. importlib.import_module() goes
        # through sys.modules and returns the actual module either way.
        update_module = importlib.import_module("simplesvgd.update")

        monkeypatch.setattr(
            update_module,
            "_rescale_for_minibatch",
            lambda all_grads, misfits, _batch_indices, _n_data_samples: (all_grads, misfits),
        )

        rng = np.random.default_rng(123)
        data = _make_gaussian_data(rng, self.N_DATA, theta_true=0.0, sigma_true=self.SIGMA_TRUE)
        grad_fn = _sum_grad_fn(data)

        x0_rng = np.random.default_rng(0)
        x0 = x0_rng.normal(0, 2, size=(30, 1))

        full_sigma = self._run_full(x0, grad_fn)
        mini_sigmas = [self._run_minibatch(x0, grad_fn, seed) for seed in range(4)]
        mean_mini_sigma = float(np.mean(mini_sigmas))

        assert mean_mini_sigma != pytest.approx(full_sigma, rel=0.3)
