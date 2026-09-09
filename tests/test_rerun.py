"""Smoke tests for the Rerun live-visualization path (requires rerun-sdk)."""

import numpy as np
import pytest

pytest.importorskip("rerun")

import simplesvgd
from simplesvgd import RerunConfig, SigmaConfig, SVGDConfig


def _run(*, d, dimensions_to_plot=None, sigma=None, n_iter=5):
    true_mean = np.ones(d)

    def grad_fn(x):
        if sigma is not None:
            misfits = 0.5 * np.sum((x - true_mean) ** 2, axis=1)
            return x - true_mean, misfits
        return x - true_mean

    rng = np.random.default_rng(0)
    x0 = rng.normal(0, 1, (10, d))
    rerun_config = (
        RerunConfig(enabled=True, spawn=False, dimensions_to_plot=dimensions_to_plot)
        if dimensions_to_plot is not None
        else RerunConfig(enabled=True, spawn=False)
    )
    return simplesvgd.update(
        x0,
        grad_fn,
        SVGDConfig(
            n_iter=n_iter,
            stepsize=0.3,
            disable_progressbar=True,
            sigma=SigmaConfig(value=sigma) if sigma is not None else SigmaConfig(),
            rerun=rerun_config,
        ),
    )


class TestRerunLogging:
    def test_runs_without_a_spawned_viewer(self):
        state = _run(d=2)
        assert state.particles.shape == (10, 2)

    def test_logs_misfit_and_sigma_when_tracked(self):
        state = _run(d=2, sigma=1.0)
        assert state.particles.shape == (10, 2)

    def test_custom_dimensions_to_plot_on_higher_dimensional_problem(self):
        state = _run(d=5, dimensions_to_plot=[1, 3])
        assert state.particles.shape == (10, 5)

    def test_multiple_runs_in_one_process_do_not_crash(self):
        # Each update() call opens its own recording (a fresh recording_id) --
        # this exercises that repeated calls in the same process don't share
        # state in a way that breaks logging.
        state1 = _run(d=2)
        state2 = _run(d=2, n_iter=3)
        assert state1.particles.shape == (10, 2)
        assert state2.particles.shape == (10, 2)


class TestRerunDisabledByDefault:
    def test_disabled_by_default(self):
        assert RerunConfig().enabled is False

    def test_disabled_run_does_not_touch_rerun(self):
        # rerun is only imported inside setup_rerun/log_iteration, which are
        # only called when config.rerun.enabled -- so a disabled run works
        # identically regardless of whether rerun-sdk is installed.
        rng = np.random.default_rng(0)
        x0 = rng.normal(0, 1, (10, 2))

        def grad_fn(x):
            return x - np.ones(2)

        state = simplesvgd.update(
            x0, grad_fn, SVGDConfig(n_iter=3, stepsize=0.3, disable_progressbar=True)
        )
        assert state.particles.shape == (10, 2)
