"""Tests for temperature_schedule (issue #8): tempering the misfit gradient.

TestResolveTemperatureFn exercises the schedule-shape math directly (the
private helper is the simplest place to pin exact ramp values). The rest
goes through the public update() API: that a custom callable is wired in
correctly, that an unknown name raises, and -- the acceptance criterion
from the issue -- that annealing measurably improves variance recovery on
a target where plain SVGD is known to under-estimate posterior variance.
"""

import itertools

import numpy as np
import pytest

import simplesvgd
from simplesvgd import SVGDConfig
from simplesvgd.update import _resolve_temperature_fn


class TestResolveTemperatureFn:
    def test_none_is_constant_one(self):
        fn = _resolve_temperature_fn(None, n_iter=50)
        assert fn(0) == 1.0
        assert fn(25) == 1.0
        assert fn(49) == 1.0

    def test_linear_ramps_from_floor_to_one(self):
        fn = _resolve_temperature_fn("linear", n_iter=101)
        assert fn(0) == pytest.approx(0.01)
        assert fn(50) == pytest.approx(0.505)
        assert fn(100) == pytest.approx(1.0)

    def test_linear_is_monotonically_increasing(self):
        fn = _resolve_temperature_fn("linear", n_iter=50)
        values = [fn(i) for i in range(50)]
        assert all(b >= a for a, b in itertools.pairwise(values))

    def test_geometric_ramps_from_floor_to_one(self):
        fn = _resolve_temperature_fn("geometric", n_iter=101)
        assert fn(0) == pytest.approx(0.01)
        assert fn(50) == pytest.approx(0.1)
        assert fn(100) == pytest.approx(1.0)

    def test_geometric_is_monotonically_increasing(self):
        fn = _resolve_temperature_fn("geometric", n_iter=50)
        values = [fn(i) for i in range(50)]
        assert all(b >= a for a, b in itertools.pairwise(values))

    def test_custom_callable_is_returned_directly(self):
        def custom(_iteration):
            return 0.5

        assert _resolve_temperature_fn(custom, n_iter=50) is custom

    def test_single_iteration_run_does_not_divide_by_zero(self):
        fn = _resolve_temperature_fn("linear", n_iter=1)
        assert fn(0) == pytest.approx(0.01)

    def test_unknown_schedule_name_raises(self):
        with pytest.raises(ValueError, match="Unknown temperature_schedule"):
            _resolve_temperature_fn("not-a-schedule", n_iter=50)


class TestUpdateWithTemperatureSchedule:
    def test_custom_callable_is_called_once_per_displacement_step(self):
        """temperature_fn is called for every iteration that computes a
        displacement (0..n_iter-2) -- not the last iteration, which only
        records diagnostics before breaking."""
        calls = []

        def recording_schedule(iteration):
            calls.append(iteration)
            return 1.0

        rng = np.random.default_rng(0)
        x0 = rng.normal(size=(5, 2))
        simplesvgd.update(
            x0,
            lambda x: x,
            SVGDConfig(
                n_iter=5,
                stepsize=0.1,
                temperature_schedule=recording_schedule,
                disable_progressbar=True,
            ),
        )
        assert calls == list(range(4))

    def test_unknown_temperature_schedule_string_raises(self):
        rng = np.random.default_rng(0)
        x0 = rng.normal(size=(5, 2))
        with pytest.raises(ValueError, match="Unknown temperature_schedule"):
            simplesvgd.update(
                x0,
                lambda x: x,
                SVGDConfig(
                    n_iter=2, temperature_schedule="not-a-schedule", disable_progressbar=True
                ),
            )


class TestAnnealingImprovesVarianceRecovery:
    """Acceptance criterion from issue #8: on a synthetic multimodal target,
    annealing should recover posterior variance better than applying the
    full misfit strength from iteration 0.

    Target: a D=20 two-component Gaussian mixture (well-separated, equal
    weight), the setting where SVGD's kernel repulsion is well documented to
    weaken relative to the attractive term as dimension grows (Ba et al.,
    "Understanding the Variance Collapse of SVGD in High Dimensions", ICLR
    2022) -- exactly the failure mode annealing is meant to mitigate for
    full-waveform-inversion-style posteriors (see issue #8's cited paper).
    A "constant" step schedule is used so the comparison isolates the
    temperature scaling itself, without AdaGrad's own gradient normalization
    also damping the early steps.
    """

    D = 20
    MU = 3.0
    SIGMA = 0.7

    def _grad_fn(self, x):
        d1 = x - self.MU
        d2 = x + self.MU
        logn1 = -0.5 * np.sum(d1**2, axis=1) / self.SIGMA**2
        logn2 = -0.5 * np.sum(d2**2, axis=1) / self.SIGMA**2
        m = np.maximum(logn1, logn2)
        w1 = np.exp(logn1 - m)
        w2 = np.exp(logn2 - m)
        denom = w1 + w2
        score = (w1[:, None] * (-d1 / self.SIGMA**2) + w2[:, None] * (-d2 / self.SIGMA**2)) / denom[
            :, None
        ]
        return -score

    def test_annealed_variance_exceeds_unannealed_across_seeds(self):
        base_vars = []
        anneal_vars = []
        for seed in range(6):
            rng = np.random.default_rng(seed)
            x0 = rng.normal(0.0, 2.0, size=(60, self.D))

            state_base = simplesvgd.update(
                x0.copy(),
                self._grad_fn,
                SVGDConfig(
                    n_iter=100,
                    stepsize=0.02,
                    step_schedule="constant",
                    disable_progressbar=True,
                ),
            )
            state_anneal = simplesvgd.update(
                x0.copy(),
                self._grad_fn,
                SVGDConfig(
                    n_iter=100,
                    stepsize=0.02,
                    step_schedule="constant",
                    temperature_schedule="linear",
                    disable_progressbar=True,
                ),
            )
            base_vars.append(float(np.mean(np.var(state_base.particles, axis=0))))
            anneal_vars.append(float(np.mean(np.var(state_anneal.particles, axis=0))))

        # Holds for every individual seed in this configuration, not just on
        # average -- a stronger and more reproducible claim than a mean-only
        # comparison.
        assert all(a > b for a, b in zip(anneal_vars, base_vars, strict=True))
        assert np.mean(anneal_vars) > np.mean(base_vars) * 1.02
