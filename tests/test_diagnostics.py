"""Tests for the variance-collapse diagnostics (particle_variance_history,
repulsion_ratio_history)."""

import numpy as np

import simplesvgd
from simplesvgd import SVGDConfig


def _run(d, n_iter=200, stepsize=0.3, n_particles=30, seed=42):
    true_mean = np.zeros(d)

    def grad_fn(x):
        return x - true_mean

    rng = np.random.default_rng(seed)
    x0 = rng.normal(0, 1, (n_particles, d))
    return simplesvgd.update(
        x0, grad_fn, SVGDConfig(n_iter=n_iter, stepsize=stepsize, disable_progressbar=True)
    )


class TestDiagnosticsBasics:
    def test_particle_variance_history_length_matches_n_iter(self):
        state = _run(d=4, n_iter=10)
        assert len(state.particle_variance_history) == 10

    def test_repulsion_ratio_history_is_one_shorter(self):
        """The last iteration only records state, it doesn't compute a step."""
        state = _run(d=4, n_iter=10)
        assert len(state.repulsion_ratio_history) == 9

    def test_diagnostics_survive_resume(self):
        rng = np.random.default_rng(0)
        x0 = rng.normal(0, 1, (10, 3))

        def grad_fn(x):
            return x

        state1 = simplesvgd.update(
            x0, grad_fn, SVGDConfig(n_iter=5, stepsize=0.1, disable_progressbar=True)
        )
        state2 = simplesvgd.update(
            x0,
            grad_fn,
            SVGDConfig(n_iter=5, stepsize=0.1, resume_from=state1, disable_progressbar=True),
        )
        assert len(state2.particle_variance_history) == 10
        # Each of the two 5-iteration update() calls independently skips
        # computing a displacement (and thus a repulsion ratio) on its own
        # last iteration, so resuming drops one more entry than a single
        # continuous 10-iteration run would (which has 9, not 8).
        assert len(state2.repulsion_ratio_history) == 8
        assert state2.particle_variance_history[:5] == state1.particle_variance_history


class TestVarianceCollapseDiagnosticsFlagCollapse:
    """A standard RBF kernel on a d=500 isotropic Gaussian target collapses
    (the same scenario as test_svgd.py's TestHighDimensionalGaussian); a
    low-dimensional (d=2) run on the same target family is well-behaved. The
    diagnostics should tell the two apart, robustly across seeds.
    """

    def test_particle_variance_collapses_in_high_dimensions(self):
        for seed in range(6):
            state_low = _run(d=2, seed=seed)
            state_high = _run(d=500, seed=seed)

            var_ratio_low = state_low.particle_variance_history[-1] / (
                state_low.particle_variance_history[0]
            )
            var_ratio_high = state_high.particle_variance_history[-1] / (
                state_high.particle_variance_history[0]
            )

            # Low-d: variance stays within a factor of ~1.5 of its start
            # (the target's own per-dim variance is 1, matching the initial
            # spread, so it neither collapses nor blows up).
            assert 0.5 < var_ratio_low < 1.5, (
                f"seed={seed}: expected the low-d run to keep its variance, got "
                f"ratio={var_ratio_low:.3f}"
            )
            # High-d: variance collapses to a small fraction of its start
            # (empirically <0.03 across seeds, with a lot of margin below
            # the low-d floor of 0.5 used above).
            assert var_ratio_high < 0.1, (
                f"seed={seed}: expected the high-d run to variance-collapse, got "
                f"ratio={var_ratio_high:.4f}"
            )

    def test_repulsion_ratio_flags_collapse_early_in_the_run(self):
        """Early in the run (before the attractive term has decayed toward
        zero near convergence, which would swamp the ratio regardless of
        collapse), a repulsion ratio far below 1 means repulsion is already
        overwhelmed by attraction -- the direct mechanism behind collapse.
        """
        for seed in range(6):
            state_low = _run(d=2, seed=seed)
            state_high = _run(d=500, seed=seed)

            early_low = np.mean(state_low.repulsion_ratio_history[:5])
            early_high = np.mean(state_high.repulsion_ratio_history[:5])

            assert early_low > 0.5, (
                f"seed={seed}: expected the low-d run's early repulsion ratio near 1, "
                f"got {early_low:.4f}"
            )
            assert early_high < 0.05, (
                f"seed={seed}: expected the high-d run's early repulsion ratio to be "
                f"tiny (repulsion overwhelmed by attraction), got {early_high:.4f}"
            )
