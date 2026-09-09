"""Tests for the rich-based live progress display (simplesvgd._progress)."""

import numpy as np

import simplesvgd
from simplesvgd import SVGDConfig
from simplesvgd._progress import format_stats


class TestFormatStats:
    def test_includes_only_available_fields(self):
        stats = format_stats(None, None, 1.5, None)
        assert "misfit" not in stats
        assert "sigma" not in stats
        assert "rep" not in stats
        assert "var=1.500e+00" in stats

    def test_includes_all_fields_when_available(self):
        stats = format_stats(0.01, 0.5, 1.5, 0.9)
        assert "misfit=1.000e-02" in stats
        assert "sigma=5.00e-01" in stats
        assert "var=1.500e+00" in stats
        assert "rep=0.90" in stats


class TestProgressDisplay:
    def _run(self, *, disable_progressbar):
        d = 2
        true_mean = np.ones(d)

        def grad_fn(x):
            return x - true_mean

        rng = np.random.default_rng(0)
        x0 = rng.normal(0, 1, (10, d))
        return simplesvgd.update(
            x0,
            grad_fn,
            SVGDConfig(n_iter=5, stepsize=0.3, disable_progressbar=disable_progressbar),
        )

    def test_enabled_progressbar_runs_without_error(self, capsys):
        state = self._run(disable_progressbar=False)
        assert state.particles.shape == (10, 2)
        captured = capsys.readouterr()
        assert "SVGD" in captured.out
        assert "var=" in captured.out

    def test_disabled_progressbar_produces_no_output(self, capsys):
        state = self._run(disable_progressbar=True)
        assert state.particles.shape == (10, 2)
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""
