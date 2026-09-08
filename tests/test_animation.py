"""Tests for the legacy live-scatter animation path (requires matplotlib)."""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import simplesvgd  # noqa: E402 -- backend must be set before matplotlib.pyplot is imported
from simplesvgd import AnimationConfig, SVGDConfig  # noqa: E402
from simplesvgd._animation import draw_frame, setup_animation  # noqa: E402


class TestSetupAnimation:
    def test_scatter_starts_at_particle_positions(self):
        rng = np.random.default_rng(0)
        particles = rng.normal(size=(10, 2))

        anim = setup_animation(
            figure=None, background=None, particles=particles, dimensions_to_plot=[0, 1]
        )

        np.testing.assert_allclose(np.asarray(anim.scatter.get_offsets()), particles)

    def test_background_sets_axis_limits_to_grid_extent(self):
        rng = np.random.default_rng(1)
        particles = rng.normal(size=(5, 2))
        x1s = np.linspace(-3, 4, 20)
        x2s = np.linspace(-2, 5, 20)
        background_image = np.zeros((20, 20))

        anim = setup_animation(
            figure=None,
            background=(x1s, x2s, background_image),
            particles=particles,
            dimensions_to_plot=[0, 1],
        )

        assert anim.axis.get_xlim() == pytest.approx((-3, 4))
        assert anim.axis.get_ylim() == pytest.approx((-2, 5))

    def test_picks_the_requested_dimensions(self):
        particles = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        anim = setup_animation(
            figure=None, background=None, particles=particles, dimensions_to_plot=[0, 2]
        )

        np.testing.assert_allclose(np.asarray(anim.scatter.get_offsets()), particles[:, [0, 2]])


class TestDrawFrame:
    def test_updates_scatter_offsets_to_new_positions(self):
        rng = np.random.default_rng(2)
        particles = rng.normal(size=(8, 2))
        anim = setup_animation(
            figure=None, background=None, particles=particles, dimensions_to_plot=[0, 1]
        )

        moved_particles = particles + 5.0
        draw_frame(anim, moved_particles, dimensions_to_plot=[0, 1])

        np.testing.assert_allclose(np.asarray(anim.scatter.get_offsets()), moved_particles)


class TestUpdateWithAnimation:
    def test_animate_true_runs_end_to_end(self):
        rng = np.random.default_rng(3)
        x0 = rng.normal(size=(15, 2))

        def grad_fn(x):
            return x

        state = simplesvgd.update(
            x0,
            grad_fn,
            SVGDConfig(
                n_iter=3,
                stepsize=0.1,
                animation=AnimationConfig(enabled=True, dimensions_to_plot=[0, 1]),
                disable_progressbar=True,
            ),
        )

        assert state.particles.shape == x0.shape
        assert state.iteration == 3
