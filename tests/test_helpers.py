"""Tests for gradient_vectorizer."""

import numpy as np

import simplesvgd


class TestGradientVectorizer:
    def test_matches_manually_vectorized_gradient(self):
        """Wrapping a single-point gradient should match a hand-vectorized one."""

        def single_point_grad(x):
            # x has shape (d, 1); mimics the README's Himmelblau_grad convention.
            return 2 * x + 1

        def vectorized_reference(batch):
            return 2 * batch + 1

        vectorized = simplesvgd.gradient_vectorizer(single_point_grad)
        batch = np.array([[1.0, 2.0], [3.0, 4.0], [-5.0, 6.0]])

        result = vectorized(batch)

        assert result.shape == batch.shape
        np.testing.assert_allclose(result, vectorized_reference(batch))

    def test_usable_directly_in_update(self):
        """The vectorized wrapper should work as update()'s gradient_fn."""
        true_mean = np.array([1.0, -1.0])

        def single_point_grad(x):
            return x - true_mean[:, None]

        vectorized_grad = simplesvgd.gradient_vectorizer(single_point_grad)

        rng = np.random.default_rng(0)
        x0 = rng.normal(0, 3, (100, 2))
        state = simplesvgd.update(
            x0,
            vectorized_grad,
            simplesvgd.SVGDConfig(n_iter=200, stepsize=0.3, disable_progressbar=True),
        )
        final_error = np.linalg.norm(np.mean(state.particles, axis=0) - true_mean)
        assert final_error < 0.5
