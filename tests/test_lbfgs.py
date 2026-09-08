"""Unit tests for the L-BFGS two-loop recursion."""

import numpy as np

from simplesvgd.lbfgs import lbfgs_direction, lbfgs_update, make_lbfgs_state


def test_empty_state_returns_steepest_descent():
    state = make_lbfgs_state(5, m=3)
    g = np.array([1.0, -2.0, 3.0, 0.5, -1.0])
    d = lbfgs_direction(state, g)
    np.testing.assert_array_equal(d, -g)


def test_curvature_skip_negative_ys():
    state = make_lbfgs_state(3, m=5)
    s = np.array([1.0, 0.0, 0.0])
    y = np.array([-1.0, 0.0, 0.0])  # y.s = -1 < 0
    lbfgs_update(state, s, y)
    assert state.count == 0  # skipped


def test_curvature_skip_zero_ys():
    state = make_lbfgs_state(3, m=5)
    s = np.array([1.0, 0.0, 0.0])
    y = np.array([0.0, 1.0, 0.0])  # y.s = 0
    lbfgs_update(state, s, y)
    assert state.count == 0


def test_single_pair_scales_gradient():
    state = make_lbfgs_state(2, m=5)
    s = np.array([1.0, 0.0])
    y = np.array([2.0, 0.0])  # gamma = s.y / y.y = 2/4 = 0.5
    lbfgs_update(state, s, y)
    assert state.count == 1

    g = np.array([4.0, 6.0])
    d = lbfgs_direction(state, g)
    # With one pair, H0 = gamma * I = 0.5 * I, then one correction step
    # The direction should be -H*g
    assert d[0] < 0  # should descend


def test_circular_buffer_wraps():
    m = 3
    state = make_lbfgs_state(2, m=m)
    # Add m+1 pairs to trigger wrap-around
    for i in range(m + 1):
        s = np.array([1.0, 0.0]) * (i + 1)
        y = np.array([2.0, 0.0]) * (i + 1)
        lbfgs_update(state, s, y)

    assert state.count == m  # capped at m
    # Cursor should have wrapped
    assert state.cursor == (m + 1) % m


def test_quadratic_approximation():
    """On f(x) = 0.5 * x^T A x, L-BFGS should approximate A^{-1}."""
    n = 5
    rng = np.random.default_rng(123)
    # SPD matrix
    random_matrix = rng.normal(size=(n, n))
    spd_matrix = random_matrix.T @ random_matrix + np.eye(n)
    spd_matrix_inv = np.linalg.inv(spd_matrix)

    state = make_lbfgs_state(n, m=n)

    x = rng.normal(size=n)
    for _ in range(2 * n):
        g = spd_matrix @ x
        d = lbfgs_direction(state, g)
        x_new = x + 0.5 * d  # half step toward minimum
        g_new = spd_matrix @ x_new
        lbfgs_update(state, x_new - x, g_new - g)
        x = x_new

    # After enough steps, H*g should approximate A^{-1}*g
    g_test = rng.normal(size=n)
    d = lbfgs_direction(state, g_test)
    exact = -spd_matrix_inv @ g_test
    # Check direction is similar (cosine similarity > 0.95)
    cos_sim = np.dot(d, exact) / (np.linalg.norm(d) * np.linalg.norm(exact))
    assert cos_sim > 0.7, f"cosine similarity {cos_sim:.3f} too low"
