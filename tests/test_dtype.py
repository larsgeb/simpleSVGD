"""dtype-preservation tests: float32 inputs should stay float32 throughout.

NumPy silently upcasts on any float32/float64 mix, so it's easy for a
single stray np.zeros(...)/np.log(...)/scipy call to promote an entire
pipeline to float64. These tests pin dtype at each layer (kernels, L-BFGS,
the end-to-end update loop) so a regression shows up as a failing dtype
assertion rather than a quiet 2x memory/compute regression.
"""

import numpy as np
import pytest

import simplesvgd
from simplesvgd.kernels import rbf_kernel, rbf_kernel_normalized
from simplesvgd.lbfgs import lbfgs_direction, lbfgs_update, make_lbfgs_state

DTYPES = [np.float32, np.float64]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kernel_fn", [rbf_kernel, rbf_kernel_normalized])
def test_kernel_preserves_dtype(kernel_fn, dtype):
    rng = np.random.default_rng(0)
    particles = rng.normal(size=(15, 4)).astype(dtype)
    kxy, dxkxy = kernel_fn(particles)
    assert kxy.dtype == dtype
    assert dxkxy.dtype == dtype


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kernel_fn", [rbf_kernel, rbf_kernel_normalized])
def test_kernel_degenerate_bandwidth_preserves_dtype(kernel_fn, dtype):
    """All particles identical: zero-bandwidth guard branch."""
    particles = np.ones((5, 3), dtype=dtype)
    kxy, dxkxy = kernel_fn(particles)
    assert kxy.dtype == dtype
    assert dxkxy.dtype == dtype


@pytest.mark.parametrize("dtype", DTYPES)
def test_kernel_explicit_bandwidth_preserves_dtype(dtype):
    rng = np.random.default_rng(1)
    particles = rng.normal(size=(10, 3)).astype(dtype)
    kxy, dxkxy = rbf_kernel(particles, h=1.5)
    assert kxy.dtype == dtype
    assert dxkxy.dtype == dtype


def test_kernel_float32_matches_float64_closely():
    """float32 should be numerically close to float64, not just same-shaped noise."""
    rng = np.random.default_rng(2)
    particles64 = rng.normal(size=(20, 6)).astype(np.float64)
    particles32 = particles64.astype(np.float32)

    _, g64 = rbf_kernel(particles64)
    _, g32 = rbf_kernel(particles32)

    rel_err = np.max(np.abs(g32.astype(np.float64) - g64)) / np.max(np.abs(g64))
    assert rel_err < 1e-5


@pytest.mark.parametrize("dtype", DTYPES)
def test_lbfgs_direction_preserves_dtype(dtype):
    state = make_lbfgs_state(4, m=3, dtype=dtype)
    assert state.s_history.dtype == dtype
    assert state.y_history.dtype == dtype

    rng = np.random.default_rng(3)
    for _ in range(3):
        s = rng.normal(size=4).astype(dtype)
        y = s * dtype(0.5) + rng.normal(size=4).astype(dtype) * dtype(0.01)
        lbfgs_update(state, s, y)

    grad = rng.normal(size=4).astype(dtype)
    direction = lbfgs_direction(state, grad)
    assert direction.dtype == dtype


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("preconditioner", [None, "lbfgs"])
@pytest.mark.parametrize("kernel", [None, "rbf_normalized"])
def test_update_preserves_dtype_end_to_end(dtype, preconditioner, kernel):
    true_mean = np.zeros(5, dtype=dtype)

    def grad_fn(x):
        return x - true_mean

    rng = np.random.default_rng(4)
    x0 = rng.normal(size=(10, 5)).astype(dtype)
    state = simplesvgd.update(
        x0,
        grad_fn,
        n_iter=5,
        stepsize=0.1,
        preconditioner=preconditioner,
        kernel=kernel,
        disable_progressbar=True,
    )
    assert state.particles.dtype == dtype


@pytest.mark.parametrize("dtype", DTYPES)
def test_update_constant_step_schedule_preserves_dtype(dtype):
    true_mean = np.zeros(5, dtype=dtype)

    def grad_fn(x):
        return x - true_mean

    rng = np.random.default_rng(6)
    x0 = rng.normal(size=(10, 5)).astype(dtype)
    state = simplesvgd.update(
        x0, grad_fn, n_iter=5, stepsize=0.1, step_schedule="constant",
        disable_progressbar=True,
    )
    assert state.particles.dtype == dtype


@pytest.mark.parametrize("dtype", DTYPES)
def test_update_bounds_preserve_dtype(dtype):
    true_mean = np.zeros(5, dtype=dtype)

    def grad_fn(x):
        return x - true_mean

    rng = np.random.default_rng(7)
    x0 = rng.normal(size=(10, 5)).astype(dtype)
    state = simplesvgd.update(
        x0, grad_fn, n_iter=5, stepsize=0.1, bounds=(-2.0, 2.0),
        disable_progressbar=True,
    )
    assert state.particles.dtype == dtype


@pytest.mark.parametrize("dtype", DTYPES)
def test_update_hierarchical_sigma_preserves_dtype(dtype):
    true_mean = np.zeros(3, dtype=dtype)

    def grad_fn(x):
        misfits = np.sum((x - true_mean) ** 2, axis=1)
        return (x - true_mean), misfits

    rng = np.random.default_rng(8)
    x0 = rng.normal(size=(10, 3)).astype(dtype)
    state = simplesvgd.update(
        x0, grad_fn, n_iter=5, stepsize=0.1,
        data_sigma=1.0, estimate_sigma=True, n_data_samples=100,
        disable_progressbar=True,
    )
    assert state.particles.dtype == dtype


def test_update_survives_a_float64_gradient_fn_with_float32_particles():
    """A gradient_fn that carelessly returns float64 shouldn't upcast particles."""
    true_mean = np.zeros(5, dtype=np.float64)  # deliberately float64

    def careless_grad_fn(x):
        return x.astype(np.float64) - true_mean

    rng = np.random.default_rng(5)
    x0 = rng.normal(size=(10, 5)).astype(np.float32)
    state = simplesvgd.update(
        x0, careless_grad_fn, n_iter=5, stepsize=0.1, disable_progressbar=True,
    )
    assert state.particles.dtype == np.float32
