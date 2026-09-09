"""Core SVGD update function with optional preconditioning and hierarchical sigma."""

import math
from dataclasses import dataclass
from typing import Generic, cast

import numpy as np
import numpy.typing as npt
import tqdm.auto as tqdm_auto

from ._animation import Animation, draw_frame, setup_animation
from ._typing import BatchIndices, FloatDType, GradientFn, KernelFn, MinibatchGradientFn
from .config import SVGDConfig
from .kernels import rbf_kernel, rbf_kernel_normalized
from .lbfgs import LBFGSState, lbfgs_direction, lbfgs_update, make_lbfgs_state
from .state import SVGDState

# AdaGrad's momentum and fudge factor aren't user-configurable -- they're
# implementation details of the legacy default preconditioner, not knobs.
_ADAGRAD_ALPHA = 0.9
_ADAGRAD_FUDGE = 1e-6


@dataclass
class _RunState(Generic[FloatDType]):
    """Mutable loop-local state for one `update()` call (fresh or resumed)."""

    particles: npt.NDArray[FloatDType]
    n_particles: int
    start_iter: int
    sigma_history: list[float]
    misfit_history: list[float]
    particle_misfit_history: list[list[float]]
    prev_particles: npt.NDArray[FloatDType] | None
    prev_grads: npt.NDArray[FloatDType] | None
    lbfgs_states: list[LBFGSState[FloatDType]] | None
    historical_grad: npt.NDArray[FloatDType] | None
    current_sigma: float | None


@dataclass
class _StepInputs(Generic[FloatDType]):
    """Per-iteration quantities needed to compute the particle displacement."""

    kernel_matrix: npt.NDArray[FloatDType]
    kernel_grad: npt.NDArray[FloatDType]
    phi: npt.NDArray[FloatDType]
    attractive: npt.NDArray[FloatDType]
    precond_grads: npt.NDArray[FloatDType]


def _resolve_kernel_fn(kernel: "str | KernelFn[FloatDType] | None") -> KernelFn[FloatDType]:
    if kernel is None or kernel == "rbf":
        return rbf_kernel
    if kernel == "rbf_normalized":
        return rbf_kernel_normalized
    if not isinstance(kernel, str):
        return kernel
    raise ValueError(f"Unknown kernel: {kernel!r}")


def _resolve_step_schedule(step_schedule: str | None, preconditioner: str | None) -> str:
    if step_schedule is not None:
        return step_schedule
    return "robbins-monro" if preconditioner == "lbfgs" else "adagrad"


def _init_run_state(
    x0: npt.NDArray[FloatDType],
    config: SVGDConfig[FloatDType],
    *,
    use_lbfgs: bool,
    step_schedule: str,
) -> _RunState[FloatDType]:
    resume_from = config.resume_from
    if resume_from is None:
        particles = np.copy(x0)
        n_particles = particles.shape[0]
        lbfgs_states = (
            [
                make_lbfgs_state(particles.shape[1], m=config.lbfgs.history, dtype=particles.dtype)
                for _ in range(n_particles)
            ]
            if use_lbfgs
            else None
        )
        return _RunState(
            particles=particles,
            n_particles=n_particles,
            start_iter=0,
            sigma_history=[],
            misfit_history=[],
            particle_misfit_history=[],
            prev_particles=None,
            prev_grads=None,
            lbfgs_states=lbfgs_states,
            historical_grad=np.zeros_like(particles) if step_schedule == "adagrad" else None,
            current_sigma=config.sigma.value,
        )

    particles = resume_from.particles.copy()
    n_particles = particles.shape[0]
    if use_lbfgs and resume_from.lbfgs_states is not None:
        lbfgs_states = resume_from.lbfgs_states
    elif use_lbfgs:
        lbfgs_states = [
            make_lbfgs_state(particles.shape[1], m=config.lbfgs.history, dtype=particles.dtype)
            for _ in range(n_particles)
        ]
    else:
        lbfgs_states = None

    if step_schedule == "adagrad":
        historical_grad = (
            resume_from.historical_grad.copy()
            if resume_from.historical_grad is not None
            else np.zeros_like(particles)
        )
    else:
        historical_grad = None

    return _RunState(
        particles=particles,
        n_particles=n_particles,
        start_iter=resume_from.iteration,
        sigma_history=list(resume_from.sigma_history),
        misfit_history=list(resume_from.misfit_history),
        particle_misfit_history=list(resume_from.particle_misfit_history),
        prev_particles=(
            resume_from.prev_particles.copy() if resume_from.prev_particles is not None else None
        ),
        prev_grads=resume_from.prev_grads.copy() if resume_from.prev_grads is not None else None,
        lbfgs_states=lbfgs_states,
        historical_grad=historical_grad,
        current_sigma=(
            resume_from.data_sigma if resume_from.data_sigma is not None else config.sigma.value
        ),
    )


def _resolve_sigma_prior_beta(
    config: SVGDConfig[FloatDType], current_sigma: float | None
) -> float | None:
    if not config.sigma.estimate:
        return config.sigma.prior_beta
    if config.sigma.n_data_samples is None:
        raise ValueError("n_data_samples is required when estimate_sigma=True")
    if current_sigma is None:
        raise ValueError("data_sigma is required when estimate_sigma=True")
    if config.sigma.prior_beta is not None:
        return config.sigma.prior_beta
    return (config.sigma.prior_alpha - 1.0) * current_sigma**2


def _setup_animation(
    config: SVGDConfig[FloatDType], particles: npt.NDArray[FloatDType]
) -> Animation[FloatDType] | None:
    if not config.animation.enabled:
        return None
    return setup_animation(
        figure=config.animation.figure,
        background=config.animation.background,
        particles=particles,
        dimensions_to_plot=config.animation.dimensions_to_plot,
    )


def _evaluate_gradients(
    gradient_fn: "GradientFn[FloatDType] | MinibatchGradientFn[FloatDType]",
    particles: npt.NDArray[FloatDType],
    *,
    needs_misfits: bool,
    batch_indices: BatchIndices | None,
) -> tuple[npt.NDArray[FloatDType], npt.NDArray[FloatDType] | None]:
    # gradient_fn's actual arity/return shape depends on batch_indices and
    # needs_misfits, runtime values the type system can't narrow --
    # re-asserted via cast() rather than weakening the signature to Any.
    if batch_indices is not None:
        result = cast("MinibatchGradientFn[FloatDType]", gradient_fn)(particles, batch_indices)
    else:
        result = cast("GradientFn[FloatDType]", gradient_fn)(particles)
    misfits: npt.NDArray[FloatDType] | None
    if needs_misfits:
        all_grads, misfits = cast(
            "tuple[npt.NDArray[FloatDType], npt.NDArray[FloatDType]]", result
        )
        misfits = np.asarray(misfits)
    else:
        all_grads = cast("npt.NDArray[FloatDType]", result)
        misfits = None
    # gradient_fn is user-supplied and easy to write in a way that silently
    # returns float64 (e.g. building constants with np.zeros(d) instead of
    # matching particles' dtype) -- without this, that alone would upcast
    # the whole particle array on the first `particles + displacement` below.
    all_grads = np.asarray(all_grads, dtype=particles.dtype)
    return all_grads, misfits


def _rescale_for_minibatch(
    all_grads: npt.NDArray[FloatDType],
    misfits: npt.NDArray[FloatDType] | None,
    batch_indices: BatchIndices | None,
    n_data_samples: int | None,
) -> tuple[npt.NDArray[FloatDType], npt.NDArray[FloatDType] | None]:
    if batch_indices is None:
        return all_grads, misfits
    assert n_data_samples is not None  # noqa: S101 -- checked before the loop starts
    # Unbiased estimate of the full-dataset sum from a subsampled sum, so
    # downstream code (attractive term, L-BFGS curvature, sigma estimation)
    # sees quantities of the same scale as a full-batch evaluation would
    # have produced.
    scale = n_data_samples / len(batch_indices)
    scaled_grads = cast("npt.NDArray[FloatDType]", all_grads * scale)
    scaled_misfits = (
        cast("npt.NDArray[FloatDType]", misfits * scale) if misfits is not None else None
    )
    return scaled_grads, scaled_misfits


def _lbfgs_curvature_update(
    run: _RunState[FloatDType], all_grads: npt.NDArray[FloatDType], *, use_lbfgs: bool
) -> None:
    if not (use_lbfgs and run.prev_particles is not None):
        return
    assert run.lbfgs_states is not None  # noqa: S101 -- use_lbfgs invariant
    assert run.prev_grads is not None  # noqa: S101 -- set together with prev_particles
    for i in range(run.n_particles):
        s_vec = run.particles[i] - run.prev_particles[i]
        y_vec = all_grads[i] - run.prev_grads[i]
        lbfgs_update(run.lbfgs_states[i], s_vec, y_vec)


def _record_misfits(
    run: _RunState[FloatDType], misfits: npt.NDArray[FloatDType] | None
) -> float | None:
    if misfits is None:
        return None
    mean_misfit = float(np.mean(misfits))
    run.misfit_history.append(mean_misfit)
    run.particle_misfit_history.append(misfits.tolist())
    return mean_misfit


def _update_sigma(
    run: _RunState[FloatDType],
    config: SVGDConfig[FloatDType],
    misfits: npt.NDArray[FloatDType] | None,
    sigma_prior_beta: float | None,
) -> None:
    if not (config.sigma.estimate and misfits is not None):
        return
    assert sigma_prior_beta is not None  # noqa: S101 -- set by _resolve_sigma_prior_beta when estimate_sigma
    assert config.sigma.n_data_samples is not None  # noqa: S101 -- checked by _resolve_sigma_prior_beta
    raw_misfits_total = float(np.sum(misfits))
    alpha_post = config.sigma.prior_alpha + run.n_particles * config.sigma.n_data_samples / 2.0
    beta_post = sigma_prior_beta + raw_misfits_total
    sigma_sq = beta_post / (alpha_post - 1.0)
    run.current_sigma = float(np.sqrt(sigma_sq))


def _record_sigma_history(run: _RunState[FloatDType]) -> None:
    if run.current_sigma is not None:
        run.sigma_history.append(run.current_sigma)


def _scale_grads_by_sigma(
    precond_grads: npt.NDArray[FloatDType], current_sigma: float | None
) -> npt.NDArray[FloatDType]:
    if current_sigma is None:
        return precond_grads
    return cast("npt.NDArray[FloatDType]", precond_grads / (current_sigma**2))


def _store_lbfgs_prev(
    run: _RunState[FloatDType], all_grads: npt.NDArray[FloatDType], *, use_lbfgs: bool
) -> None:
    if not use_lbfgs:
        return
    run.prev_particles = run.particles.copy()
    run.prev_grads = all_grads.copy()


def _report_progress(
    outer: tqdm_auto.tqdm, mean_misfit: float | None, current_sigma: float | None
) -> None:
    postfix = {}
    if mean_misfit is not None:
        postfix["misfit"] = f"{mean_misfit:.4e}"
    if current_sigma is not None:
        postfix["sigma"] = f"{current_sigma:.2e}"
    if postfix:
        outer.set_postfix(**postfix)


def _snapshot(run: _RunState[FloatDType], iteration: int) -> SVGDState[FloatDType]:
    return SVGDState(
        particles=run.particles.copy(),
        iteration=iteration,
        lbfgs_states=run.lbfgs_states,
        historical_grad=run.historical_grad,
        data_sigma=run.current_sigma,
        sigma_history=list(run.sigma_history),
        misfit_history=list(run.misfit_history),
        particle_misfit_history=list(run.particle_misfit_history),
        prev_particles=run.prev_particles,
        prev_grads=run.prev_grads,
    )


def _precondition_gradients(
    run: _RunState[FloatDType], all_grads: npt.NDArray[FloatDType], *, use_lbfgs: bool
) -> npt.NDArray[FloatDType]:
    if not use_lbfgs:
        return all_grads
    assert run.lbfgs_states is not None  # noqa: S101 -- use_lbfgs invariant
    precond_grads = np.zeros_like(run.particles)
    for i in range(run.n_particles):
        # lbfgs_direction returns -H*g; negate to get H*g
        precond_grads[i] = -lbfgs_direction(run.lbfgs_states[i], all_grads[i])
    return precond_grads


def _compute_displacement(
    step_schedule: str,
    run: _RunState[FloatDType],
    config: SVGDConfig[FloatDType],
    inputs: _StepInputs[FloatDType],
    iteration: int,
) -> npt.NDArray[FloatDType]:
    if step_schedule == "robbins-monro":
        attr_max = np.max(np.abs(inputs.attractive)) / run.n_particles
        # math.sqrt (not np.sqrt) on this plain-Python scalar: np.sqrt of a
        # bare int/float with no array context always returns float64, which
        # would silently upcast `step` and then `displacement` even when phi
        # is float32.
        decay = math.sqrt(1.0 + iteration)
        step = config.stepsize / (attr_max * decay) if attr_max > 0 else config.stepsize / decay
        # scalar * NDArray[FloatDType] loses the TypeVar binding in numpy's
        # stubs (same rough edge as the .min()/.max() cases elsewhere).
        return cast("npt.NDArray[FloatDType]", step * inputs.phi)

    if step_schedule == "adagrad":
        assert run.historical_grad is not None  # noqa: S101 -- step_schedule invariant
        grad_theta = (
            np.matmul(inputs.kernel_matrix, inputs.precond_grads) - inputs.kernel_grad
        ) / run.n_particles
        if iteration == 0:
            run.historical_grad = grad_theta**2
        else:
            run.historical_grad = (
                _ADAGRAD_ALPHA * run.historical_grad + (1 - _ADAGRAD_ALPHA) * grad_theta**2
            )
        adj_grad = grad_theta / (_ADAGRAD_FUDGE + np.sqrt(run.historical_grad))
        return -config.stepsize * adj_grad

    if step_schedule == "constant":
        return cast("npt.NDArray[FloatDType]", config.stepsize * inputs.phi)

    raise ValueError(f"Unknown step_schedule: {step_schedule!r}")


def update(
    x0: npt.NDArray[FloatDType],
    gradient_fn: "GradientFn[FloatDType] | MinibatchGradientFn[FloatDType]",
    config: SVGDConfig[FloatDType] | None = None,
) -> SVGDState[FloatDType]:
    """Update a collection of samples using Stein Variational Gradient Descent.

    Parameters
    ----------
    x0 : np.ndarray
        Initial particle positions, shape ``(n_particles, n_dims)``. Its
        dtype (e.g. ``float32`` or ``float64``) is preserved throughout the
        run -- pass ``float32`` particles to run the whole optimization in
        single precision. ``gradient_fn`` should return gradients in the
        same dtype; if it doesn't, they are cast to ``x0``'s dtype before
        use, so a careless ``gradient_fn`` can't silently upcast the run.
        Ignored when ``config.resume_from`` is set -- the run continues from
        ``resume_from.particles`` instead.
    gradient_fn : callable
        Computes gradients of the negative log-probability. Accepts particles
        of shape ``(n_particles, n_dims)`` and returns gradients of the same
        shape. When ``config.sigma.value`` is set or ``config.sigma.estimate``
        is ``True``, must return ``(gradients, misfits)`` where misfits has
        shape ``(n_particles,)``. When ``config.minibatch_sampler`` is set,
        called instead as ``gradient_fn(particles, batch_indices)`` -- see
        :class:`SVGDConfig`.
    config : SVGDConfig or None
        Every other tunable of the run (iteration count, step size,
        preconditioner, kernel, bounds, callback, resume, animation, ...).
        ``None`` uses ``SVGDConfig()``'s defaults. See :class:`SVGDConfig`
        for the full list of fields.

    Returns
    -------
    SVGDState
        Final optimizer state. Access ``.particles`` for the particle array.

    """
    if x0 is None or gradient_fn is None:
        raise ValueError("x0 and gradient_fn cannot be None!")
    if config is None:
        config = SVGDConfig()
    if config.minibatch_sampler is not None and config.sigma.n_data_samples is None:
        raise ValueError("n_data_samples is required when minibatch_sampler is set")

    kernel_fn = _resolve_kernel_fn(config.kernel)
    needs_misfits = config.sigma.value is not None or config.sigma.estimate
    step_schedule = _resolve_step_schedule(config.step_schedule, config.preconditioner)
    use_lbfgs = config.preconditioner == "lbfgs"

    run = _init_run_state(x0, config, use_lbfgs=use_lbfgs, step_schedule=step_schedule)
    sigma_prior_beta = _resolve_sigma_prior_beta(config, run.current_sigma)
    anim = _setup_animation(config, run.particles)

    outer = tqdm_auto.trange(
        config.n_iter, desc="SVGD", unit="iter", disable=config.disable_progressbar
    )
    try:
        for loop_iter in outer:
            iteration = run.start_iter + loop_iter

            batch_indices = (
                config.minibatch_sampler(iteration)
                if config.minibatch_sampler is not None
                else None
            )
            all_grads, misfits = _evaluate_gradients(
                gradient_fn, run.particles, needs_misfits=needs_misfits, batch_indices=batch_indices
            )
            all_grads, misfits = _rescale_for_minibatch(
                all_grads, misfits, batch_indices, config.sigma.n_data_samples
            )
            _lbfgs_curvature_update(run, all_grads, use_lbfgs=use_lbfgs)

            mean_misfit = _record_misfits(run, misfits)
            _update_sigma(run, config, misfits, sigma_prior_beta)
            _record_sigma_history(run)

            _report_progress(outer, mean_misfit, run.current_sigma)

            if config.callback is not None:
                config.callback(iteration, _snapshot(run, iteration))

            # Last iteration: don't update particles
            if loop_iter == config.n_iter - 1:
                break

            precond_grads = _precondition_gradients(run, all_grads, use_lbfgs=use_lbfgs)
            kernel_matrix, kernel_grad = kernel_fn(run.particles, config.bandwidth)
            grads_scaled = _scale_grads_by_sigma(precond_grads, run.current_sigma)
            # SVGD update direction:
            #   phi = -(K @ grads_scaled - nabla_K) / n_particles
            attractive = np.matmul(kernel_matrix, grads_scaled)
            phi = -(attractive - kernel_grad) / run.n_particles

            inputs = _StepInputs(
                kernel_matrix=kernel_matrix,
                kernel_grad=kernel_grad,
                phi=phi,
                attractive=attractive,
                precond_grads=precond_grads,
            )
            displacement = _compute_displacement(step_schedule, run, config, inputs, iteration)
            _store_lbfgs_prev(run, all_grads, use_lbfgs=use_lbfgs)

            run.particles = run.particles + displacement

            if config.bounds is not None:
                run.particles = np.clip(run.particles, config.bounds[0], config.bounds[1])

            if anim is not None:
                draw_frame(anim, run.particles, config.animation.dimensions_to_plot)

    except KeyboardInterrupt:
        pass
    finally:
        outer.close()

    return SVGDState(
        particles=run.particles,
        iteration=run.start_iter + config.n_iter,
        lbfgs_states=run.lbfgs_states,
        historical_grad=run.historical_grad,
        data_sigma=run.current_sigma,
        sigma_history=run.sigma_history,
        misfit_history=run.misfit_history,
        particle_misfit_history=run.particle_misfit_history,
        prev_particles=run.prev_particles if use_lbfgs else None,
        prev_grads=run.prev_grads if use_lbfgs else None,
    )
