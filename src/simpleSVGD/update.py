"""Core SVGD update function with optional preconditioning and hierarchical sigma."""

import math as _math
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import tqdm.auto as _tqdm_auto

from .kernels import rbf_kernel, rbf_kernel_normalized
from .lbfgs import LBFGSState, lbfgs_direction, lbfgs_update, make_lbfgs_state
from .state import SVGDState


def update(
    x0: np.ndarray,
    gradient_fn: Callable,
    *,
    n_iter: int = 1000,
    stepsize: float = 1e-3,
    bandwidth: float = -1,
    # --- Preconditioning ---
    preconditioner: Optional[str] = None,
    lbfgs_history: int = 10,
    # --- Step schedule ---
    step_schedule: Optional[str] = None,
    # --- Hierarchical sigma ---
    data_sigma: Optional[float] = None,
    estimate_sigma: bool = False,
    sigma_prior_alpha: float = 2.0,
    sigma_prior_beta: Optional[float] = None,
    n_data_samples: Optional[int] = None,
    # --- Kernel ---
    kernel: Optional[str] = None,
    # --- Bounds ---
    bounds: Optional[Tuple[float, float]] = None,
    # --- Callback and control ---
    callback: Optional[Callable] = None,
    disable_progressbar: bool = False,
    # --- Resume ---
    resume_from: Optional[SVGDState] = None,
    # --- Legacy animation ---
    animate: bool = False,
    figure=None,
    dimensions_to_plot: List[int] = [0, 1],
    background=None,
) -> SVGDState:
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
    gradient_fn : callable
        Computes gradients of the negative log-probability. Accepts particles
        of shape ``(n_particles, n_dims)`` and returns gradients of the same
        shape. When ``data_sigma`` is set or ``estimate_sigma`` is ``True``,
        must return ``(gradients, misfits)`` where misfits has shape
        ``(n_particles,)``.
    n_iter : int
        Number of iterations.
    stepsize : float
        Base step size (interpretation depends on ``step_schedule``).
    bandwidth : float
        RBF kernel bandwidth. ``-1`` for automatic (median heuristic).
    preconditioner : str or None
        ``None`` for AdaGrad+momentum (legacy). ``"lbfgs"`` for per-particle
        L-BFGS preconditioning with Robbins-Monro step decay.
    lbfgs_history : int
        Number of curvature pairs to store per particle (only for ``"lbfgs"``).
    step_schedule : str or None
        ``None`` uses default for the preconditioner (AdaGrad for ``None``,
        Robbins-Monro for ``"lbfgs"``). ``"robbins-monro"`` uses
        ``stepsize / (|attractive|_max * sqrt(1+t))``. ``"constant"`` uses
        fixed ``stepsize``. ``"adagrad"`` uses AdaGrad+momentum.
    data_sigma : float or None
        Likelihood noise standard deviation. When set, gradients are scaled
        by ``1/sigma**2`` in the attractive SVGD term.
    estimate_sigma : bool
        If ``True``, update ``data_sigma`` at each iteration using a conjugate
        inverse-gamma posterior. Requires ``gradient_fn`` to return misfits.
    sigma_prior_alpha : float
        Shape parameter of the inverse-gamma prior on ``sigma**2``.
    sigma_prior_beta : float or None
        Scale parameter of the inverse-gamma prior. Defaults to
        ``(sigma_prior_alpha - 1) * data_sigma**2``.
    n_data_samples : int or None
        Total number of data samples (needed for hierarchical sigma update).
    kernel : str or None
        Kernel type. ``None`` or ``"rbf"`` for standard RBF. ``"rbf_normalized"``
        for per-dimension normalized RBF, recommended for high-dimensional
        parameter spaces (d > ~100) where standard RBF repulsion vanishes.
    bounds : tuple or None
        ``(lower, upper)`` bounds for particle clipping.
    callback : callable or None
        Called as ``callback(iteration, state)`` after each gradient evaluation.
    disable_progressbar : bool
        Suppress the tqdm progress bar.
    resume_from : SVGDState or None
        Resume from a previous run's state.
    animate : bool
        Legacy animation support (requires matplotlib).

    Returns
    -------
    SVGDState
        Final optimizer state. Access ``.particles`` for the particle array.
    """
    if x0 is None or gradient_fn is None:
        raise ValueError("x0 and gradient_fn cannot be None!")

    # Resolve kernel function
    if kernel is None or kernel == "rbf":
        kernel_fn = rbf_kernel
    elif kernel == "rbf_normalized":
        kernel_fn = rbf_kernel_normalized
    elif callable(kernel):
        kernel_fn = kernel
    else:
        raise ValueError(f"Unknown kernel: {kernel!r}")

    # Determine whether gradient_fn returns misfits
    needs_misfits = data_sigma is not None or estimate_sigma

    # Resolve step schedule
    if step_schedule is None:
        if preconditioner == "lbfgs":
            step_schedule = "robbins-monro"
        else:
            step_schedule = "adagrad"

    # Resolve preconditioner
    use_lbfgs = preconditioner == "lbfgs"

    # -------------------------------------------------------------------------
    # Initialize or resume state
    # -------------------------------------------------------------------------
    if resume_from is not None:
        particles = resume_from.particles.copy()
        n_particles = particles.shape[0]
        start_iter = resume_from.iteration
        sigma_history = list(resume_from.sigma_history)
        misfit_history = list(resume_from.misfit_history)
        particle_misfit_history = list(resume_from.particle_misfit_history)
        prev_particles = (
            resume_from.prev_particles.copy()
            if resume_from.prev_particles is not None
            else None
        )
        prev_grads = (
            resume_from.prev_grads.copy()
            if resume_from.prev_grads is not None
            else None
        )

        if use_lbfgs and resume_from.lbfgs_states is not None:
            lbfgs_states = resume_from.lbfgs_states
        elif use_lbfgs:
            lbfgs_states = [
                make_lbfgs_state(particles.shape[1], m=lbfgs_history, dtype=particles.dtype)
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

        current_sigma = resume_from.data_sigma if resume_from.data_sigma is not None else data_sigma
    else:
        particles = np.copy(x0)
        n_particles = particles.shape[0]
        start_iter = 0
        sigma_history = []
        misfit_history = []
        particle_misfit_history = []
        prev_particles = None
        prev_grads = None

        if use_lbfgs:
            lbfgs_states = [
                make_lbfgs_state(particles.shape[1], m=lbfgs_history, dtype=particles.dtype)
                for _ in range(n_particles)
            ]
        else:
            lbfgs_states = None

        historical_grad = np.zeros_like(particles) if step_schedule == "adagrad" else None
        current_sigma = data_sigma

    # Default sigma prior beta
    if estimate_sigma:
        if n_data_samples is None:
            raise ValueError("n_data_samples is required when estimate_sigma=True")
        if current_sigma is None:
            raise ValueError("data_sigma is required when estimate_sigma=True")
        if sigma_prior_beta is None:
            sigma_prior_beta = (sigma_prior_alpha - 1.0) * current_sigma**2

    # -------------------------------------------------------------------------
    # Animation setup (legacy)
    # -------------------------------------------------------------------------
    if animate:
        import matplotlib.pyplot as _plt

        if figure is None:
            figure = _plt.figure(figsize=(8, 8))
        axis = _plt.gca()

        if background is not None:
            x1s, x2s, background_image = background
            axis.contour(
                x1s, x2s, np.exp(-background_image), levels=20, alpha=0.5, zorder=0
            )

        scatter = axis.scatter(
            particles[:, dimensions_to_plot[0]],
            particles[:, dimensions_to_plot[1]],
        )

        if background is not None:
            _plt.xlim([x1s.min(), x1s.max()])
            _plt.ylim([x2s.min(), x2s.max()])

        axis.set_aspect(1)
        figure.canvas.draw()
        _plt.pause(1e-5)

    # -------------------------------------------------------------------------
    # AdaGrad parameters
    # -------------------------------------------------------------------------
    adagrad_alpha = 0.9
    adagrad_fudge = 1e-6

    # -------------------------------------------------------------------------
    # Main SVGD loop
    # -------------------------------------------------------------------------
    outer = _tqdm_auto.trange(
        n_iter, desc="SVGD", unit="iter", disable=disable_progressbar
    )
    try:
        for loop_iter in outer:
            iteration = start_iter + loop_iter

            # Evaluate gradients (and optionally misfits) at all particles
            result = gradient_fn(particles)
            if needs_misfits:
                all_grads, misfits = result
                misfits = np.asarray(misfits)
            else:
                all_grads = result
                misfits = None
            # gradient_fn is user-supplied and easy to write in a way that
            # silently returns float64 (e.g. building constants with
            # np.zeros(d) instead of matching particles' dtype) -- without
            # this, that alone would upcast the whole particle array on the
            # first `particles + displacement` below.
            all_grads = np.asarray(all_grads, dtype=particles.dtype)

            # Deferred L-BFGS curvature update
            if use_lbfgs and prev_particles is not None:
                for i in range(n_particles):
                    s_vec = particles[i] - prev_particles[i]
                    y_vec = all_grads[i] - prev_grads[i]
                    lbfgs_update(lbfgs_states[i], s_vec, y_vec)

            # Record misfits
            if misfits is not None:
                mean_misfit = float(np.mean(misfits))
                misfit_history.append(mean_misfit)
                particle_misfit_history.append(misfits.tolist())

            # Hierarchical sigma estimation
            if estimate_sigma and misfits is not None:
                raw_misfits_total = float(np.sum(misfits))
                alpha_post = sigma_prior_alpha + n_particles * n_data_samples / 2.0
                beta_post = sigma_prior_beta + raw_misfits_total
                sigma_sq = beta_post / (alpha_post - 1.0)
                current_sigma = float(np.sqrt(sigma_sq))

            if current_sigma is not None:
                sigma_history.append(current_sigma)

            # Progress bar
            postfix = {}
            if misfits is not None:
                postfix["misfit"] = f"{mean_misfit:.4e}"
            if current_sigma is not None:
                postfix["sigma"] = f"{current_sigma:.2e}"
            if postfix:
                outer.set_postfix(**postfix)

            # Callback
            if callback is not None:
                state_snapshot = SVGDState(
                    particles=particles.copy(),
                    iteration=iteration,
                    lbfgs_states=lbfgs_states,
                    historical_grad=historical_grad,
                    data_sigma=current_sigma,
                    sigma_history=list(sigma_history),
                    misfit_history=list(misfit_history),
                    particle_misfit_history=list(particle_misfit_history),
                    prev_particles=prev_particles,
                    prev_grads=prev_grads,
                )
                callback(iteration, state_snapshot)

            # Last iteration: don't update particles
            if loop_iter == n_iter - 1:
                break

            # Precondition gradients
            if use_lbfgs:
                precond_grads = np.zeros_like(particles)
                for i in range(n_particles):
                    # lbfgs_direction returns -H*g; negate to get H*g
                    precond_grads[i] = -lbfgs_direction(lbfgs_states[i], all_grads[i])
            else:
                precond_grads = all_grads

            # Compute kernel
            kxy, dxkxy = kernel_fn(particles, h=bandwidth)

            # Apply sigma scaling to the attractive term
            if current_sigma is not None:
                grads_scaled = precond_grads / (current_sigma**2)
            else:
                grads_scaled = precond_grads

            # SVGD update direction:
            #   phi = -(K @ grads_scaled - nabla_K) / n_particles
            attractive = np.matmul(kxy, grads_scaled)
            phi = -(attractive - dxkxy) / n_particles

            # Compute step size
            if step_schedule == "robbins-monro":
                attr_max = np.max(np.abs(attractive)) / n_particles
                # math.sqrt (not np.sqrt) on this plain-Python scalar: np.sqrt
                # of a bare int/float with no array context always returns
                # float64, which would silently upcast `step` and then
                # `displacement` even when phi is float32.
                decay = _math.sqrt(1.0 + iteration)
                if attr_max > 0:
                    step = stepsize / (attr_max * decay)
                else:
                    step = stepsize / decay
                displacement = step * phi

            elif step_schedule == "adagrad":
                # AdaGrad with momentum
                grad_theta = (np.matmul(kxy, precond_grads) - dxkxy) / n_particles
                if iteration == 0:
                    historical_grad = grad_theta**2
                else:
                    historical_grad = (
                        adagrad_alpha * historical_grad
                        + (1 - adagrad_alpha) * grad_theta**2
                    )
                adj_grad = grad_theta / (adagrad_fudge + np.sqrt(historical_grad))
                displacement = -stepsize * adj_grad

            elif step_schedule == "constant":
                displacement = stepsize * phi

            else:
                raise ValueError(f"Unknown step_schedule: {step_schedule!r}")

            # Store for deferred L-BFGS
            if use_lbfgs:
                prev_particles = particles.copy()
                prev_grads = all_grads.copy()

            # Update particles
            particles = particles + displacement

            # Enforce bounds
            if bounds is not None:
                particles = np.clip(particles, bounds[0], bounds[1])

            # Animation
            if animate:
                scatter.set_offsets(
                    np.hstack(
                        (
                            particles[:, dimensions_to_plot[0], None],
                            particles[:, dimensions_to_plot[1], None],
                        )
                    )
                )
                figure.canvas.draw()
                _plt.pause(1e-5)

    except KeyboardInterrupt:
        pass
    finally:
        outer.close()

    return SVGDState(
        particles=particles,
        iteration=start_iter + n_iter,
        lbfgs_states=lbfgs_states,
        historical_grad=historical_grad,
        data_sigma=current_sigma,
        sigma_history=sigma_history,
        misfit_history=misfit_history,
        particle_misfit_history=particle_misfit_history,
        prev_particles=prev_particles if use_lbfgs else None,
        prev_grads=prev_grads if use_lbfgs else None,
    )
