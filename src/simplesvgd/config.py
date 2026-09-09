"""SVGDConfig — a single dataclass grouping every tunable of :func:`update`."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic

from ._typing import Background, BatchIndices, FloatDType, KernelFn
from .state import SVGDState

if TYPE_CHECKING:
    from matplotlib.figure import Figure


@dataclass
class LBFGSConfig:
    """Per-particle L-BFGS preconditioning parameters.

    Only used when ``SVGDConfig.preconditioner == "lbfgs"``.

    Attributes
    ----------
    history : int
        Number of curvature pairs to store per particle.

    """

    history: int = 10


@dataclass
class SigmaConfig:
    """Hierarchical likelihood-noise (sigma) estimation.

    Attributes
    ----------
    value : float or None
        Likelihood noise standard deviation. When set, gradients are scaled
        by ``1/value**2`` in the attractive SVGD term.
    estimate : bool
        If ``True``, update ``value`` at each iteration using a conjugate
        inverse-gamma posterior. Requires ``gradient_fn`` to return misfits.
    prior_alpha : float
        Shape parameter of the inverse-gamma prior on ``value**2``.
    prior_beta : float or None
        Scale parameter of the inverse-gamma prior. Defaults to
        ``(prior_alpha - 1) * value**2``.
    n_data_samples : int or None
        Total number of data samples (needed when ``estimate=True``).

    """

    value: float | None = None
    estimate: bool = False
    prior_alpha: float = 2.0
    prior_beta: float | None = None
    n_data_samples: int | None = None


@dataclass
class AnimationConfig(Generic[FloatDType]):
    """Legacy live-scatter animation of the particle cloud (requires matplotlib).

    Attributes
    ----------
    enabled : bool
        Turn the animation on.
    figure : matplotlib.figure.Figure or None
        Figure to draw the animation on; created if not given.
    dimensions_to_plot : list[int]
        Which two particle dimensions to animate.
    background : tuple or None
        ``(x1s, x2s, background_image)`` contour data to draw behind the
        animation.

    """

    enabled: bool = False
    figure: "Figure | None" = None
    dimensions_to_plot: list[int] = field(default_factory=lambda: [0, 1])
    background: "Background[FloatDType] | None" = None


@dataclass
class SVGDConfig(Generic[FloatDType]):
    """Every tunable of :func:`update`, grouped into one object.

    All fields have defaults, so ``SVGDConfig()`` reproduces ``update()``'s
    previous behavior with no arguments. Pass a partially-filled instance to
    override just what you need, e.g. ``SVGDConfig(n_iter=500, bandwidth=2.0)``.
    Related tunables are grouped into sub-objects -- L-BFGS settings under
    ``lbfgs``, hierarchical noise estimation under ``sigma``, and the legacy
    animation under ``animation`` -- constructed the same way, e.g.
    ``SVGDConfig(sigma=SigmaConfig(value=0.1, estimate=True))``.

    Attributes
    ----------
    n_iter : int
        Number of iterations.
    stepsize : float
        Base step size (interpretation depends on ``step_schedule``).
    bandwidth : float
        RBF kernel bandwidth. ``-1`` for automatic (median heuristic).
    preconditioner : str or None
        ``None`` for AdaGrad+momentum (legacy). ``"lbfgs"`` for per-particle
        L-BFGS preconditioning with Robbins-Monro step decay.
    lbfgs : LBFGSConfig
        L-BFGS preconditioning parameters (only used for ``"lbfgs"``).
    step_schedule : str or None
        ``None`` uses the default for the preconditioner (AdaGrad for
        ``None``, Robbins-Monro for ``"lbfgs"``). ``"robbins-monro"`` uses
        ``stepsize / (|attractive|_max * sqrt(1+t))``. ``"constant"`` uses
        fixed ``stepsize``. ``"adagrad"`` uses AdaGrad+momentum.
    temperature_schedule : str, callable, or None
        Anneals the data-misfit gradient contribution: the (preconditioned,
        sigma-scaled) gradient is multiplied by a temperature in ``(0, 1]``
        before the SVGD attractive term is formed, so a temperature below 1
        reads as an inflated apparent noise level. Useful when the posterior
        is sharply peaked relative to the initial particle spread (e.g.
        full-waveform-inversion-style problems), where applying the full
        likelihood from iteration 0 risks collapsing particles onto a subset
        of modes before repulsion has had a chance to spread them out.
        ``None`` applies temperature 1.0 throughout (current behavior,
        unchanged). ``"linear"`` ramps linearly from a small floor to 1.0
        over the run. ``"geometric"`` ramps geometrically (log-linear) from
        the same floor to 1.0. A callable is called as
        ``temperature_schedule(iteration)`` and must return a float in
        ``(0, 1]``.
    sigma : SigmaConfig
        Hierarchical likelihood-noise estimation parameters.
    minibatch_sampler : callable or None
        Called as ``minibatch_sampler(iteration)``, returning an array of
        indices into the data dimension (e.g. sources/receivers) to use for
        that iteration. When set, ``gradient_fn`` is called as
        ``gradient_fn(particles, batch_indices)`` instead of
        ``gradient_fn(particles)``, and must accept that second argument.
        The returned gradients (and misfits, if ``sigma.value`` is set or
        ``sigma.estimate`` is ``True``) are treated as sums over just the
        sampled subset and rescaled by ``sigma.n_data_samples /
        len(batch_indices)`` -- an unbiased estimate of the full-dataset sum
        -- before use, so the SVGD attractive term and the hierarchical
        sigma estimate stay calibrated as if the full dataset had been
        evaluated. Requires ``sigma.n_data_samples`` to be set. ``None``
        (default) evaluates the full dataset every iteration, unchanged.
    kernel : str, KernelFn, or None
        Kernel type. ``None`` or ``"rbf"`` for standard RBF.
        ``"rbf_normalized"`` for per-dimension normalized RBF, recommended
        for high-dimensional parameter spaces (d > ~100) where standard RBF
        repulsion vanishes. Can also be a custom ``KernelFn``, e.g. one built
        with ``make_mass_weighted_kernel(weights)`` for particles
        representing a field discretized on a mesh or grid -- weighting
        distances by cell volume/quadrature weight keeps the kernel (and the
        resulting posterior statistics) stable as the mesh is refined,
        unlike plain Euclidean RBF.
    bounds : tuple[float, float] or None
        ``(lower, upper)`` bounds for particle clipping.
    callback : callable or None
        Called as ``callback(iteration, state)`` after each gradient
        evaluation.
    disable_progressbar : bool
        Suppress the tqdm progress bar.
    resume_from : SVGDState or None
        Resume from a previous run's state. When set, the run continues from
        ``resume_from.particles`` rather than ``x0``.
    animation : AnimationConfig
        Legacy live-scatter animation parameters.

    """

    n_iter: int = 1000
    stepsize: float = 1e-3
    bandwidth: float = -1
    preconditioner: str | None = None
    lbfgs: LBFGSConfig = field(default_factory=LBFGSConfig)
    step_schedule: str | None = None
    temperature_schedule: "str | Callable[[int], float] | None" = None
    sigma: SigmaConfig = field(default_factory=SigmaConfig)
    minibatch_sampler: Callable[[int], BatchIndices] | None = None
    kernel: "str | KernelFn[FloatDType] | None" = None
    bounds: tuple[float, float] | None = None
    callback: Callable[[int, SVGDState[FloatDType]], None] | None = None
    disable_progressbar: bool = False
    resume_from: SVGDState[FloatDType] | None = None
    animation: "AnimationConfig[FloatDType]" = field(default_factory=AnimationConfig)
