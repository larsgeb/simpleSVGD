"""SVGDConfig — a single dataclass grouping every tunable of :func:`update`."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic

from ._typing import Background, FloatDType, KernelFn
from .state import SVGDState

if TYPE_CHECKING:
    from matplotlib.figure import Figure


@dataclass
class SVGDConfig(Generic[FloatDType]):
    """Every tunable of :func:`update`, grouped into one object.

    All fields have defaults, so ``SVGDConfig()`` reproduces ``update()``'s
    previous behavior with no arguments. Pass a partially-filled instance to
    override just what you need, e.g. ``SVGDConfig(n_iter=500, bandwidth=2.0)``.

    Attributes:
        n_iter: Number of iterations.
        stepsize: Base step size (interpretation depends on ``step_schedule``).
        bandwidth: RBF kernel bandwidth. ``-1`` for automatic (median heuristic).
        preconditioner: ``None`` for AdaGrad+momentum (legacy). ``"lbfgs"`` for
            per-particle L-BFGS preconditioning with Robbins-Monro step decay.
        lbfgs_history: Number of curvature pairs to store per particle (only
            for ``"lbfgs"``).
        step_schedule: ``None`` uses the default for the preconditioner
            (AdaGrad for ``None``, Robbins-Monro for ``"lbfgs"``).
            ``"robbins-monro"`` uses ``stepsize / (|attractive|_max *
            sqrt(1+t))``. ``"constant"`` uses fixed ``stepsize``.
            ``"adagrad"`` uses AdaGrad+momentum.
        data_sigma: Likelihood noise standard deviation. When set, gradients
            are scaled by ``1/sigma**2`` in the attractive SVGD term.
        estimate_sigma: If ``True``, update ``data_sigma`` at each iteration
            using a conjugate inverse-gamma posterior. Requires
            ``gradient_fn`` to return misfits.
        sigma_prior_alpha: Shape parameter of the inverse-gamma prior on
            ``sigma**2``.
        sigma_prior_beta: Scale parameter of the inverse-gamma prior.
            Defaults to ``(sigma_prior_alpha - 1) * data_sigma**2``.
        n_data_samples: Total number of data samples (needed for hierarchical
            sigma update).
        kernel: Kernel type. ``None`` or ``"rbf"`` for standard RBF.
            ``"rbf_normalized"`` for per-dimension normalized RBF, recommended
            for high-dimensional parameter spaces (d > ~100) where standard
            RBF repulsion vanishes. Can also be a custom ``KernelFn``.
        bounds: ``(lower, upper)`` bounds for particle clipping.
        callback: Called as ``callback(iteration, state)`` after each
            gradient evaluation.
        disable_progressbar: Suppress the tqdm progress bar.
        resume_from: Resume from a previous run's state. When set, the run
            continues from ``resume_from.particles`` rather than ``x0``.
        animate: Legacy animation support (requires matplotlib).
        figure: Figure to draw the animation on; created if not given.
        dimensions_to_plot: Which two particle dimensions to animate.
        background: ``(x1s, x2s, background_image)`` contour data to draw
            behind the animation.

    """

    n_iter: int = 1000
    stepsize: float = 1e-3
    bandwidth: float = -1
    # --- Preconditioning ---
    preconditioner: str | None = None
    lbfgs_history: int = 10
    # --- Step schedule ---
    step_schedule: str | None = None
    # --- Hierarchical sigma ---
    data_sigma: float | None = None
    estimate_sigma: bool = False
    sigma_prior_alpha: float = 2.0
    sigma_prior_beta: float | None = None
    n_data_samples: int | None = None
    # --- Kernel ---
    kernel: "str | KernelFn[FloatDType] | None" = None
    # --- Bounds ---
    bounds: tuple[float, float] | None = None
    # --- Callback and control ---
    callback: Callable[[int, SVGDState[FloatDType]], None] | None = None
    disable_progressbar: bool = False
    # --- Resume ---
    resume_from: SVGDState[FloatDType] | None = None
    # --- Legacy animation ---
    animate: bool = False
    figure: "Figure | None" = None
    dimensions_to_plot: list[int] = field(default_factory=lambda: [0, 1])
    background: "Background[FloatDType] | None" = None
