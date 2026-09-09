"""SVGDState dataclass — full optimizer state sufficient for resume."""

from dataclasses import dataclass, field
from typing import Generic

import numpy.typing as npt

from ._typing import FloatDType
from .lbfgs import LBFGSState


@dataclass
class SVGDState(Generic[FloatDType]):
    """Complete state of an SVGD run.

    This object is returned by :func:`simplesvgd.update` and can be passed
    back via ``SVGDConfig(resume_from=...)`` to continue optimization.

    Attributes:
        particles: Current particle positions, shape ``(n_particles, n_dims)``.
        iteration: Total number of completed iterations.
        lbfgs_states: Per-particle L-BFGS states (``None`` when using AdaGrad).
        historical_grad: AdaGrad accumulator (``None`` when using L-BFGS).
        data_sigma: Current likelihood noise standard deviation.
        sigma_history: ``data_sigma`` at each iteration.
        misfit_history: Mean misfit across particles at each iteration.
        particle_misfit_history: Per-particle misfits at each iteration.
        particle_variance_history: Total particle-ensemble variance (trace of
            the empirical covariance, i.e. sum of per-dimension variances) at
            each iteration -- a cheap proxy for ensemble spread. A value that
            shrinks steadily over the run, well below what the target
            distribution's actual variance should be, is a variance-collapse
            warning sign (see Ba et al., "Understanding the Variance Collapse
            of SVGD in High Dimensions", ICLR 2022).
        repulsion_ratio_history: Ratio of the repulsive kernel-gradient term's
            norm to the attractive term's norm, at each iteration a
            displacement is computed (shorter than the other histories -- the
            final iteration of any given ``update()`` call only records
            state, it doesn't step, so each ``resume_from`` boundary drops
            one more entry than the other histories accumulate). Read
            this early in a run, not as a monotonic trend over the whole
            run -- the attractive term shrinks toward zero near any converged
            mode regardless of collapse, which swamps the ratio's trend late
            in a run. A ratio far below 1 in the first few iterations (while
            particles are still diffuse and attraction hasn't decayed yet) is
            the direct, cheap signature of the collapse mechanism in the
            paper above: repulsion already overwhelmed by attraction before
            it's had any chance to spread the ensemble out.
        prev_particles: Previous particle positions (for deferred L-BFGS update).
        prev_grads: Previous gradients (for deferred L-BFGS update).

    """

    particles: npt.NDArray[FloatDType]
    iteration: int = 0
    lbfgs_states: list[LBFGSState[FloatDType]] | None = None
    historical_grad: npt.NDArray[FloatDType] | None = None
    data_sigma: float | None = None
    sigma_history: list[float] = field(default_factory=list)
    misfit_history: list[float] = field(default_factory=list)
    particle_misfit_history: list[list[float]] = field(default_factory=list)
    particle_variance_history: list[float] = field(default_factory=list)
    repulsion_ratio_history: list[float] = field(default_factory=list)
    prev_particles: npt.NDArray[FloatDType] | None = None
    prev_grads: npt.NDArray[FloatDType] | None = None
