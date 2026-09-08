"""SVGDState dataclass — full optimizer state sufficient for resume."""

from dataclasses import dataclass, field
from typing import Generic

import numpy.typing as npt

from ._typing import FloatDType
from .lbfgs import LBFGSState


@dataclass
class SVGDState(Generic[FloatDType]):
    """Complete state of an SVGD run.

    This object is returned by :func:`simpleSVGD.update` and can be passed
    back via the ``resume_from`` parameter to continue optimization.

    Attributes:
        particles: Current particle positions, shape ``(n_particles, n_dims)``.
        iteration: Total number of completed iterations.
        lbfgs_states: Per-particle L-BFGS states (``None`` when using AdaGrad).
        historical_grad: AdaGrad accumulator (``None`` when using L-BFGS).
        data_sigma: Current likelihood noise standard deviation.
        sigma_history: ``data_sigma`` at each iteration.
        misfit_history: Mean misfit across particles at each iteration.
        particle_misfit_history: Per-particle misfits at each iteration.
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
    prev_particles: npt.NDArray[FloatDType] | None = None
    prev_grads: npt.NDArray[FloatDType] | None = None
