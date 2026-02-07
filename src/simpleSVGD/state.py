"""SVGDState dataclass — full optimizer state sufficient for resume."""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .lbfgs import LBFGSState


@dataclass
class SVGDState:
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

    particles: np.ndarray
    iteration: int = 0
    lbfgs_states: Optional[List[LBFGSState]] = None
    historical_grad: Optional[np.ndarray] = None
    data_sigma: Optional[float] = None
    sigma_history: List[float] = field(default_factory=list)
    misfit_history: List[float] = field(default_factory=list)
    particle_misfit_history: List[List[float]] = field(default_factory=list)
    prev_particles: Optional[np.ndarray] = None
    prev_grads: Optional[np.ndarray] = None
