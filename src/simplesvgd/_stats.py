"""Shared per-iteration run stats, consumed by the progress bar and Rerun logging."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RunStats:
    """One iteration's live diagnostic values.

    ``mean_misfit``/``current_sigma`` are ``None`` when the run doesn't
    track them (no ``sigma`` config). ``repulsion_ratio`` is the previous
    iteration's value (this iteration's hasn't been computed yet when these
    are gathered) and is ``None`` on the very first iteration.
    """

    mean_misfit: float | None
    current_sigma: float | None
    particle_variance: float
    repulsion_ratio: float | None
