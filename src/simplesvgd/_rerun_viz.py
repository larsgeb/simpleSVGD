"""Live particle + diagnostics logging to the Rerun viewer.

Unlike ``_animation.py``'s static matplotlib scatter, this logs a scrubbable
timeline: particle positions and the run's scalar diagnostics (misfit,
sigma, particle variance, repulsion ratio) each iteration, so you can pause
and rewind through a run in the Rerun viewer to see exactly how a knob
(stepsize, kernel, annealing schedule, ...) changed its behavior -- not just
inspect the final state.
"""

from dataclasses import dataclass
from typing import Generic
from uuid import uuid4

import numpy.typing as npt

from ._stats import RunStats
from ._typing import FloatDType


@dataclass
class RerunSession(Generic[FloatDType]):
    """Handle to one ``update()`` call's Rerun recording."""

    dimensions_to_plot: list[int]


def setup_rerun(
    *, application_id: str, spawn: bool, dimensions_to_plot: list[int]
) -> RerunSession[FloatDType]:
    """Start a fresh Rerun recording for one ``update()`` run.

    Each call gets its own ``recording_id`` (rather than rerun's per-process
    default) so that running ``update()`` more than once in the same
    process -- e.g. comparing two configs in a notebook -- logs to separate
    recordings instead of overlaying both runs onto the same timeline. A
    ``resume_from`` continuation currently starts a new recording too (its
    ``iteration`` numbering still continues correctly, it just won't stitch
    onto the original run's recording in the viewer).
    """
    import rerun as rr  # noqa: PLC0415 -- rerun is an optional extra  # ty: ignore[unresolved-import]

    rr.init(application_id, recording_id=uuid4(), spawn=spawn)
    return RerunSession(dimensions_to_plot=dimensions_to_plot)


def log_iteration(
    session: RerunSession[FloatDType],
    iteration: int,
    particles: npt.NDArray[FloatDType],
    stats: RunStats,
) -> None:
    """Log one iteration's particle positions and scalar diagnostics."""
    import rerun as rr  # noqa: PLC0415 -- rerun is an optional extra  # ty: ignore[unresolved-import]

    rr.set_time("iteration", sequence=iteration)
    d0, d1 = session.dimensions_to_plot
    rr.log("particles", rr.Points2D(particles[:, [d0, d1]]))
    if stats.mean_misfit is not None:
        rr.log("diagnostics/misfit", rr.Scalars(stats.mean_misfit))
    if stats.current_sigma is not None:
        rr.log("diagnostics/sigma", rr.Scalars(stats.current_sigma))
    rr.log("diagnostics/particle_variance", rr.Scalars(stats.particle_variance))
    if stats.repulsion_ratio is not None:
        rr.log("diagnostics/repulsion_ratio", rr.Scalars(stats.repulsion_ratio))


def maybe_log_iteration(
    session: RerunSession[FloatDType] | None,
    iteration: int,
    particles: npt.NDArray[FloatDType],
    stats: RunStats,
) -> None:
    """Log one iteration if ``session`` is active; a no-op otherwise (rerun disabled)."""
    if session is not None:
        log_iteration(session, iteration, particles, stats)
