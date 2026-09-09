"""Rich-based live progress display for update()'s run loop."""

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


def make_progress(*, disable: bool, console: Console | None = None) -> Progress:
    """Build the live SVGD progress bar.

    Spinner, bar, percentage, iteration count, elapsed/remaining time -- the
    usual set -- plus a trailing stats field (``format_stats``) updated once
    per iteration with the run's live misfit/sigma/diagnostic values.
    ``console`` is exposed for tests to render into an in-memory buffer
    instead of the real terminal.
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TextColumn("[dim]•"),
        TimeElapsedColumn(),
        TextColumn("[dim]<"),
        TimeRemainingColumn(),
        TextColumn("{task.fields[stats]}", style="dim"),
        disable=disable,
        console=console,
    )


def format_stats(
    mean_misfit: float | None,
    current_sigma: float | None,
    particle_variance: float,
    repulsion_ratio: float | None,
) -> str:
    """Render one line of live run stats: misfit, sigma, particle variance, repulsion ratio.

    ``mean_misfit``/``current_sigma`` are omitted when the run doesn't track
    them (no ``sigma`` config). ``repulsion_ratio`` is the previous
    iteration's value -- this iteration's hasn't been computed yet when the
    progress bar updates -- and is omitted on the very first iteration, before
    any displacement has been taken.
    """
    parts = []
    if mean_misfit is not None:
        parts.append(f"misfit={mean_misfit:.3e}")
    if current_sigma is not None:
        parts.append(f"sigma={current_sigma:.2e}")
    parts.append(f"var={particle_variance:.3e}")
    if repulsion_ratio is not None:
        parts.append(f"rep={repulsion_ratio:.2f}")
    return "  " + "  ".join(parts) if parts else ""
