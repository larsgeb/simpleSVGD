"""Shared legacy live-scatter animation helpers, used by update() and update_torch()."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic

import numpy as np
import numpy.typing as npt

from ._typing import Background, FloatDType

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.collections import PathCollection
    from matplotlib.figure import Figure


@dataclass
class Animation(Generic[FloatDType]):
    """Handles to the legacy live-scatter animation."""

    figure: "Figure"
    axis: "Axes"
    scatter: "PathCollection"


def setup_animation(
    *,
    figure: "Figure | None",
    background: Background[FloatDType] | None,
    particles: npt.NDArray[FloatDType],
    dimensions_to_plot: list[int],
) -> Animation[FloatDType]:
    import matplotlib.pyplot as plt  # noqa: PLC0415 -- matplotlib is an optional extra

    resolved_figure = figure if figure is not None else plt.figure(figsize=(8, 8))
    axis = plt.gca()

    if background is not None:
        x1s, x2s, background_image = background
        axis.contour(x1s, x2s, np.exp(-background_image), levels=20, alpha=0.5, zorder=0)

    scatter = axis.scatter(
        particles[:, dimensions_to_plot[0]], particles[:, dimensions_to_plot[1]]
    )

    if background is not None:
        x1s, x2s, _background_image = background
        plt.xlim(float(np.min(x1s)), float(np.max(x1s)))
        plt.ylim(float(np.min(x2s)), float(np.max(x2s)))

    axis.set_aspect(1)
    resolved_figure.canvas.draw()
    plt.pause(1e-5)
    return Animation(figure=resolved_figure, axis=axis, scatter=scatter)


def draw_frame(
    anim: Animation[FloatDType],
    particles: npt.NDArray[FloatDType],
    dimensions_to_plot: list[int],
) -> None:
    import matplotlib.pyplot as plt  # noqa: PLC0415 -- matplotlib is an optional extra

    anim.scatter.set_offsets(
        np.hstack(
            (
                particles[:, dimensions_to_plot[0], None],
                particles[:, dimensions_to_plot[1], None],
            )
        )
    )
    anim.figure.canvas.draw()
    plt.pause(1e-5)
