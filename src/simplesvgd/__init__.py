"""simplesvgd — Stein Variational Gradient Descent.

A lightweight, pure-NumPy implementation of the SVGD algorithm for Bayesian
inference and distribution approximation.  Supports optional L-BFGS
preconditioning, hierarchical noise estimation, bounds, callbacks, and resume.
"""

from collections.abc import Callable
from time import sleep
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import numpy.typing as npt
import tqdm.auto as tqdm_auto

from ._animation import draw_frame, setup_animation
from ._typing import Background, FloatDType
from .config import SVGDConfig
from .kernels import rbf_kernel, rbf_kernel_normalized
from .lbfgs import LBFGSState, lbfgs_direction, lbfgs_update, make_lbfgs_state
from .state import SVGDState
from .update import update

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__version__ = "1.0.0"

__all__ = [
    "LBFGSState",
    "SVGDConfig",
    "SVGDState",
    "gradient_vectorizer",
    "lbfgs_direction",
    "lbfgs_update",
    "make_lbfgs_state",
    "rbf_kernel_normalized",
    "update",
    "update_torch",
]


class _TorchOptimizerLike(Protocol):
    """Structural type for the torch.optim.Optimizer methods update_torch actually calls."""

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...  # noqa: ANN401 -- constructor args vary per torch optimizer class
    def zero_grad(self) -> None: ...
    def step(self, closure: Callable[[], Any]) -> Any: ...  # noqa: ANN401 -- torch.Tensor loss, unavailable without installing torch for type-checking


class _TorchSchedulerLike(Protocol):
    """Structural type for a torch LR scheduler's step() method."""

    def step(self) -> None: ...


def update_torch(  # noqa: PLR0913 -- legacy PyTorch bridge; not covered by update()'s SVGDConfig redesign
    x0: npt.NDArray[FloatDType],
    gradient_fn: Callable[[npt.NDArray[FloatDType]], npt.NDArray[FloatDType]],
    optimizer_class: type[_TorchOptimizerLike],
    optimizer_parameters: dict[str, Any] | None = None,
    schedulers: list[_TorchSchedulerLike] | None = None,
    *,
    n_iter: int = 1000,
    animate: bool = False,
    figure: "Figure | None" = None,
    dimensions_to_plot: list[int] | None = None,
    background: Background[FloatDType] | None = None,
    disable_progressbar: bool = False,
) -> npt.NDArray[FloatDType]:
    """Update samples using SVGD with a PyTorch optimizer.

    Parameters
    ----------
    x0 : np.ndarray
        Initial samples, shape ``(n_samples, dimensionality)``.
    gradient_fn : callable
        Computes gradients of the negative log-probability.
    optimizer_class : torch.optim.Optimizer subclass
        PyTorch optimizer to use.
    optimizer_parameters : dict or None
        Keyword arguments forwarded to the optimizer constructor.
    schedulers : list or None
        Learning rate schedulers to step after each iteration.
    n_iter : int
        Number of iterations.
    animate : bool
        Enable 2D scatter animation.
    figure : matplotlib Figure or None
        Figure to draw the animation on; created if not given.
    dimensions_to_plot : list of int or None
        Which two particle dimensions to animate. Defaults to ``[0, 1]``.
    background : tuple or None
        ``(x1s, x2s, background_image)`` contour data to draw behind the
        animation.
    disable_progressbar : bool
        Suppress the tqdm progress bar.

    """
    import torch  # noqa: PLC0415 -- torch is an optional extra  # ty: ignore[unresolved-import]

    from .helpers import torch_wrapper  # noqa: PLC0415 -- only needed for this optional path

    if x0 is None or gradient_fn is None:
        raise ValueError("x0 or gradient_fn cannot be None!")

    if optimizer_parameters is None:
        optimizer_parameters = {}
    if schedulers is None:
        schedulers = []
    if dimensions_to_plot is None:
        dimensions_to_plot = [0, 1]

    x0_updated = np.copy(x0)

    anim = (
        setup_animation(
            figure=figure,
            background=background,
            particles=x0_updated,
            dimensions_to_plot=dimensions_to_plot,
        )
        if animate
        else None
    )

    x0_updated_tensor = torch.tensor(x0_updated, requires_grad=True)
    total = torch_wrapper(gradient_fn, rbf_kernel)
    optimizer = optimizer_class([x0_updated_tensor], **optimizer_parameters)

    try:
        for _ in tqdm_auto.trange(n_iter, disable=disable_progressbar):
            def closure() -> Any:  # noqa: ANN401 -- torch.Tensor loss, unavailable without installing torch for type-checking
                optimizer.zero_grad()
                loss = total(x0_updated_tensor).mean()
                loss.backward()
                return loss

            optimizer.step(closure)

            for scheduler in schedulers:
                scheduler.step()

            if anim is not None:
                draw_frame(anim, x0_updated_tensor.detach().numpy(), dimensions_to_plot)

    except KeyboardInterrupt:
        sleep(0.5)

    return x0_updated_tensor.detach().numpy()


def gradient_vectorizer(
    non_vectorized_gradient: Callable[[npt.NDArray[FloatDType]], npt.NDArray[FloatDType]],
) -> Callable[[npt.NDArray[FloatDType]], npt.NDArray[FloatDType]]:
    """Wrap a single-point gradient function to accept batched inputs."""
    def grd(m: npt.NDArray[FloatDType]) -> npt.NDArray[FloatDType]:
        return np.hstack(
            [non_vectorized_gradient(m[idm, :, None]) for idm in range(m.shape[0])]
        ).T
    return grd
