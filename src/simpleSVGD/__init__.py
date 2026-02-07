"""simpleSVGD — Stein Variational Gradient Descent.

A lightweight, pure-NumPy implementation of the SVGD algorithm for Bayesian
inference and distribution approximation.  Supports optional L-BFGS
preconditioning, hierarchical noise estimation, bounds, callbacks, and resume.
"""

from time import sleep

import numpy as _numpy
import tqdm.auto as _tqdm_auto
from typing import Callable as _Callable, List as _List, Tuple as _Tuple

from .kernels import rbf_kernel as _rbf_kernel, rbf_kernel_normalized
from .lbfgs import LBFGSState, lbfgs_direction, lbfgs_update, make_lbfgs_state
from .state import SVGDState
from .update import update

__version__ = "1.0.0"


def update_torch(
    x0: _numpy.array,
    gradient_fn: _Callable,
    optimizer_class,
    optimizer_parameters={},
    schedulers=[],
    n_iter: int = 1000,
    animate: bool = False,
    figure=None,
    dimensions_to_plot: _List[float] = [0, 1],
    background: _Tuple[_numpy.array] = None,
    disable_progressbar: bool = False,
):
    """Update samples using SVGD with a PyTorch optimizer.

    Parameters
    ----------
    x0 : np.ndarray
        Initial samples, shape ``(n_samples, dimensionality)``.
    gradient_fn : callable
        Computes gradients of the negative log-probability.
    optimizer_class : torch.optim.Optimizer subclass
        PyTorch optimizer to use.
    optimizer_parameters : dict
        Keyword arguments forwarded to the optimizer constructor.
    schedulers : list
        Learning rate schedulers to step after each iteration.
    n_iter : int
        Number of iterations.
    animate : bool
        Enable 2D scatter animation.
    """
    import torch as _torch
    import matplotlib.pyplot as _plt
    from .helpers import TorchWrapper as _TorchWrapper

    if x0 is None or gradient_fn is None:
        raise ValueError("x0 or gradient_fn cannot be None!")

    if animate:
        if figure is None:
            figure = _plt.figure(figsize=(8, 8))
        axis = _plt.gca()

    x0_updated = _numpy.copy(x0)

    if animate:
        if background is not None:
            x1s, x2s, background_image = background
            axis.contour(
                x1s, x2s, _numpy.exp(-background_image),
                levels=20, alpha=0.5, zorder=0,
            )

        scatter = axis.scatter(
            x0_updated[:, dimensions_to_plot[0]],
            x0_updated[:, dimensions_to_plot[1]],
        )

        if background is not None:
            _plt.xlim([x1s.min(), x1s.max()])
            _plt.ylim([x2s.min(), x2s.max()])

        axis.set_aspect(1)
        figure.canvas.draw()
        _plt.pause(1e-5)

    x0_updated = _torch.tensor(x0_updated, requires_grad=True)
    total = _TorchWrapper(gradient_fn, _rbf_kernel)
    optimizer = optimizer_class([x0_updated], **optimizer_parameters)

    try:
        for _ in _tqdm_auto.trange(n_iter, disable=disable_progressbar):
            def closure():
                optimizer.zero_grad()
                loss = total(x0_updated).mean()
                loss.backward()
                return loss

            optimizer.step(closure)

            for scheduler in schedulers:
                scheduler.step()

            if animate:
                scatter.set_offsets(
                    _numpy.hstack((
                        x0_updated.detach()[:, dimensions_to_plot[0], None],
                        x0_updated.detach()[:, dimensions_to_plot[1], None],
                    ))
                )
                figure.canvas.draw()
                _plt.pause(1e-5)

    except KeyboardInterrupt:
        sleep(0.5)

    return x0_updated.detach().numpy()


def gradient_vectorizer(non_vectorized_gradient: _Callable):
    """Wrap a single-point gradient function to accept batched inputs."""
    def grd(m: _numpy.array) -> _numpy.array:
        return _numpy.hstack(
            [non_vectorized_gradient(m[idm, :, None]) for idm in range(m.shape[0])]
        ).T
    return grd
