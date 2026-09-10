"""Shared typing helpers.

``FloatDType`` is the generic dtype parameter used throughout this package:
functions and containers parametrized over it preserve whichever floating
dtype (``np.float32``, ``np.float64``, ...) the caller passes in, rather than
committing to one at the type level.

``KernelFn``, ``GradientFn``, ``HessianVectorProductFn`` and ``Background``
live here (rather than in ``update.py``, where they're used) so that both
``update.py`` and ``config.py`` can import them without a circular import.
"""

from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt

FloatDType = TypeVar("FloatDType", bound=np.floating[Any])

KernelFn = Callable[
    [npt.NDArray[FloatDType], float],
    tuple[npt.NDArray[FloatDType], npt.NDArray[FloatDType]],
]
GradientFn = Callable[
    [npt.NDArray[FloatDType]],
    npt.NDArray[FloatDType] | tuple[npt.NDArray[FloatDType], npt.NDArray[FloatDType]],
]
HessianVectorProductFn = Callable[
    [npt.NDArray[FloatDType], npt.NDArray[FloatDType]],
    npt.NDArray[FloatDType],
]
BatchIndices = npt.NDArray[np.integer[Any]]
MinibatchGradientFn = Callable[
    [npt.NDArray[FloatDType], BatchIndices],
    npt.NDArray[FloatDType] | tuple[npt.NDArray[FloatDType], npt.NDArray[FloatDType]],
]
Background = tuple[npt.NDArray[FloatDType], npt.NDArray[FloatDType], npt.NDArray[FloatDType]]
