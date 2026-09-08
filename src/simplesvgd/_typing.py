"""Shared typing helpers.

``FloatDType`` is the generic dtype parameter used throughout this package:
functions and containers parametrized over it preserve whichever floating
dtype (``np.float32``, ``np.float64``, ...) the caller passes in, rather than
committing to one at the type level.
"""

from typing import Any, TypeVar

import numpy as np

FloatDType = TypeVar("FloatDType", bound=np.floating[Any])
