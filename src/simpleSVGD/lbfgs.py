"""L-BFGS inverse Hessian approximation via two-loop recursion.

Provides a lightweight, pure-NumPy implementation of the limited-memory BFGS
algorithm for use as a preconditioner in SVGD or as a standalone optimizer.
"""

from dataclasses import dataclass
from typing import Generic, overload

import numpy as np
import numpy.typing as npt

from ._typing import FloatDType


@dataclass
class LBFGSState(Generic[FloatDType]):
    """Circular buffer storing L-BFGS curvature pairs (s, y).

    Attributes:
        S: Array of shape (m, n) storing s_k = x_{k+1} - x_k vectors.
        Y: Array of shape (m, n) storing y_k = g_{k+1} - g_k vectors.
        cursor: Index where the next pair will be written.
        count: Number of pairs stored so far (up to m).
    """

    S: npt.NDArray[FloatDType]
    Y: npt.NDArray[FloatDType]
    cursor: int = 0
    count: int = 0


@overload
def make_lbfgs_state(n: int, m: int = 10) -> LBFGSState[np.float64]: ...
@overload
def make_lbfgs_state(
    n: int, m: int = 10, *, dtype: np.dtype[FloatDType]
) -> LBFGSState[FloatDType]: ...
def make_lbfgs_state(
    n: int, m: int = 10, *, dtype: np.dtype[FloatDType] | type[np.float64] = np.float64
) -> LBFGSState[FloatDType] | LBFGSState[np.float64]:
    """Create an empty L-BFGS state with history size *m* for vectors of length *n*.

    *dtype* should match the dtype of the gradients/particles this state
    will be used with (e.g. ``particles.dtype``), so the two-loop recursion
    in :func:`lbfgs_direction` doesn't get upcast by a mismatched buffer
    dtype. Defaults to float64 when omitted.
    """
    return LBFGSState(
        S=np.zeros((m, n), dtype=dtype),
        Y=np.zeros((m, n), dtype=dtype),
        cursor=0,
        count=0,
    )


def lbfgs_update(
    state: LBFGSState[FloatDType], s: npt.NDArray[FloatDType], y: npt.NDArray[FloatDType]
) -> None:
    """Push a new (s, y) pair into the circular buffer.

    Skips the update if the curvature condition y.s > 0 is not satisfied.
    """
    ys = float(np.dot(s, y))
    if ys <= 0:
        return
    m = state.S.shape[0]
    idx = state.cursor % m
    state.S[idx] = s
    state.Y[idx] = y
    state.cursor = (idx + 1) % m
    state.count = min(state.count + 1, m)


def lbfgs_direction(
    state: LBFGSState[FloatDType], grad: npt.NDArray[FloatDType]
) -> npt.NDArray[FloatDType]:
    """Compute the L-BFGS search direction via two-loop recursion.

    Returns ``-H_k @ grad`` where H_k is the L-BFGS approximation to the
    inverse Hessian.  Falls back to ``-grad`` when the history is empty.
    """
    k = state.count
    if k == 0:
        return -grad.copy()

    m = state.S.shape[0]
    # Indices from newest to oldest
    indices = [(state.cursor - 1 - i) % m for i in range(k)]

    q = grad.copy()
    # dtype=grad.dtype matters: a plain np.zeros(k) defaults to float64, and
    # a float64 numpy scalar pulled from it would upcast every float32 array
    # it later touches (unlike a plain Python float, which stays "weak" and
    # doesn't force a promotion).
    alphas = np.zeros(k, dtype=grad.dtype)
    rhos = np.zeros(k, dtype=grad.dtype)

    # Forward pass (newest to oldest)
    for j, idx in enumerate(indices):
        s_j = state.S[idx]
        y_j = state.Y[idx]
        rho_j = 1.0 / np.dot(y_j, s_j)
        rhos[j] = rho_j
        alpha_j = rho_j * np.dot(s_j, q)
        alphas[j] = alpha_j
        q = q - alpha_j * y_j

    # Initial Hessian scaling: H0 = (y_k . s_k) / (y_k . y_k) * I
    newest = indices[0]
    s_newest = state.S[newest]
    y_newest = state.Y[newest]
    gamma = np.dot(y_newest, s_newest) / np.dot(y_newest, y_newest)
    r = gamma * q

    # Backward pass (oldest to newest)
    for j in reversed(range(k)):
        idx = indices[j]
        y_j = state.Y[idx]
        beta = rhos[j] * np.dot(y_j, r)
        r = r + (alphas[j] - beta) * state.S[idx]

    return -r
