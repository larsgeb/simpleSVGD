"""Matrix-free, mean-field Stein Variational Newton (SVN) preconditioning.

Full SVN (Detommaso, Cui, Marzouk & Scheichl, "A Stein variational Newton
method", NeurIPS 2018) builds a *distinct* Newton system per particle, from
that particle's own Hessian plus kernel-weighted cross terms with every
other particle's Hessian -- O(n) dense Hessians and an O(n^2) cross-term
computation per iteration. That's not tractable here: the whole point of
only exposing curvature through a Hessian-vector product (matching what an
adjoint-state solver actually gives you, e.g. in full-waveform inversion) is
to avoid ever forming a dense ``d x d`` matrix, and the paper's per-particle
distinct-Hessian system would need its own Krylov solve per particle against
a kernel-weighted sum of every *other* particle's Hessian-vector product --
back to O(n^2) Hessian-vector products per SVGD iteration, however it's
arranged.

What's implemented here is a much cheaper "mean-field" variant instead: a
single shared curvature operator -- the (damped) Hessian-vector product
evaluated once at the particle ensemble's mean position -- used to
Newton-precondition every particle's already kernel-combined SVGD direction
(``phi``, attraction and repulsion together; unlike L-BFGS, which only
preconditions the raw gradient *before* kernel combination). Solving
``M @ v_i = phi_i`` for all n particles against that one shared operator is
done with a single vectorized conjugate-gradient loop (independent per-row
step sizes, not a shared Krylov subspace in the formal "block CG" sense), so
the whole SVGD iteration costs ``cg_iters`` calls to
``hessian_vector_product`` -- each batched over all n particles' vectors at
once -- rather than ``cg_iters * n`` or ``cg_iters * n**2``.

This is *not* the paper's algorithm: one shared operator stands in for n
distinct per-particle Hessians, and there are no kernel-weighted cross
terms, so it only attacks the drift/preconditioning side of variance
collapse (the same role L-BFGS plays), not the kernel/repulsion side. Refer
to it as "mean-field SVN", not "SVN", in anything user-facing.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import numpy as np
import numpy.typing as npt

from ._typing import FloatDType, HessianVectorProductFn

if TYPE_CHECKING:
    from .config import SVNConfig

# Guards the CG step-size divisions against an exact zero denominator (a
# particle whose residual or search direction is already exactly zero) --
# not a convergence tolerance.
_CG_EPS = 1e-30


def _batched_cg(
    apply_operator: Callable[[npt.NDArray[FloatDType]], npt.NDArray[FloatDType]],
    b: npt.NDArray[FloatDType],
    cg_iters: int,
) -> npt.NDArray[FloatDType]:
    """Solve ``apply_operator(v) == b`` for every row of *b* at once.

    *b* has shape ``(n_particles, n_dims)``; each row is an independent CG
    solve against the same shared linear operator, with per-row (not
    per-matrix) step sizes -- vectorized for one ``apply_operator`` call per
    iteration, not true block CG (no shared Krylov subspace).
    """
    v = np.zeros_like(b)
    r = b.copy()
    p = r.copy()
    rs_old = np.sum(r * r, axis=1)

    for _ in range(cg_iters):
        if bool(np.all(rs_old <= _CG_EPS)):
            break
        ap = apply_operator(p)
        pap = np.sum(p * ap, axis=1)
        alpha = rs_old / (pap + _CG_EPS)
        v = cast("npt.NDArray[FloatDType]", v + alpha[:, None] * p)
        r = cast("npt.NDArray[FloatDType]", r - alpha[:, None] * ap)
        rs_new = np.sum(r * r, axis=1)
        beta = rs_new / (rs_old + _CG_EPS)
        p = cast("npt.NDArray[FloatDType]", r + beta[:, None] * p)
        rs_old = rs_new

    return v


def svn_direction(
    hessian_vector_product: HessianVectorProductFn[FloatDType],
    particles: npt.NDArray[FloatDType],
    phi: npt.NDArray[FloatDType],
    svn_config: "SVNConfig",
) -> npt.NDArray[FloatDType]:
    """Newton-precondition *phi* with the shared, mean-field curvature operator.

    Solves ``(H_mean + damping * I) @ v_i = phi_i`` for every particle *i*
    via batched CG, where ``H_mean`` is *hessian_vector_product* evaluated at
    the particle ensemble's mean position (the same operator for every
    particle). *phi* is the already kernel-combined SVGD direction
    (attraction and repulsion together), shape ``(n_particles, n_dims)``.
    """
    n_particles = particles.shape[0]
    mean_particle = np.mean(particles, axis=0, keepdims=True)
    mean_particles = np.tile(mean_particle, (n_particles, 1)).astype(particles.dtype)
    damping = svn_config.damping

    def apply_operator(v: npt.NDArray[FloatDType]) -> npt.NDArray[FloatDType]:
        hv = np.asarray(hessian_vector_product(mean_particles, v), dtype=phi.dtype)
        return cast("npt.NDArray[FloatDType]", hv + damping * v)

    return _batched_cg(apply_operator, phi, svn_config.cg_iters)
