# PoC: profile mean-field SVN vs per-particle SVN vs compact-kernel exact
# cross-term SVN vs dense-exact cross-term SVN vs baseline/L-BFGS, on a
# high-dimensional, position-dependent-curvature target.
#
# Target: elementwise nonlinear (Gauss-Newton) model, separable across
# dimensions, f(x_k) = x_k + a*sin(x_k), target y_k=0, so the true log-density
# is NOT quadratic and the Hessian genuinely varies with particle position
# (unlike a plain Gaussian, where mean-field/per-particle/exact cross-term
# are all identical since H is spatially constant). Separability means the
# true per-dimension variance is available via cheap 1D quadrature -- a real
# ground truth to measure variance-collapse against, not just a proxy.
#
# Run from repo root: uv run python docs/assets/benchmarks/svn_variants.py

import time

import numpy as np

from simplesvgd.kernels import rbf_kernel_normalized
from simplesvgd.lbfgs import lbfgs_direction, lbfgs_update, make_lbfgs_state
from simplesvgd.svn import _batched_cg  # noqa: PLC2701 -- reusing internal CG for a PoC, not shipped code

# ---------------------------------------------------------------------------
# Target: separable nonlinear Gauss-Newton model. f(x) = x + a*sin(x) keeps
# f'(x) in [1-a, 1+a] -- bounded, position-dependent curvature (like FWI's
# locally-varying sensitivity) without the misfit ever exploding (f(x) ~ x
# for large |x|, same tail growth as a plain Gaussian, so a broad initial
# particle spread stays numerically safe for every variant, including
# baseline/L-BFGS).
# ---------------------------------------------------------------------------
_A = 0.5
_SIGMA = 1.0


def f_of(x):
    return x + _A * np.sin(x)


def fprime_of(x):
    return 1.0 + _A * np.cos(x)


def grad_fn(x):
    r = f_of(x)
    return r * fprime_of(x) / _SIGMA**2


def hvp_raw(particles, vectors):
    """Gauss-Newton (diagonal) Hessian-vector product at each row's own position."""
    return (fprime_of(particles) ** 2 / _SIGMA**2) * vectors


def true_marginal_variance():
    """1D quadrature ground truth for Var[x_k] under p(x_k) ~ exp(-0.5 f(x_k)^2)."""
    grid = np.linspace(-6, 6, 200_001)
    log_dens = -0.5 * f_of(grid) ** 2 / _SIGMA**2
    log_dens -= log_dens.max()
    dens = np.exp(log_dens)
    z = np.trapezoid(dens, grid)
    mean = np.trapezoid(dens * grid, grid) / z
    var = np.trapezoid(dens * (grid - mean) ** 2, grid) / z
    return var


# ---------------------------------------------------------------------------
# Hvp call/row counter -- the real FWI cost proxy (1 row ~ 1 adjoint solve)
# ---------------------------------------------------------------------------
class HvpCounter:
    def __init__(self, raw_fn):
        self.raw_fn = raw_fn
        self.calls = 0
        self.rows = 0

    def __call__(self, particles, vectors):
        self.calls += 1
        self.rows += vectors.shape[0]
        return self.raw_fn(particles, vectors)

    def reset(self):
        self.calls = 0
        self.rows = 0


# ---------------------------------------------------------------------------
# Newton-direction operators (all share the same batched-CG solver)
# ---------------------------------------------------------------------------
def mean_field_operator(hvp_counted, particles, damping):
    n = particles.shape[0]
    mean_particles = np.tile(np.mean(particles, axis=0, keepdims=True), (n, 1))

    def apply(v):
        return hvp_counted(mean_particles, v) + damping * v

    return apply


def per_particle_operator(hvp_counted, particles, damping):
    def apply(v):
        return hvp_counted(particles, v) + damping * v

    return apply


def _crossterm_operator(hvp_counted, particles, weights, damping):
    """result[i] = sum_j weights[i,j] * H_j(v[i]) + damping*v[i].

    Grouped by owner j (one batched Hvp call per particle whose Hessian is
    used, covering every i that needs it) -- the "batch by owner, not by
    solver" trick that keeps this O(n) or O(n*neighbors) hvp_fn calls
    instead of O(n^2) independent calls.
    """
    n = particles.shape[0]

    def apply(v):
        result = np.zeros_like(v)
        for j in range(n):
            col = weights[:, j]
            mask = col > 0
            if not np.any(mask):
                continue
            owner = np.tile(particles[j : j + 1], (int(mask.sum()), 1))
            hj_v = hvp_counted(owner, v[mask])
            result[mask] += col[mask, None] * hj_v
        return result + damping * v

    return apply


def dense_exact_operator(hvp_counted, particles, kernel_matrix, damping):
    # NOT dividing by n here: phi (the RHS) already carries its own /n from
    # the standard SVGD formula, and an *additional* /n on the operator
    # breaks the implicit A/phi scale-cancellation (while a fixed damping
    # constant doesn't rescale along with it) rather than matching the
    # paper's intent -- confirmed empirically: adding /n here made the
    # solve blow up harder, not better.
    weights = kernel_matrix**2
    return _crossterm_operator(hvp_counted, particles, weights, damping)


def compact_operator(hvp_counted, particles, damping, neighbor_frac=0.15):
    """Compact-support bump weight w(r) = (1 - r/h)_+^2, h = the radius that
    gives each particle ~neighbor_frac*n neighbors on average (a percentile
    of the current pairwise-distance distribution -- adaptive, like the
    median-heuristic bandwidth). NOT a validated positive-definite (Wendland)
    kernel -- a practical bump for this PoC's cost/behavior profiling, not a
    proposed new repulsion kernel.
    """
    n = particles.shape[0]
    diffs = particles[:, None, :] - particles[None, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=-1))
    h = max(np.percentile(dists[~np.eye(n, dtype=bool)], neighbor_frac * 100), 1e-6)
    weights = np.clip(1.0 - dists / h, 0.0, None) ** 2  # see dense_exact_operator's note on /n
    return _crossterm_operator(hvp_counted, particles, weights, damping), (weights > 0).sum(axis=1).mean()


# ---------------------------------------------------------------------------
# Minimal, uniform SVGD driver (mirrors update.py's core numerics closely
# enough for a fair comparison, with direct hooks for custom preconditioners
# update() doesn't expose)
# ---------------------------------------------------------------------------
def run_svgd(
    x0,
    *,
    n_iter,
    stepsize,
    mode,  # "baseline" | "lbfgs" | "mean_field" | "per_particle" | "dense_exact" | "compact"
    cg_iters=8,
    damping=1e-4,
    lbfgs_history=10,
):
    particles = x0.copy()
    n, d = particles.shape
    lbfgs_states = (
        [make_lbfgs_state(d, m=lbfgs_history) for _ in range(n)] if mode == "lbfgs" else None
    )
    prev_particles = prev_grads = None

    hvp_counted = HvpCounter(hvp_raw)
    variance_history = []
    mean_neighbors = []
    t0 = time.perf_counter()

    for it in range(n_iter):
        grads = grad_fn(particles)
        variance_history.append(float(np.sum(np.var(particles, axis=0))))

        if mode == "lbfgs" and prev_particles is not None:
            for i in range(n):
                lbfgs_update(lbfgs_states[i], particles[i] - prev_particles[i], grads[i] - prev_grads[i])

        precond_grads = grads
        if mode == "lbfgs":
            precond_grads = np.zeros_like(particles)
            for i in range(n):
                precond_grads[i] = -lbfgs_direction(lbfgs_states[i], grads[i])

        kernel_matrix, kernel_grad = rbf_kernel_normalized(particles, -1.0)
        attractive = kernel_matrix @ precond_grads
        phi = -(attractive - kernel_grad) / n

        if mode in ("baseline", "lbfgs"):
            attr_max = np.max(np.abs(attractive)) / n
            decay = np.sqrt(1.0 + it)
            step = stepsize / (attr_max * decay) if attr_max > 0 else stepsize / decay
            displacement = step * phi
        else:
            if mode == "mean_field":
                op = mean_field_operator(hvp_counted, particles, damping)
            elif mode == "per_particle":
                op = per_particle_operator(hvp_counted, particles, damping)
            elif mode == "dense_exact":
                op = dense_exact_operator(hvp_counted, particles, kernel_matrix, damping)
            elif mode == "compact":
                op, nbrs = compact_operator(hvp_counted, particles, damping)
                mean_neighbors.append(nbrs)
            else:
                raise ValueError(mode)
            newton_phi = _batched_cg(op, phi, cg_iters)
            displacement = stepsize * newton_phi

        if mode == "lbfgs":
            prev_particles = particles.copy()
            prev_grads = grads.copy()

        particles = particles + displacement

    wall = time.perf_counter() - t0
    return {
        "particles": particles,
        "variance_history": variance_history,
        "wall_time": wall,
        "hvp_calls": hvp_counted.calls,
        "hvp_rows": hvp_counted.rows,
        "mean_neighbors": float(np.mean(mean_neighbors)) if mean_neighbors else None,
    }


def summarize(name, result, true_total_var):
    final_var = result["variance_history"][-1]
    mean_abs = float(np.mean(np.abs(result["particles"])))
    print(
        f"{name:14s}  final_var={final_var:9.3f}  var_ratio={final_var / true_total_var:6.3f}"
        f"  mean|x|={mean_abs:7.4f}  wall={result['wall_time']:7.3f}s"
        f"  hvp_calls={result['hvp_calls']:6d}  hvp_rows={result['hvp_rows']:8d}"
        f"  avg_nbrs={result['mean_neighbors']}"
    )


if __name__ == "__main__":
    # Every knob below is load-bearing for the published table in
    # docs/high-dimensional.md -- stepsize, cg_iters and damping all move the
    # variance ratios substantially, and the variants are only comparable at
    # shared settings. Change them and the table must be regenerated from this
    # file rather than edited, or the two silently drift apart.
    n, d = 150, 300
    n_iter = 150
    rng = np.random.default_rng(0)
    x0 = rng.normal(0, 3.0, (n, d))

    true_var_per_dim = true_marginal_variance()
    true_total_var = d * true_var_per_dim
    print(f"target: separable nonlinear GN model, n={n} d={d} n_iter={n_iter}")
    print(f"true per-dim variance (quadrature) = {true_var_per_dim:.4f}, true total = {true_total_var:.2f}\n")

    print(
        f"{'variant':14s}  {'final_var':>9s}  {'var_ratio':>6s}  {'mean|x|':>7s}"
        f"  {'wall':>8s}  {'hvp_calls':>9s}  {'hvp_rows':>8s}  avg_nbrs"
    )

    r = run_svgd(x0, n_iter=n_iter, stepsize=1.0, mode="baseline")
    summarize("baseline", r, true_total_var)

    r = run_svgd(x0, n_iter=n_iter, stepsize=1.0, mode="lbfgs")
    summarize("lbfgs", r, true_total_var)

    r = run_svgd(x0, n_iter=n_iter, stepsize=1.0, mode="mean_field", cg_iters=8, damping=1e-4)
    summarize("mean_field", r, true_total_var)

    r = run_svgd(x0, n_iter=n_iter, stepsize=1.0, mode="per_particle", cg_iters=8, damping=1e-4)
    summarize("per_particle", r, true_total_var)

    r = run_svgd(x0, n_iter=n_iter, stepsize=1.0, mode="compact", cg_iters=8, damping=1e-4)
    summarize("compact", r, true_total_var)

    r = run_svgd(x0, n_iter=n_iter, stepsize=1.0, mode="dense_exact", cg_iters=8, damping=1e-4)
    summarize("dense_exact", r, true_total_var)
