# PoC round 2: dimension-reduction approaches for FWI-scale SVGD.
#
# Benchmark is built to match Bayesian FWI's actual structure rather than a
# generic high-d Gaussian:
#   - smooth (stiff) Gaussian prior on a 1D grid, like a GRF/smoothness prior
#   - observation operator with a rapidly decaying spectrum (smooth sensitivity
#     kernels), so only r << d directions are informed by the data
#   - Gaussian noise
# That makes the posterior analytic: in prior-whitened coordinates rotated to
# the eigenbasis of the prior-preconditioned Gauss-Newton Hessian H~, the
# posterior is diagonal with variance 1/(1+lambda_k). Informed modes have
# lambda_k >> 1 (variance << 1); uninformed modes have lambda_k ~ 0 (variance
# ~ 1, i.e. the prior). So we get EXACT per-direction ground truth, and the
# question "does this method recover prior variance where the data says
# nothing?" becomes directly measurable -- which is the real FWI question,
# not aggregate particle variance.
#
# Run from repo root: uv run python <this file>

import time

import numpy as np

from simplesvgd.kernels import rbf_kernel, rbf_kernel_normalized

# ---------------------------------------------------------------------------
# Problem setup
# ---------------------------------------------------------------------------


def build_problem(d=500, n_obs=40, corr_len=0.05, obs_width=0.03, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    grid = np.linspace(0.0, 1.0, d)

    # Smooth squared-exponential prior (GRF-like), floored to keep the
    # condition number survivable for the unwhitened baselines.
    sq = (grid[:, None] - grid[None, :]) ** 2
    c_prior = np.exp(-0.5 * sq / corr_len**2)
    evals, evecs = np.linalg.eigh(c_prior)
    evals = np.maximum(evals, evals.max() * 1e-6)
    c_prior = (evecs * evals) @ evecs.T
    l_factor = evecs * np.sqrt(evals)  # C = L L^T
    l_inv = (evecs / np.sqrt(evals)).T  # L^{-1}
    c_prior_inv = (evecs / evals) @ evecs.T

    # Observation operator: smooth bumps -> rapidly decaying spectrum, the way
    # FWI sensitivity kernels are smooth and highly correlated.
    centers = np.linspace(0.1, 0.9, n_obs)
    g_op = np.exp(-0.5 * (grid[None, :] - centers[:, None]) ** 2 / obs_width**2)
    g_op /= np.linalg.norm(g_op, axis=1, keepdims=True)

    x_true = l_factor @ rng.normal(size=d)
    y_data = g_op @ x_true + noise * rng.normal(size=n_obs)

    # Prior-preconditioned Gauss-Newton Hessian in whitened coordinates.
    gl = g_op @ l_factor
    h_tilde = gl.T @ gl / noise**2
    lam, psi = np.linalg.eigh(h_tilde)
    order = np.argsort(lam)[::-1]
    lam, psi = lam[order], psi[:, order]

    # Analytic posterior (whitened): mean solves (I + H~) m = L^T G^T y/noise^2
    rhs = gl.T @ y_data / noise**2
    post_mean_z = psi @ (psi.T @ rhs / (1.0 + lam))
    post_var_modes = 1.0 / (1.0 + lam)  # true variance of w_k = psi_k^T z

    return {
        "d": d,
        "grid": grid,
        "c_prior": c_prior,
        "c_prior_inv": c_prior_inv,
        "l_factor": l_factor,
        "l_inv": l_inv,
        "g_op": g_op,
        "y_data": y_data,
        "noise": noise,
        "gl": gl,
        "lam": lam,
        "psi": psi,
        "post_mean_z": post_mean_z,
        "post_var_modes": post_var_modes,
        "x_true": x_true,
    }


def make_grad_x(p):
    """grad of -log posterior in ORIGINAL coordinates.

    The prior and likelihood operators are summed ONCE here rather than applied
    as two separate matmuls per call: this gradient dominates the benchmark's
    runtime (two 500x500 products per iteration, tens of thousands of
    iterations), and fusing them halves it for free.
    """
    posterior_op = p["c_prior_inv"] + p["g_op"].T @ p["g_op"] / p["noise"] ** 2
    gt_y = p["g_op"].T @ p["y_data"] / p["noise"] ** 2
    op_t = np.ascontiguousarray(posterior_op.T)

    def grad(x):
        return x @ op_t - gt_y

    return grad


def make_grad_z(p):
    """grad of -log posterior in PRIOR-WHITENED coordinates z (prior = N(0,I))."""
    gl = p["gl"]
    posterior_op = np.eye(p["d"]) + gl.T @ gl / p["noise"] ** 2
    gty = gl.T @ p["y_data"] / p["noise"] ** 2
    op_t = np.ascontiguousarray(posterior_op.T)

    def grad(z):
        return z @ op_t - gty

    return grad


# ---------------------------------------------------------------------------
# Plain SVGD loop (robbins-monro, matching the library's schedule)
# ---------------------------------------------------------------------------
def svgd_loop(x0, grad_fn, kernel_fn, n_iter, stepsize):
    particles = x0.copy()
    n = particles.shape[0]
    for it in range(n_iter):
        grads = grad_fn(particles)
        kmat, kgrad = kernel_fn(particles, -1.0)
        attractive = kmat @ grads
        phi = -(attractive - kgrad) / n
        attr_max = np.max(np.abs(attractive)) / n
        decay = np.sqrt(1.0 + it)
        step = stepsize / (attr_max * decay) if attr_max > 0 else stepsize / decay
        particles = particles + step * phi
    return particles


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------
def run_plain(p, n_particles, n_iter, stepsize, kernel_fn, seed=1):
    """SVGD in original coordinates."""
    rng = np.random.default_rng(seed)
    x0 = (p["l_factor"] @ rng.normal(size=(p["d"], n_particles))).T
    grad = make_grad_x(p)
    return svgd_loop(x0, grad, kernel_fn, n_iter, stepsize)


def run_whitened(p, n_particles, n_iter, stepsize, kernel_fn, seed=1):
    """SVGD in prior-whitened coordinates, mapped back at the end."""
    rng = np.random.default_rng(seed)
    z0 = rng.normal(size=(n_particles, p["d"]))
    grad = make_grad_z(p)
    z = svgd_loop(z0, grad, kernel_fn, n_iter, stepsize)
    return z @ p["l_factor"].T


def svgd_loop_newton(x0, grad_fn, kernel_fn, n_iter, stepsize, precond):
    """SVGD with the full kernel-combined direction Newton-preconditioned, and
    a constant step (a Newton direction wants a damped-Newton step fraction,
    not Robbins-Monro decay -- see update.py's _resolve_step_schedule).
    """
    particles = x0.copy()
    n = particles.shape[0]
    for _ in range(n_iter):
        grads = grad_fn(particles)
        kmat, kgrad = kernel_fn(particles, -1.0)
        attractive = kmat @ grads
        phi = -(attractive - kgrad) / n
        particles = particles + stepsize * precond(phi)
    return particles


def run_whitened_newton(p, n_particles, n_iter, stepsize, kernel_fn, seed=1):
    """Whitened SVGD, Newton-preconditioned by the exact whitened Hessian
    (I + H~). Upper bound on what any SVN variant can do in the full space.
    """
    rng = np.random.default_rng(seed)
    z0 = rng.normal(size=(n_particles, p["d"]))
    grad = make_grad_z(p)
    psi, lam = p["psi"], p["lam"]

    def precond(phi):
        return ((phi @ psi) / (1.0 + lam)) @ psi.T

    z = svgd_loop_newton(z0, grad, kernel_fn, n_iter, stepsize, precond)
    return z @ p["l_factor"].T


def run_psvgd_newton(p, n_particles, n_iter, stepsize, kernel_fn, rank, seed=1):
    """pSVGD + exact Newton inside the subspace. The projection basis IS the
    eigenbasis of the prior-preconditioned Gauss-Newton Hessian, so the
    subspace Hessian is diagonal (1 + lambda_k) -- exact Newton for free, no
    CG, no Hessian-vector products beyond the ones the subspace already cost.
    """
    rng = np.random.default_rng(seed)
    z0 = rng.normal(size=(n_particles, p["d"]))
    psi_r = p["psi"][:, :rank]
    lam_r = p["lam"][:rank]

    w0 = z0 @ psi_r
    z_perp = z0 - w0 @ psi_r.T
    grad_z = make_grad_z(p)

    def grad_w(w):
        return grad_z(w @ psi_r.T + z_perp) @ psi_r

    def precond(phi):
        return phi / (1.0 + lam_r)

    w = svgd_loop_newton(w0, grad_w, kernel_fn, n_iter, stepsize, precond)
    return (w @ psi_r.T + z_perp) @ p["l_factor"].T


def run_whitened_metric(p, n_particles, n_iter, stepsize, kernel_fn, seed=1):
    """Full-space SVGD in HESSIAN-METRIC coordinates u = (I+H~)^{1/2} z, where
    the posterior is isotropic -- equivalent to a matrix-valued kernel with
    Q = (I+H~). Fixes anisotropy, but cannot beat the n-particles-span-<=n-1
    -dimensions ceiling, so it should still collapse at d >> n.
    """
    rng = np.random.default_rng(seed)
    psi, lam = p["psi"], p["lam"]
    scale = np.sqrt(1.0 + lam)
    grad_z = make_grad_z(p)

    z0 = rng.normal(size=(n_particles, p["d"]))
    u0 = (z0 @ psi) * scale

    def grad_u(u):
        z = (u / scale) @ psi.T
        return (grad_z(z) @ psi) / scale

    u = svgd_loop(u0, grad_u, kernel_fn, n_iter, stepsize)
    z = (u / scale) @ psi.T
    return z @ p["l_factor"].T


def run_psvgd_metric(p, n_particles, n_iter, stepsize, kernel_fn, rank, seed=1):
    """pSVGD in Hessian-metric subspace coordinates: dimension reduction AND
    an isotropic posterior inside the subspace. Expected best-of-both.
    """
    rng = np.random.default_rng(seed)
    psi_r = p["psi"][:, :rank]
    lam_r = p["lam"][:rank]
    scale = np.sqrt(1.0 + lam_r)

    z0 = rng.normal(size=(n_particles, p["d"]))
    w0 = z0 @ psi_r
    z_perp = z0 - w0 @ psi_r.T
    grad_z = make_grad_z(p)

    def grad_u(u):
        w = u / scale
        return (grad_z(w @ psi_r.T + z_perp) @ psi_r) / scale

    u = svgd_loop(w0 * scale, grad_u, kernel_fn, n_iter, stepsize)
    z = (u / scale) @ psi_r.T + z_perp
    return z @ p["l_factor"].T


def run_psvgd(p, n_particles, n_iter, stepsize, kernel_fn, rank, seed=1):
    """Projected SVGD: run SVGD only on the r informed coefficients; take the
    complement straight from the prior (which in whitened coords is N(0,I)).
    """
    rng = np.random.default_rng(seed)
    z0 = rng.normal(size=(n_particles, p["d"]))
    psi_r = p["psi"][:, :rank]

    w0 = z0 @ psi_r  # (n, rank)
    z_perp = z0 - w0 @ psi_r.T  # prior samples in the uninformed complement

    grad_z = make_grad_z(p)

    def grad_w(w):
        z = w @ psi_r.T + z_perp
        return grad_z(z) @ psi_r

    w = svgd_loop(w0, grad_w, kernel_fn, n_iter, stepsize)
    z = w @ psi_r.T + z_perp
    return z @ p["l_factor"].T


# ---------------------------------------------------------------------------
# Metrics: per-mode variance recovery against the analytic posterior
# ---------------------------------------------------------------------------
def evaluate(p, particles, subspace_rank=None):
    """Per-mode variance recovery, with TRUNCATION separated from COLLAPSE.

    A projected method with rank r < (number of informed modes) leaves the
    informed-but-truncated modes sitting at their prior variance, which is far
    too LARGE. Pooling those with the modes the method actually samples gives a
    median ratio >> 1 that reads like over-dispersion but is really an
    under-sized subspace. `ratio_informed` therefore covers only informed modes
    the method represents; `ratio_truncated` reports the ones it dropped.
    """
    z = particles @ p["l_inv"].T
    w = z @ p["psi"]  # coefficients in the H~ eigenbasis
    emp_var = np.var(w, axis=0)
    true_var = p["post_var_modes"]
    ratio = emp_var / true_var
    lam = p["lam"]
    informed_all = lam > 1.0  # data actually shrinks these modes
    inside = np.ones_like(informed_all) if subspace_rank is None else np.arange(lam.size) < subspace_rank
    informed = informed_all & inside
    truncated = informed_all & ~inside
    weak = (lam > 1e-3) & (lam <= 1.0)
    null = lam <= 1e-3  # posterior == prior here

    # Restricted to the informed modes ON PURPOSE: over all d modes, the
    # sample mean of n=50 prior draws in ~480 uninformed dimensions has norm
    # ~sqrt(d_perp/n) >> 0, so a full-space mean error mostly measures Monte
    # Carlo noise in directions the data says nothing about -- and REWARDS a
    # method that collapsed those directions to a point. The posterior mean
    # is only determined where the likelihood informs it.
    mean_w = np.mean(w, axis=0)
    true_w = p["psi"].T @ p["post_mean_z"]
    mean_err = np.linalg.norm(mean_w[informed_all] - true_w[informed_all]) / np.linalg.norm(
        true_w[informed_all]
    )
    return {
        "n_informed": int(informed.sum()),
        "ratio_informed": float(np.median(ratio[informed])) if informed.any() else float("nan"),
        "ratio_truncated": float(np.median(ratio[truncated])) if truncated.any() else float("nan"),
        "ratio_weak": float(np.median(ratio[weak])) if weak.any() else float("nan"),
        "ratio_null": float(np.median(ratio[null])) if null.any() else float("nan"),
        "mean_err": float(mean_err),
    }


def report(name, particles, p, wall, subspace_rank=None):
    m = evaluate(p, particles, subspace_rank=subspace_rank)
    print(
        f"{name:22s} var_ratio: informed={m['ratio_informed']:7.3f} "
        f"trunc={m['ratio_truncated']:8.3f} null={m['ratio_null']:7.4f}  "
        f"mean_err={m['mean_err']:6.3f}  wall={wall:6.2f}s"
    )
    return m


if __name__ == "__main__":
    p = build_problem(d=500, n_obs=40, noise=1.0, seed=0)
    # 2000 iterations, not a few hundred: a convergence check over
    # {2000, 8000, 20000} showed the informed-mode ratio still moving a lot at
    # low iteration counts, and MORE so for larger ensembles (n=200 read 2.418
    # at 2000 iters but 0.243 at 8000 and beyond). Short runs make slow
    # convergence look like over-dispersion, and make bigger ensembles look
    # better than they are. At n=50 the same check was flat from 2000 on
    # (0.184 / 0.179 / 0.184), so 2000 is converged HERE and would not be if
    # n_particles were raised -- re-check before changing it.
    n_particles, n_iter = 50, 2000
    print(f"d={p['d']}  n_particles={n_particles}  n_iter={n_iter}")
    print(f"H~ spectrum: lambda_max={p['lam'][0]:.1f}, #modes with lambda>1: {(p['lam'] > 1).sum()}")
    print(
        "true posterior mode variances: informed(median)="
        f"{np.median(p['post_var_modes'][p['lam'] > 1]):.4f}, null=1.0 (prior)\n"
    )
    print("var_ratio = empirical / true. 1.0 is perfect; <<1 is variance collapse.\n")

    t = time.perf_counter()
    x = run_plain(p, n_particles, n_iter, 1.0, rbf_kernel)
    report("plain (rbf)", x, p, time.perf_counter() - t)

    t = time.perf_counter()
    x = run_plain(p, n_particles, n_iter, 1.0, rbf_kernel_normalized)
    report("plain (rbf_norm)", x, p, time.perf_counter() - t)

    t = time.perf_counter()
    x = run_whitened(p, n_particles, n_iter, 1.0, rbf_kernel)
    report("whitened (rbf)", x, p, time.perf_counter() - t)

    t = time.perf_counter()
    x = run_whitened(p, n_particles, n_iter, 1.0, rbf_kernel_normalized)
    report("whitened (rbf_norm)", x, p, time.perf_counter() - t)

    t = time.perf_counter()
    x = run_whitened_newton(p, n_particles, n_iter, 1.0, rbf_kernel)
    report("whitened+Newton", x, p, time.perf_counter() - t)

    t = time.perf_counter()
    x = run_whitened_metric(p, n_particles, n_iter, 1.0, rbf_kernel)
    report("whitened+Hmetric", x, p, time.perf_counter() - t)

    for rank in (10, 20, 40):
        t = time.perf_counter()
        x = run_psvgd(p, n_particles, n_iter, 1.0, rbf_kernel, rank=rank)
        report(f"pSVGD (r={rank})", x, p, time.perf_counter() - t, subspace_rank=rank)

    for rank in (10, 20, 40):
        t = time.perf_counter()
        x = run_psvgd_newton(p, n_particles, n_iter, 1.0, rbf_kernel, rank=rank)
        report(f"pSVGD+Newton (r={rank})", x, p, time.perf_counter() - t, subspace_rank=rank)

    for rank in (10, 20, 40):
        t = time.perf_counter()
        x = run_psvgd_metric(p, n_particles, n_iter, 1.0, rbf_kernel, rank=rank)
        report(f"pSVGD+Hmetric (r={rank})", x, p, time.perf_counter() - t, subspace_rank=rank)
