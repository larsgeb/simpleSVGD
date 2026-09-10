# SVGD in high dimensions

Everything on this page is about one regime: **many more parameters than
particles** (`d >> n`). That's where PDE-constrained Bayesian inverse
problems live -- full-waveform inversion (FWI), tomography, subsurface
imaging -- and it's where plain SVGD quietly gives you a *confident wrong
answer* rather than an obviously broken one.

If your problem has `d` in the tens and `n` in the hundreds, none of this
applies and the [mini-tutorial](index.md) is all you need.

## The failure mode: variance collapse

SVGD balances two forces: an attractive term pulling particles toward
high-density regions, and a repulsive kernel-gradient term pushing them
apart so they *spread over* the posterior rather than piling onto its mode.
In high dimensions the repulsive term loses.

With a standard RBF kernel and the median-heuristic bandwidth, `h² ~ O(d)`,
so the repulsive gradient shrinks like `O(1/d)` while attraction does not.
Past some dimension the ensemble stops representing posterior *spread* and
becomes an expensive way to find the MAP point several times over. The
posterior mean can look fine while every uncertainty estimate you draw from
the ensemble is far too small.

This is documented behavior, not an implementation defect -- see Ba et al.,
*Understanding the Variance Collapse of SVGD in High Dimensions* (ICLR
2022). It shows up across particle-based variational methods (SVGD, Stein
variants, ensemble Kalman) once dimension exceeds ensemble size by a large
factor.

### Detecting it

Two diagnostics ship in [`SVGDState`][simplesvgd.SVGDState] for exactly this:

- `particle_variance_history` -- trace of the empirical covariance. Watch for
  it shrinking steadily to a value far below what the posterior's actual
  variance should be.
- `repulsion_ratio_history` -- ratio of the repulsive term's norm to the
  attractive term's norm. **Read this early in the run**, in the first
  handful of iterations while particles are still diffuse. A value far below
  1 there means repulsion was already overwhelmed before it had any chance
  to spread the ensemble. Late in a run this ratio is uninformative -- the
  attractive term decays near any converged mode whether or not the ensemble
  collapsed, which swamps the trend.

See the [variance-collapse demo](advanced-demos.md#variance-collapse-diagnostics)
for both traces on a `d=2` vs `d=500` run of the same target.

## The ceiling no kernel can lift

Before reaching for a better kernel, be clear about a hard limit: **`n`
particles span an affine subspace of dimension at most `n-1`.** With 30
particles you cannot represent posterior variance in more than 29
independent directions, no matter how the kernel is built. Every remaining
direction has *exactly zero* ensemble variance, by construction.

So "fix the kernel" can only ever address the directions you have particles
to spare for. Genuinely resolving uncertainty in `d` directions requires
either more particles, or -- far more practically -- **reducing the number of
directions that need resolving**. That is what the rest of this page is
about.

## Practical order of operations

Roughly in descending order of payoff for an FWI-shaped problem:

### 1. Whiten by the prior

Work in `z = L⁻¹x` where `LLᵀ = C_prior`. The prior becomes `N(0, I)`, so
one isotropic kernel bandwidth is meaningful across all coordinates, and
step sizes stop being dominated by the prior's stiffest direction. Smooth
FWI priors routinely carry condition numbers of `10³`–`10⁶`; unwhitened,
the kernel and the step schedule are both fighting that.

This is cheap and foundational, but note what it does *not* do: in the
benchmark below, whitening alone still collapsed once run long enough. It
improves conditioning, it does not add degrees of freedom.

### 2. Reduce to the likelihood-informed subspace

This is the highest-payoff step, because it attacks the ceiling directly.

Data informs far fewer directions than you have parameters. The dominant
eigenvectors of the prior-preconditioned Gauss-Newton Hessian
`H̃ = C_prior^{1/2} Jᵀ Σ⁻¹ J C_prior^{1/2}` span the directions the
likelihood actually constrains; in the complement, the posterior *equals the
prior*, and you can sample it exactly instead of asking SVGD to discover it.

Run SVGD on the `r` informed coefficients only, and draw the complement from
the prior. Now the requirement is `n > r`, not `n > d`. This is projected
SVGD (pSVGD; Chen & Ghattas, NeurIPS 2020), demonstrated on PDE-constrained
inverse problems.

`H̃`'s dominant eigenpairs come from randomized eigendecomposition, which
needs only Hessian-vector products -- the same adjoint-state primitive FWI
already has. Budget roughly `r + oversampling` Hvps, once (or refreshed
occasionally as particles move).

!!! note "Not in simplesvgd yet"
    pSVGD is not currently a library feature. The benchmark script below
    implements it standalone against this library's kernels.

#### Choosing the rank

`r` has two distinct failure directions, and they are not symmetric:

- **`r` too small** truncates informed directions. Those modes keep their
  *prior* variance, which is far too wide, and -- more seriously -- the
  posterior *mean* is wrong in them. In the benchmark below, `r=10` (against
  15 informed modes) gave a relative mean error of `0.496`, versus `0.001`
  once the rank covered them all.
- **`r` too large** spreads a fixed particle budget over more directions than
  it can populate, and ordinary collapse takes them back: in-subspace recovery
  fell from `0.184` at `r=20` to `0.131` at `r=40`. This wastes particles but
  leaves the mean intact.

These are not equally bad, so the rule is not "balance them". **Take every
mode with `λₖ` meaningfully above 1** -- the directions where data beats the
prior -- and accept the variance cost. Under-ranking to flatter a variance
diagnostic corrupts the posterior mean, which is usually the primary
deliverable.

Then check `n/r` separately. At fixed `r=5`, in-subspace variance recovery
went `0.39 → 0.64 → 0.70 → 0.77 → 0.82` for `n = 25, 50, 100, 200, 400`: it
improves with the particle-to-rank ratio, but *slowly* -- a 16× particle
increase bought about 2×. If the spectrum demands a rank your particle budget
can't support, that is real information about the problem, not a tuning
failure.

!!! warning "Measure truncation separately from collapse"
    A single pooled "informed-mode variance ratio" mixes these two failures
    and is actively misleading -- truncated modes (ratio >> 1) and collapsed
    modes (ratio << 1) partly cancel in a median. Our first pass at this
    measurement did exactly that, and reported an "over-dispersion at low
    rank" that was pure truncation bias. Report the two groups separately.

### 3. Match the kernel metric to the curvature

Inside the informed subspace the posterior is strongly anisotropic --
variances can span two or three orders of magnitude across modes. A single
isotropic bandwidth cannot serve all of them: it is far too wide for the
tightly-constrained directions, so repulsion there is negligible and those
directions collapse even though you have enough particles for them.

The fix is a curvature-aware metric -- a matrix-valued kernel (Wang et al.,
NeurIPS 2019) with `Q = (I + H̃)`, equivalently a change of variables
`u = (I + H̃)^{1/2} z` that makes the posterior isotropic before an ordinary
RBF kernel is applied.

There is a convenient accident here: **the pSVGD projection basis is already
the eigenbasis of `H̃`**, so inside the subspace the metric is diagonal
(`1 + λₖ`). Both the metric rescaling and an exact Newton preconditioner
come for free from eigenpairs you computed anyway -- no CG, no extra Hvps.

### 4. Then, the knobs this library already gives you

- **Kernel**: [`rbf_normalized`][simplesvgd.rbf_kernel_normalized] normalizes
  per-dimension before computing distances, keeping the median-heuristic
  bandwidth dimension-independent. Use it whenever `d > ~100`.
  [`make_mass_weighted_kernel`][simplesvgd.make_mass_weighted_kernel] weights
  distances by cell volume for mesh-discretized fields, so posterior
  statistics stay stable under mesh refinement.
- **Preconditioners**: `preconditioner="lbfgs"` (per-particle curvature from
  gradient history, no Hessian needed) and `preconditioner="svn"` (mean-field
  Stein Variational Newton, needs `hessian_vector_product`). Both address the
  *drift* term; neither addresses repulsion, so neither is a collapse remedy.
  Both also over-dispersed on the position-dependent-curvature target below,
  so check `particle_variance_history` against something you trust rather than
  assuming a preconditioner can only help.
- **Annealing**: `temperature_schedule="linear"` ramps the likelihood's
  influence up over the run, letting particles spread before the full
  gradient pulls them in. Useful when the posterior is sharply peaked
  relative to the initial spread, which describes most FWI setups.

### 5. Watch the step schedule

`step_schedule="adagrad"` (the library default when no preconditioner is set)
**never decays**. Its per-coordinate normalization divides the gradient by an
estimate of its own recent magnitude, so every particle keeps taking a step
of size ~`stepsize` on every iteration forever, however close it already is.
Lowering `stepsize` shrinks that step but never makes it decay.

In practice this doesn't look like noise -- it settles into an exact
back-and-forth, with diagnostics alternating between two fixed values
indefinitely. If your `particle_variance` never settles, this is almost
certainly why. Use `step_schedule="robbins-monro"`, which adds a growing
`sqrt(1 + iteration)` divisor.

Newton-preconditioned runs are the exception: `preconditioner="svn"` defaults
to `"constant"`, because a genuine Newton direction wants a fixed
damped-Newton step fraction rather than an unrelated decay, and `phi` already
tends to zero near convergence on its own.

### 6. Confirm convergence before believing any variance diagnostic

Variance diagnostics read at a fixed iteration budget are not comparable
across configurations, because **larger ensembles converge more slowly**. In
the projected benchmark below, the in-subspace variance ratio at `r=20` read:

| iterations | `n=50` | `n=100` | `n=200` |
|---|---|---|---|
| 2 000 | 0.184 | 0.481 | 2.418 |
| 8 000 | 0.179 | 0.188 | 0.251 |
| 20 000 | 0.184 | 0.186 | 0.243 |

At 2 000 iterations the `n=200` run looks *over-dispersed* and the ensemble
size looks like it fixed collapse. It did not: the run was simply still on its
way down, and by 8 000 iterations all three sit in the same band. Anyone who
stopped at the first row would have concluded that more particles solves
collapse, and shipped that.

Before comparing configurations, re-run at least one of them for 4× the
iterations and check the diagnostic has actually stopped moving. This costs
one extra run and is the single cheapest way to avoid publishing a
convergence artifact as a result.

## Benchmarks

!!! warning "Prototype results"
    The numbers below come from prototype benchmark scripts, single seed, on
    synthetic targets. They are directional evidence for choosing an approach,
    not validated library behavior. The library's own test suite is the
    authority on shipped features.

    Both tables are regenerated by running the linked script as-is. They are
    sensitive to the settings in each script's `__main__` block -- an earlier
    draft of this page carried numbers from a script revision that no longer
    existed, and they were wrong in ways that changed the conclusions. If you
    change a script, re-run it and replace the table; don't edit the table.

### Which preconditioner?

Separable nonlinear Gauss-Newton target (`f(xₖ) = xₖ + 0.5·sin(xₖ)`, so
curvature genuinely varies with position), `d=300`, `n=150`, 150 iterations.
True variance from 1D quadrature. Script:
[`svn_variants.py`](https://github.com/larsgeb/simpleSVGD/blob/master/docs/assets/benchmarks/svn_variants.py).

`variance / true` is the ensemble's total variance over the quadrature truth:
`1.0` is exact, `<< 1` collapse, `>> 1` over-dispersion.

| variant | variance / true | Hvp rows |
|---|---|---|
| no preconditioner | 0.011 | 0 |
| L-BFGS | 4.139 | 0 |
| mean-field SVN (shipped) | 5.906 | 180,000 |
| per-particle SVN | **1.013** | 180,000 |
| compact-kernel cross-terms | 1.077 | 4,202,400 |
| dense exact cross-terms | 1.060 | 27,000,000 |

Three findings worth carrying forward:

- **Per-particle curvature is free, and here it was the difference between a
  usable answer and a bad one.** Evaluating each particle's Hessian at its own
  position rather than at the ensemble mean recovered the true variance almost
  exactly (`1.013`), while the mean-field approximation over-dispersed by ~6×
  -- at *byte-identical* Hvp cost (both 1 200 batched calls, 180 000 rows;
  the batched per-row convention already evaluates one Hessian per particle,
  so tiling the mean across rows throws that away for nothing).
- **This is a mark against the currently shipped mode.** `preconditioner="svn"`
  is mean-field today. On this target it is the worst Newton variant tested.
  Treat the shipped mode as the conservative default it is, and see the
  roadmap below.
- **Cross-terms bought nothing** on this target -- they were slightly *worse*
  than per-particle (`1.077` and `1.060` against `1.013`) for 23× and 150× the
  Hvp cost. Note that the compact-support approximation tracked the dense
  exact computation closely at 6× less work, so if cross-terms are ever worth
  having, the compact form is the way to get them. Caveat: this target is
  separable and the particle cloud homogeneous, so neighbouring particles
  share similar curvature -- precisely the condition under which
  kernel-weighted Hessian averaging degenerates to what per-particle already
  gives. A coupled, non-separable target could differ.

Note that L-BFGS also over-disperses badly here (`4.139`). Both it and
mean-field SVN build a curvature estimate that is *shared or stale* relative
to where the particle actually is; on a target whose curvature genuinely
varies with position, that mismatch shows up as too much spread, not too
little.

### Which dimension-reduction strategy?

Linear-Gaussian FWI surrogate: smooth GRF-like prior on a 500-point grid,
40 smooth observations with a rapidly decaying spectrum (15 modes with
`λ > 1`), Gaussian noise. The posterior is analytic, so per-direction
variance has exact ground truth. `d=500`, `n=50`. Script:
[`dimension_reduction.py`](https://github.com/larsgeb/simpleSVGD/blob/master/docs/assets/benchmarks/dimension_reduction.py).

Variance columns are empirical / true variance: `1.0` is exact, `<< 1` is
collapse, `>> 1` is over-dispersion. *In-subspace* covers informed modes the
method actually represents; *truncated* covers informed modes a projected
method dropped (left at prior variance, hence far too wide). `mean err` is
relative error of the posterior mean over the informed modes.

| approach | in-subspace | truncated | uninformed | mean err |
|---|---|---|---|---|
| plain SVGD (rbf) | 38.9 | -- | 0.001 | 0.48 |
| plain SVGD (rbf_normalized) | 32.5 | -- | 0.001 | 0.43 |
| prior-whitened | 0.000 | -- | 0.008 | 0.003 |
| whitened + exact Newton drift | 0.000 | -- | 0.008 | 0.000 |
| whitened + Hessian metric | 0.070 | -- | 0.006 | 0.000 |
| pSVGD (r=10) | 0.419 | 4.14 | **0.963** | 0.496 |
| pSVGD (r=20) | 0.000 | -- | **0.963** | 0.001 |
| pSVGD (r=40) | 0.000 | -- | **0.948** | 0.006 |
| pSVGD + Newton drift (r=10) | 0.416 | 4.14 | **0.963** | 0.496 |
| pSVGD + Newton drift (r=20) | 0.000 | -- | **0.963** | 0.001 |
| pSVGD + Hessian metric (r=10) | 0.321 | 4.14 | **0.963** | 0.496 |
| pSVGD + Hessian metric (r=20) | **0.184** | -- | **0.963** | 0.000 |
| pSVGD + Hessian metric (r=40) | 0.131 | -- | **0.948** | 0.000 |

Four things come out of this.

**The uninformed column is the FWI headline.** It covers ~480 of the 500
directions, and it is where honest uncertainty quantification either survives
or doesn't. Every full-space method loses it by two to three orders of
magnitude. pSVGD retains it essentially exactly -- not because its dynamics
are better there, but because it never asks SVGD to discover something the
prior already tells you.

**Plain SVGD's informed modes are over-dispersed, not collapsed** (38.9,
32.5), while its uninformed modes are flattened (0.001). Aggregate particle
variance would partly net these against each other. This is the concrete
reason to measure variance recovery *per direction* against known structure,
rather than trusting a single scalar spread diagnostic.

**Under-ranking is not a cheap trade.** `r=10` looks appealing on in-subspace
variance (0.419 vs 0.000 at `r=20`) but it is the worst row in the table on
the posterior *mean*: `0.496` relative error against `0.001`. Dropping
informed directions biases the quantity FWI usually cares about most. Take the
rank the spectrum tells you to take; do not tune it down to make a variance
diagnostic look better.

**Newton preconditioning of the drift changed essentially nothing** anywhere
it appears (0.419 → 0.416, 0.000 → 0.000). Consistent rather than surprising:
SVN-family methods precondition the drift, and collapse is a repulsion-side
failure. Use SVN for convergence speed on stiff problems, not as a collapse
remedy.

That leaves the honest gap: with all informed modes represented (`r=20`), the
best in-subspace recovery here is `0.184`, and it gets *worse* at `r=40`
(`0.131`) as the same 50 particles spread thinner. The Hessian metric is what
makes those directions non-zero at all -- without it they read `0.000` -- but
it does not make them right. Uncertainty inside the informed subspace remains
underestimated, and no variant tested here fixes it.

## Roadmap

Not currently in the library, in rough order of expected value for FWI:

1. Projected SVGD (pSVGD) with randomized subspace construction from Hvps.
   Clearly the largest effect measured here, and the one that changes the
   requirement from `n > d` to `n > r`.
2. Matrix-valued / Hessian-metric kernels. Free alongside (1), since the
   projection basis is already `H̃`'s eigenbasis.
3. Per-particle SVN (own-position Hessian) as a `preconditioner="svn"` mode
   alongside the current mean-field one. Identical Hvp cost to mean-field and
   far better behaved in the benchmark above (`1.013` vs `5.906`) -- by some
   distance the cheapest item on this list, and arguably the right default
   once it has test coverage of its own.
4. Kernel-weighted cross-term SVN -- only worth revisiting if a coupled
   target shows the cross-terms earning their cost.

Two open questions that the benchmarks here do *not* settle:

- In-subspace recovery plateaus well below 1.0 even at generous `n/r`
  (`0.82` at `n=400`, `r=5`). Whether that residual is the known finite-`n`
  SVGD variance bias, a bandwidth-selection artifact, or something specific
  to this setup is not established.
- Everything here is a linear-Gaussian surrogate with an analytic posterior,
  chosen so ground truth exists. FWI's actual nonlinearity -- multiple minima,
  cycle-skipping -- is exactly what the surrogate leaves out, and it is the
  part most likely to change the ranking.

## References

- Liu & Wang, *Stein Variational Gradient Descent* (NeurIPS 2016)
- Ba et al., *Understanding the Variance Collapse of SVGD in High Dimensions*
  (ICLR 2022)
- Chen & Ghattas, *Projected Stein Variational Gradient Descent* (NeurIPS 2020)
- Wang et al., *Stein Variational Gradient Descent with Matrix-Valued Kernels*
  (NeurIPS 2019)
- Detommaso et al., *A Stein Variational Newton Method* (NeurIPS 2018)
