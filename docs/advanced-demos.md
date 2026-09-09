# Advanced demos

Three more [`RerunConfig`][simplesvgd.RerunConfig]-style recordings, each
comparing two `simplesvgd.update()` runs side by side in one shared
timeline. Unlike the [live demo](live-demo.md), these use
`SVGDConfig(callback=...)` directly (with the two runs logged under
different entity-path prefixes) rather than `RerunConfig`, since
`RerunConfig` starts a fresh isolated recording per run and can't overlay
two of them -- see
[`_logging.py`](https://github.com/larsgeb/simpleSVGD/blob/master/docs/assets/rerun/_logging.py)
for the ~15-line helper this relies on, which you can reuse directly.

Every scenario below was checked numerically, not just by eye, for a failure
mode that's easy to miss: with `step_schedule="adagrad"` (the library
default), the per-coordinate normalization divides the gradient by an
estimate of its own recent magnitude, so every particle keeps taking a step
of magnitude ~`stepsize` on *every single iteration, forever* -- lowering
`stepsize` only shrinks that step, it never makes it decay over the course
of a run. In practice this doesn't look like noise, it settles into an exact
back-and-forth: a diagnostic like `particle_variance` ends up alternating
between two fixed values every iteration, indefinitely. `step_schedule=
"robbins-monro"` divides by an additional, growing `sqrt(1 + iteration)`
factor, so the step genuinely shrinks as the run progresses -- every demo
below other than the AdaGrad side of the L-BFGS comparison (where that
non-decaying behavior is exactly what's being contrasted against) uses it
for this reason. If you're building your own recording, check whether your
diagnostics keep changing indefinitely instead of settling down; if so this
is almost certainly why.

## Annealing / tempering

`SVGDConfig(temperature_schedule=...)` scales the log-density gradient
during the early iterations, letting particles diffuse across a
multi-modal landscape before the gradient's full strength pulls them into
the nearest mode. Both runs below start every particle clustered at just
*one* of four Gaussian mixture modes; `with_annealing` (linear temperature
ramp) spreads far more of them out to the other three than
`without_annealing` does.

Note: this uses `step_schedule="constant"`, not the library default
(`"adagrad"`) -- AdaGrad's per-step adaptive normalization renormalizes
away a smoothly-varying overall gradient scale, which cancels out most of
what `temperature_schedule` is supposed to do. If you're adding annealing
to your own run, check that your `step_schedule` doesn't do the same.

<iframe
    src="https://app.rerun.io/version/0.37.1?url=https://larsgeb.github.io/simpleSVGD/assets/rerun/annealing_demo.rrd"
    style="width: 100%; height: 600px; border: none;"
    allowfullscreen>
</iframe>

Script:
[`docs/assets/rerun/annealing_demo.py`](https://github.com/larsgeb/simpleSVGD/blob/master/docs/assets/rerun/annealing_demo.py).

## L-BFGS vs. AdaGrad preconditioning

`SVGDConfig(preconditioner="lbfgs")` replaces the default diagonal AdaGrad
step normalization with a curvature-aware L-BFGS direction. The target
below is a Gaussian whose anisotropy is rotated 45&deg; off the coordinate
axes, so AdaGrad's per-coordinate normalization can't align with it (an
*axis-aligned* anisotropic Gaussian is exactly what AdaGrad is designed to
handle well, so it wouldn't show a difference). At a stepsize large enough
to make `adagrad` oscillate indefinitely without converging, `lbfgs`
converges smoothly within about 20 iterations and stays put.

This library's L-BFGS has no line search, so it isn't unconditionally more
stable than AdaGrad -- on a target with rapidly-varying curvature (a curved
valley, say) a fixed-size L-BFGS step can overshoot far worse than AdaGrad
does. It helps specifically when the curvature is anisotropic but roughly
constant, which is the case demonstrated here.

<iframe
    src="https://app.rerun.io/version/0.37.1?url=https://larsgeb.github.io/simpleSVGD/assets/rerun/lbfgs_demo.rrd"
    style="width: 100%; height: 600px; border: none;"
    allowfullscreen>
</iframe>

Script:
[`docs/assets/rerun/lbfgs_demo.py`](https://github.com/larsgeb/simpleSVGD/blob/master/docs/assets/rerun/lbfgs_demo.py).

## Variance-collapse diagnostics

The posterior-quality diagnostics added for issue #12
(`particle_variance_history`, `repulsion_ratio_history`) exist to catch a
specific SVGD failure mode: in high dimensions, the RBF kernel's bandwidth
can collapse, the repulsive term vanishes, and the particle cloud shrinks
towards a single point instead of representing the posterior. Below,
identical isotropic-Gaussian-target runs are shown at `d=2` (well-behaved)
and `d=500` (collapses) -- same scenario as
[`tests/test_diagnostics.py`](https://github.com/larsgeb/simpleSVGD/blob/master/tests/test_diagnostics.py)'s
`TestVarianceCollapseDiagnosticsFlagCollapse`. Watch `particle_variance` in
the `high_d` diagnostics: it crashes from its initial value towards zero
within about 20 iterations and settles there, while `low_d`'s rises and
settles at a healthy, clearly nonzero plateau instead.

<iframe
    src="https://app.rerun.io/version/0.37.1?url=https://larsgeb.github.io/simpleSVGD/assets/rerun/variance_collapse_demo.rrd"
    style="width: 100%; height: 600px; border: none;"
    allowfullscreen>
</iframe>

Script:
[`docs/assets/rerun/variance_collapse_demo.py`](https://github.com/larsgeb/simpleSVGD/blob/master/docs/assets/rerun/variance_collapse_demo.py).
