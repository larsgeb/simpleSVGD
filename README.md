# simplesvgd

This package is a small implementation of the SVGD algorithm 

By default, this package uses **radial basis functions** to compute sample
interaction and **AdaGrad** to optimize the samples.

[![CI](https://github.com/larsgeb/simpleSVGD/actions/workflows/ci.yml/badge.svg)](https://github.com/larsgeb/simpleSVGD/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/simplesvgd.svg)](https://pypi.org/project/simplesvgd/)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://github.com/larsgeb/simpleSVGD/blob/master/pyproject.toml)
[![Docs](https://img.shields.io/badge/docs-mkdocs--material-blue.svg)](https://larsgeb.github.io/simpleSVGD/)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/larsgeb/simpleSVGD/HEAD?labpath=%2Fnotebooks%2FTutorial%20on%20using%20simplesvgd.ipynb)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5938430.svg)](https://doi.org/10.5281/zenodo.5938430)

Full documentation (mini-tutorial + API reference): https://larsgeb.github.io/simpleSVGD/



## Installation:

Requires Python 3.11+.

To get the latest release, simply use pip inside your favourite environment:
```sh
pip install simplesvgd
```

To install the latest version directly from GitHub:

```sh
git clone git@github.com:larsgeb/simpleSVGD.git
cd simpleSVGD
pip install -e .
```
# Mini-tutorial

This package can be used with minimal development. The only thing one needs to 
supply to the algorithm is:

1. The gradient of the function to optimize, `gradient_fn(samples)`. The function itself is not needed.
2. An initial collection of samples `initial_samples`, a `numpy.array`. It helps if these are close to the target
function/distribution. 

## Input/output of your `gradient_fn`

It is essential to get the input/output shapes of the target (gradient) right. As input, it should take an arbitrary amount of samples, with the appropriate dimensionality. This means if ones wants 430 samples on a 3 dimensional function, the input/output shapes looks like this:
```python
output_gradient = gradient_fn(input_samples)

input_samples.shape = (430, 3)
output_gradient.shape = (430, 3)
```

Typically, it is useful to instantiate the samples using a Normal distribution. Using NumPy, this is done with:
```python
import numpy as np
rng = np.random.default_rng(235)

mean = 0
standard_dev = 1
n_samples = 100
dimensions = 2

initial_samples = rng.normal(mean, standard_dev, [n_samples, dimensions])
```

## Defining an example target

A good 2-dimensional test function would be the Himmelblau function:
```python
def Himmelblau(input_array: np.array) -> np.array:

    # As this is a 2-dimensional function, assert that the passed input_array
    # is correct.
    assert input_array.shape[1] == 2

    # To simplify reading this function, we do this step in between. It is not
    # the most optimal way to program this.
    x = input_array[:, 0, None]
    y = input_array[:, 1, None]

    output_array = (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2

    # As the output should be a scalar function, assert that the
    # output is also length 1 in dim 2 of the array.
    assert output_array.shape == (input_array.shape[0], 1)

    smoothing = 100
    return output_array / smoothing
```
and its gradient:
```python
def Himmelblau_grad(input_array: np.array) -> np.array:

    # As this is a 2-dimensional function, assert that the passed input_array
    # is correct.
    assert input_array.shape[1] == 2

    # To simplify reading this function, we do this step in between. It is not
    # the most optimal way to program this.
    x = input_array[:, 0, None]
    y = input_array[:, 1, None]

    # Compute partial derivatives and combine them
    output_array_dx = 2 * (x ** 2 + y - 11) * (2 * x) + 2 * (x + y ** 2 - 7)
    output_array_dy = 2 * (x ** 2 + y - 11) + 2 * (x + y ** 2 - 7) * (2 * y)
    output_array = np.hstack((output_array_dx, output_array_dy))

    # Check if the output shape is correct
    assert output_array.shape == input_array.shape

    smoothing = 100
    return output_array / smoothing
```

## Running the algorithm

To run the algorithm with a 1000 samples that are initially Normally 
distribution (mean=0, standard deviation=3, parameters chosen based on prior
belief), we simply call `simplesvgd.update()` in the following way:

```python
rng = np.random.default_rng()
initial_samples = rng.normal(0, 3, [1000, 2])

#%matplotlib notebook

figure = plt.figure(figsize=(6, 6))
plt.xlabel("Parameter 0")
plt.ylabel("Parameter 1")
plt.title("SVGD animation on the Himmelblau function")

state = simplesvgd.update(
    initial_samples,
    Himmelblau_grad,
    simplesvgd.SVGDConfig(
        n_iter=130,
        stepsize=1e-1,
        #animation=simplesvgd.AnimationConfig(enabled=True, background=background, figure=figure),
    ),
)
final_samples = state.particles
```

`simplesvgd.update()` takes every tuning knob through a single
`simplesvgd.SVGDConfig` object rather than a long kwarg list -- construct one
with just the fields you need, the rest keep their defaults. Related tunables
are grouped into sub-objects -- `LBFGSConfig`, `SigmaConfig`,
`AnimationConfig` -- constructed the same way, e.g.
`SVGDConfig(sigma=SigmaConfig(value=0.1, estimate=True))`. It returns an
`SVGDState`, not a raw array -- `.particles` holds the current particle
positions and can also be passed back in via `SVGDConfig(resume_from=...)`
to continue a run. AdaGrad's internal parameters (momentum, fudge factor)
aren't user-configurable; use `step_schedule="constant"` or
`step_schedule="robbins-monro"` (see `help(simplesvgd.SVGDConfig)`) if you
need different step-size behavior.

To animate the algorithm, simply uncomment the comments. The result should be
similar to this:


https://user-images.githubusercontent.com/21038893/151603377-a473e7b1-f7b4-417b-a685-9c0cfa98dc15.mov


## Live visualization with Rerun

See it in action first: [a live, embedded demo](https://larsgeb.github.io/simpleSVGD/live-demo/)
runs a real Himmelblau-function example and lets you scrub through it right
in the browser, no install required. There are also [three more demos
comparing advanced features](https://larsgeb.github.io/simpleSVGD/advanced-demos/)
(annealing, L-BFGS preconditioning, and variance-collapse diagnostics) side
by side against their simpler baselines.

For 2-D problems, `RerunConfig` gives a much more useful live view than the
legacy matplotlib animation above: it logs particle positions and the run's
scalar diagnostics (misfit, sigma, `particle_variance_history`,
`repulsion_ratio_history`) to a [Rerun](https://rerun.io) recording every
iteration, opening a viewer with a scrubbable timeline -- pause, rewind, and
step through a run to see exactly how a knob (stepsize, kernel, annealing
schedule, ...) shaped its behavior, with the particle cloud and the
diagnostics plots moving in sync. Requires the `rerun` extra
(`pip install simplesvgd[rerun]`, a ~150MB dependency, hence optional):

```python
state = simplesvgd.update(
    initial_samples,
    Himmelblau_grad,
    simplesvgd.SVGDConfig(
        n_iter=200,
        stepsize=0.1,
        rerun=simplesvgd.RerunConfig(enabled=True),
    ),
)
```

This spawns a Rerun Viewer window by default (`RerunConfig(spawn=False)` to
just buffer the recording instead, e.g. for `rerun.save()`-ing it to inspect
later). Each `update()` call opens its own recording, so comparing two
configs in the same notebook session doesn't overlay them onto one
timeline; a `resume_from` continuation currently starts a fresh recording
too rather than stitching onto the original run's. `dimensions_to_plot`
picks which two particle dimensions to log (default `[0, 1]`), same
convention as `AnimationConfig`; unlike `AnimationConfig`, background
contour overlays aren't supported by this path.


## Diagnosing variance collapse

SVGD's particles can silently collapse onto a subset of the target's modes,
underestimating posterior variance -- a well-documented failure mode,
especially in high dimensions (see Ba et al., ["Understanding the Variance
Collapse of SVGD in High Dimensions"](https://openreview.net/forum?id=Qycd9j5Qp9J),
ICLR 2022). `SVGDState` tracks two cheap per-iteration diagnostics to help
catch this without needing an independent reference (e.g. a long MCMC run):

- `particle_variance_history`: total ensemble variance (trace of the
  empirical covariance) at each iteration. A value that ends up far below
  where the ensemble started -- and far below what the target's own variance
  should plausibly be -- is the direct symptom of collapse.
- `repulsion_ratio_history`: the norm ratio of SVGD's repulsive (kernel
  gradient) term to its attractive term, recorded at each iteration a
  displacement is computed. Read this **early in a run**, not as a trend
  across the whole run -- the attractive term naturally decays toward zero
  near any converged mode, collapsed or not, which swamps the ratio's trend
  late on. A ratio far below 1 in the first few iterations, while particles
  are still diffuse, means repulsion is already overwhelmed by attraction
  before it's had any chance to spread the ensemble out -- the direct
  mechanism behind collapse described in the paper above.

```python
state = simplesvgd.update(initial_samples, grad_fn, simplesvgd.SVGDConfig(n_iter=200))

var_ratio = state.particle_variance_history[-1] / state.particle_variance_history[0]
early_repulsion = np.mean(state.repulsion_ratio_history[:5])
if var_ratio < 0.1 or early_repulsion < 0.05:
    print("Warning: this run may have variance-collapsed.")
```

If you see this, consider `kernel="rbf_normalized"` (per-dimension
normalized RBF, recommended above ~100 dimensions) or a
`temperature_schedule` (anneals the likelihood in gradually, giving
repulsion time to spread particles out before the full posterior sharpens
around a mode).

When there are far more parameters than particles -- PDE-constrained inverse
problems such as full-waveform inversion, where `d` is in the thousands and
`n` in the tens -- those knobs are not enough on their own, because `n`
particles can only span `n-1` directions no matter how the kernel is built.
[SVGD in high dimensions](https://larsgeb.github.io/simpleSVGD/high-dimensional/)
covers what actually helps there (prior whitening, reducing to the
likelihood-informed subspace, curvature-aware kernel metrics) with benchmarks
against analytic ground truth.


# The origins of SVGD
SVGD is a general purpose variational inference algorithm that forms a natural
counterpart of gradient descent for optimization. SVGD iteratively transports a
set of particles to match with the target distribution, by applying a form of
functional gradient descent that minimizes the KL divergence.

For more information, please visit the original implementers project website -
[SVGD](http://www.cs.utexas.edu/~qlearning/project.html?p=vgd), or their
publication; Qiang Liu and Dilin Wang. [Stein Variational Gradient Descent (SVGD): A General Purpose Bayesian Inference Algorithm](http://arxiv.org/abs/1608.04471). NIPS, 2016.
