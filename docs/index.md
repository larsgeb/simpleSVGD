# simplesvgd

A small, pure-NumPy implementation of the SVGD algorithm.

By default, this package uses **radial basis functions** to compute sample
interaction and **AdaGrad** to optimize the samples. It also supports
per-particle **L-BFGS** preconditioning, hierarchical noise estimation,
bounds, callbacks, and resuming a run from a previous state.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/larsgeb/simpleSVGD/HEAD?labpath=%2Fnotebooks%2FTutorial%20on%20using%20simplesvgd.ipynb)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5938430.svg)](https://doi.org/10.5281/zenodo.5938430)

## Installation

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

## Mini-tutorial

This package can be used with minimal development. The only thing one needs
to supply to the algorithm is:

1. The gradient of the function to optimize, `gradient_fn(samples)`. The
   function itself is not needed.
2. An initial collection of samples `initial_samples`, a `numpy.ndarray`. It
   helps if these are close to the target function/distribution.

### Input/output of your `gradient_fn`

It is essential to get the input/output shapes of the target (gradient)
right. As input, it should take an arbitrary amount of samples, with the
appropriate dimensionality. This means if one wants 430 samples on a
3-dimensional function, the input/output shapes look like this:

```python
output_gradient = gradient_fn(input_samples)

input_samples.shape = (430, 3)
output_gradient.shape = (430, 3)
```

Typically, it is useful to instantiate the samples using a Normal
distribution. Using NumPy, this is done with:

```python
import numpy as np

rng = np.random.default_rng(235)

mean = 0
standard_dev = 1
n_samples = 100
dimensions = 2

initial_samples = rng.normal(mean, standard_dev, [n_samples, dimensions])
```

### Defining an example target

A good 2-dimensional test function is the Himmelblau function:

```python
def himmelblau(input_array: np.ndarray) -> np.ndarray:
    assert input_array.shape[1] == 2

    x = input_array[:, 0, None]
    y = input_array[:, 1, None]

    output_array = (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2

    assert output_array.shape == (input_array.shape[0], 1)

    smoothing = 100
    return output_array / smoothing
```

and its gradient:

```python
def himmelblau_grad(input_array: np.ndarray) -> np.ndarray:
    assert input_array.shape[1] == 2

    x = input_array[:, 0, None]
    y = input_array[:, 1, None]

    output_array_dx = 2 * (x**2 + y - 11) * (2 * x) + 2 * (x + y**2 - 7)
    output_array_dy = 2 * (x**2 + y - 11) + 2 * (x + y**2 - 7) * (2 * y)
    output_array = np.hstack((output_array_dx, output_array_dy))

    assert output_array.shape == input_array.shape

    smoothing = 100
    return output_array / smoothing
```

### Running the algorithm

To run the algorithm with 1000 samples that are initially normally
distributed (mean=0, standard deviation=3, parameters chosen based on prior
belief), call `simplesvgd.update()`:

```python
import simplesvgd

rng = np.random.default_rng()
initial_samples = rng.normal(0, 3, [1000, 2])

state = simplesvgd.update(
    initial_samples,
    himmelblau_grad,
    simplesvgd.SVGDConfig(n_iter=130, stepsize=1e-1),
)
final_samples = state.particles
```

`simplesvgd.update()` takes every tuning knob through a single
[`SVGDConfig`][simplesvgd.SVGDConfig] object rather than a long argument
list -- construct one with just the fields you need, the rest keep their
defaults. It returns an [`SVGDState`][simplesvgd.SVGDState], not a raw
array -- `.particles` holds the current particle positions, and the state
can be passed back in via `SVGDConfig(resume_from=...)` to continue a run.

AdaGrad's internal parameters (momentum, fudge factor) aren't
user-configurable; use `step_schedule="constant"` or
`step_schedule="robbins-monro"` if you need different step-size behavior.
See the [API reference](api.md) for the full set of `SVGDConfig` fields,
including L-BFGS preconditioning, hierarchical sigma estimation, bounds,
callbacks, and the legacy live-scatter animation.

For a runnable, interactive version of this tutorial, see the
[notebook on GitHub](https://github.com/larsgeb/simpleSVGD/blob/master/notebooks/Tutorial%20on%20using%20simplesvgd.ipynb)
or open it directly in [Binder](https://mybinder.org/v2/gh/larsgeb/simpleSVGD/HEAD?labpath=%2Fnotebooks%2FTutorial%20on%20using%20simplesvgd.ipynb).

## The origins of SVGD

SVGD is a general purpose variational inference algorithm that forms a
natural counterpart of gradient descent for optimization. SVGD iteratively
transports a set of particles to match a target distribution, by applying a
form of functional gradient descent that minimizes the KL divergence.

For more information, please visit the original implementers' project
website -- [SVGD](http://www.cs.utexas.edu/~qlearning/project.html?p=vgd), or
their publication: Qiang Liu and Dilin Wang.
[Stein Variational Gradient Descent (SVGD): A General Purpose Bayesian Inference Algorithm](http://arxiv.org/abs/1608.04471).
NIPS, 2016.
