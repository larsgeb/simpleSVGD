# Regenerates himmelblau_demo.rrd, the recording embedded on the "Live
# demo" docs page. Run from the repo root with the `rerun` extra installed:
#
#   uv run --extra rerun python docs/assets/rerun/generate_demo.py

import numpy as np
import rerun as rr

import simplesvgd

_SMOOTHING = 30


def himmelblau_grad(x):
    xx = x[:, 0, None]
    yy = x[:, 1, None]
    dx = 2 * (xx**2 + yy - 11) * (2 * xx) + 2 * (xx + yy**2 - 7)
    dy = 2 * (xx**2 + yy - 11) + 2 * (xx + yy**2 - 7) * (2 * yy)
    return np.hstack((dx, dy)) / _SMOOTHING


rng = np.random.default_rng(0)
x0 = rng.normal(0, 3, (300, 2))

# step_schedule="adagrad" (the library default, used here originally) never
# decays: its per-coordinate normalization divides the gradient by an
# estimate of its own recent magnitude, so every particle keeps taking a
# step of magnitude ~stepsize on every single iteration forever, however
# close it already is to a mode. In practice this doesn't just look like
# noise -- it settles into an exact back-and-forth: particle_variance and
# repulsion_ratio alternate between two fixed values every iteration for
# the rest of the run, however small stepsize is (verified: sign of the
# iteration-to-iteration change flips on 90%+ of steps, at any stepsize).
# step_schedule="robbins-monro" divides by an additional growing sqrt(1 +
# iteration) factor, so the step genuinely shrinks over the run: verified
# that both the particle positions and the two diagnostic traces stop
# changing meaningfully within ~150 of these 300 iterations, while mode
# coverage across all four Himmelblau minima stays as good as before.
simplesvgd.update(
    x0,
    himmelblau_grad,
    simplesvgd.SVGDConfig(
        n_iter=300,
        stepsize=1.0,
        step_schedule="robbins-monro",
        rerun=simplesvgd.RerunConfig(enabled=True, spawn=False, application_id="himmelblau-demo"),
    ),
)

rr.save("docs/assets/rerun/himmelblau_demo.rrd")
