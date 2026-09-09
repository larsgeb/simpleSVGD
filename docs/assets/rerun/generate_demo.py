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

# stepsize=1.0 (the original value here) made particles overshoot every
# iteration under AdaGrad's near-sign-gradient normalization -- each particle
# takes a step of magnitude ~stepsize every iteration regardless of how close
# it is to a mode, so a too-large stepsize shows up as a persistently
# jittery/oscillating cloud instead of one that settles down. 0.1 keeps the
# steady-state per-iteration displacement small relative to the particles'
# final spread while still finding and populating all four Himmelblau modes.
simplesvgd.update(
    x0,
    himmelblau_grad,
    simplesvgd.SVGDConfig(
        n_iter=300,
        stepsize=0.1,
        rerun=simplesvgd.RerunConfig(enabled=True, spawn=False, application_id="himmelblau-demo"),
    ),
)

rr.save("docs/assets/rerun/himmelblau_demo.rrd")
