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

simplesvgd.update(
    x0,
    himmelblau_grad,
    simplesvgd.SVGDConfig(
        n_iter=150,
        stepsize=1.0,
        rerun=simplesvgd.RerunConfig(enabled=True, spawn=False, application_id="himmelblau-demo"),
    ),
)

rr.save("docs/assets/rerun/himmelblau_demo.rrd")
