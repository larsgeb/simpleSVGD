# Regenerates lbfgs_demo.rrd, the recording embedded on the "Advanced
# demos" docs page. Run from the repo root with the `rerun` extra installed:
#
#   uv run --extra rerun python docs/assets/rerun/lbfgs_demo.py

import numpy as np
import rerun as rr
from _logging import make_logger

import simplesvgd

# A Gaussian whose anisotropy is rotated 45 degrees off the coordinate axes,
# so AdaGrad's per-coordinate step normalization can't align with it (unlike
# an axis-aligned anisotropic Gaussian, which AdaGrad handles by design).
# L-BFGS's curvature estimate captures the rotated covariance directly, so it
# can take a much larger, stable step in this direction than AdaGrad can.
_THETA = np.pi / 4
_R = np.array([[np.cos(_THETA), -np.sin(_THETA)], [np.sin(_THETA), np.cos(_THETA)]])
_SIGMA = np.array([10.0, 0.5])
_PRECISION = _R @ np.diag(1 / _SIGMA**2) @ _R.T


def gaussian_grad(x):
    return x @ _PRECISION.T


rng = np.random.default_rng(0)
start = _R @ np.array([20.0, 0.0])
x0 = rng.normal(start, 1.0, (200, 2))
n_iter = 250
# At this stepsize, AdaGrad's fixed-magnitude-per-coordinate steps overshoot
# every iteration and the particle cloud oscillates without ever settling
# (verified via the per-iteration displacement staying ~30% of the cloud's
# own spread, instead of decaying as the run converges). L-BFGS's
# curvature-aware step stays stable at the same stepsize and converges
# within the first ~20 iterations.
stepsize = 2.0

rr.init("lbfgs-demo", recording_id="lbfgs-demo", spawn=False)

simplesvgd.update(
    x0.copy(),
    gaussian_grad,
    simplesvgd.SVGDConfig(
        n_iter=n_iter,
        stepsize=stepsize,
        disable_progressbar=True,
        callback=make_logger("adagrad"),
    ),
)

simplesvgd.update(
    x0.copy(),
    gaussian_grad,
    simplesvgd.SVGDConfig(
        n_iter=n_iter,
        stepsize=stepsize,
        disable_progressbar=True,
        preconditioner="lbfgs",
        callback=make_logger("lbfgs"),
    ),
)

rr.save("docs/assets/rerun/lbfgs_demo.rrd")
