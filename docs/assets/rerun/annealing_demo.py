# Regenerates annealing_demo.rrd, the recording embedded on the "Advanced
# demos" docs page. Run from the repo root with the `rerun` extra installed:
#
#   uv run --extra rerun python docs/assets/rerun/annealing_demo.py

import numpy as np
import rerun as rr
from _logging import make_logger

import simplesvgd

# Four modes; particles all start clustered near just one of them, so the
# run has to actually spread out to find the other three. step_schedule=
# "constant" matters here -- AdaGrad's per-step adaptive normalization
# renormalizes away a smoothly-varying overall gradient scale, which
# cancels out most of what the temperature schedule is supposed to do.
_MODES = np.array([[2.5, 2.5], [-2.5, 2.5], [-2.5, -2.5], [2.5, -2.5]])
_MODE_SIGMA = 1.0


def mixture_grad(x):
    diffs = x[:, None, :] - _MODES[None, :, :]
    sq_dist = np.sum(diffs**2, axis=-1) / (2 * _MODE_SIGMA**2)
    weights = np.exp(-(sq_dist - sq_dist.min(axis=1, keepdims=True)))
    responsibilities = weights / weights.sum(axis=1, keepdims=True)
    return np.sum(responsibilities[:, :, None] * diffs, axis=1) / _MODE_SIGMA**2


rng = np.random.default_rng(0)
x0 = rng.normal(_MODES[0], 1.0, (200, 2))
n_iter = 300

rr.init("annealing-demo", recording_id="annealing-demo", spawn=False)

simplesvgd.update(
    x0.copy(),
    mixture_grad,
    simplesvgd.SVGDConfig(
        n_iter=n_iter,
        stepsize=0.3,
        step_schedule="constant",
        disable_progressbar=True,
        callback=make_logger("without_annealing"),
    ),
)

simplesvgd.update(
    x0.copy(),
    mixture_grad,
    simplesvgd.SVGDConfig(
        n_iter=n_iter,
        stepsize=0.3,
        step_schedule="constant",
        disable_progressbar=True,
        temperature_schedule="linear",
        callback=make_logger("with_annealing"),
    ),
)

rr.save("docs/assets/rerun/annealing_demo.rrd")
