# Regenerates variance_collapse_demo.rrd, the recording embedded on the
# "Advanced demos" docs page. Run from the repo root with the `rerun`
# extra installed:
#
#   uv run --extra rerun python docs/assets/rerun/variance_collapse_demo.py
#
# Same scenario as tests/test_diagnostics.py's
# TestVarianceCollapseDiagnosticsFlagCollapse: a standard RBF kernel on an
# isotropic Gaussian target is well-behaved at d=2 but variance-collapses
# at d=500 -- the failure mode issue #12's diagnostics exist to catch.

import numpy as np
import rerun as rr
from _logging import make_logger

import simplesvgd


# step_schedule="robbins-monro" (rather than the "adagrad" default) so the
# per-iteration step actually decays instead of staying ~stepsize forever --
# see generate_demo.py for why "adagrad" alone makes both the particle
# positions and particle_variance itself oscillate in a fixed back-and-forth
# for the whole run, at any stepsize. The d=500 collapse this demo exists to
# show is unaffected either way -- particle_variance crashes from ~490 to
# ~3-5 within the first 20 iterations under both schedules -- but only
# robbins-monro lets that collapsed (or, at d=2, healthy) state actually
# settle down instead of jittering indefinitely once reached.
def _run(d, prefix):
    true_mean = np.zeros(d)

    def grad_fn(x):
        return x - true_mean

    rng = np.random.default_rng(42)
    x0 = rng.normal(0, 1, (30, d))
    simplesvgd.update(
        x0,
        grad_fn,
        simplesvgd.SVGDConfig(
            n_iter=200,
            stepsize=1.0,
            step_schedule="robbins-monro",
            disable_progressbar=True,
            callback=make_logger(prefix),
        ),
    )


rr.init("variance-collapse-demo", recording_id="variance-collapse-demo", spawn=False)

_run(d=2, prefix="low_d")
_run(d=500, prefix="high_d")

rr.save("docs/assets/rerun/variance_collapse_demo.rrd")
