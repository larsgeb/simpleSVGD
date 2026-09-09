# Shared helper for the A/B comparison demos (annealing_demo.py,
# lbfgs_demo.py, variance_collapse_demo.py): logs one update() run under
# an entity-path prefix, via SVGDConfig(callback=...) rather than
# RerunConfig -- so two runs can be interleaved into one shared Rerun
# recording (RerunConfig's own rr.init() call is scoped to a single
# update() call, by design, to keep independent runs from overlaying).

import rerun as rr


def make_logger(prefix, dims=(0, 1)):
    def log_iteration(iteration, state):
        rr.set_time("iteration", sequence=iteration)
        rr.log(f"{prefix}/particles", rr.Points2D(state.particles[:, dims]))
        if state.misfit_history:
            rr.log(f"{prefix}/diagnostics/misfit", rr.Scalars(state.misfit_history[-1]))
        if state.sigma_history:
            rr.log(f"{prefix}/diagnostics/sigma", rr.Scalars(state.sigma_history[-1]))
        rr.log(
            f"{prefix}/diagnostics/particle_variance",
            rr.Scalars(state.particle_variance_history[-1]),
        )
        if state.repulsion_ratio_history:
            rr.log(
                f"{prefix}/diagnostics/repulsion_ratio",
                rr.Scalars(state.repulsion_ratio_history[-1]),
            )

    return log_iteration
