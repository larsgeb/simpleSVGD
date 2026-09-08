"""Smoke tests for update_torch, the legacy PyTorch bridge (requires torch)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import simplesvgd  # noqa: E402


class TestUpdateTorch:
    def test_converges_toward_target_mean(self):
        rng = np.random.default_rng(0)
        true_mean = np.array([2.0, -1.0])
        x0 = rng.normal(0, 3, (50, 2))

        def grad_fn(x):
            return x - true_mean

        result = simplesvgd.update_torch(
            x0,
            grad_fn,
            torch.optim.Adam,
            optimizer_parameters={"lr": 0.1},
            n_iter=200,
            disable_progressbar=True,
        )

        assert result.shape == x0.shape
        initial_error = np.linalg.norm(np.mean(x0, axis=0) - true_mean)
        final_error = np.linalg.norm(np.mean(result, axis=0) - true_mean)
        assert final_error < initial_error * 0.5

    def test_scheduler_is_stepped(self):
        """A passed-in scheduler should actually decay the optimizer's LR."""
        rng = np.random.default_rng(1)
        x0 = rng.normal(size=(10, 2))
        captured = {}

        class _CapturingSGD(torch.optim.SGD):
            def __init__(self, params, **kwargs):
                super().__init__(params, **kwargs)
                captured["optimizer"] = self

        def grad_fn(x):
            return x

        # update_torch constructs the optimizer internally, so a scheduler
        # needs a handle to that instance -- captured via the subclass above,
        # then wired in through a scheduler stand-in that steps it lazily.
        class _LazyStepLR:
            def __init__(self, gamma):
                self.gamma = gamma
                self._scheduler = None

            def step(self):
                if self._scheduler is None:
                    self._scheduler = torch.optim.lr_scheduler.StepLR(
                        captured["optimizer"], step_size=1, gamma=self.gamma
                    )
                else:
                    self._scheduler.step()

        result = simplesvgd.update_torch(
            x0,
            grad_fn,
            _CapturingSGD,
            optimizer_parameters={"lr": 1.0},
            schedulers=[_LazyStepLR(gamma=0.5)],
            n_iter=4,
            disable_progressbar=True,
        )

        assert result.shape == x0.shape
        final_lr = captured["optimizer"].param_groups[0]["lr"]
        assert final_lr < 1.0

    def test_animate_true_runs_end_to_end(self):
        mpl = pytest.importorskip("matplotlib")
        mpl.use("Agg")

        rng = np.random.default_rng(2)
        x0 = rng.normal(size=(10, 2))

        def grad_fn(x):
            return x

        result = simplesvgd.update_torch(
            x0,
            grad_fn,
            torch.optim.SGD,
            optimizer_parameters={"lr": 0.1},
            n_iter=3,
            animate=True,
            dimensions_to_plot=[0, 1],
            disable_progressbar=True,
        )

        assert result.shape == x0.shape

    def test_none_gradient_fn_raises(self):
        rng = np.random.default_rng(6)
        x0 = rng.normal(size=(5, 2))
        with pytest.raises(ValueError, match="cannot be None"):
            simplesvgd.update_torch(x0, None, torch.optim.SGD)  # ty: ignore[invalid-argument-type]

    def test_omitting_optimizer_parameters_and_schedulers_uses_defaults(self):
        rng = np.random.default_rng(4)
        x0 = rng.normal(size=(5, 2))

        def grad_fn(x):
            return x

        result = simplesvgd.update_torch(
            x0, grad_fn, torch.optim.SGD, n_iter=3, disable_progressbar=True
        )
        assert result.shape == x0.shape

    def test_keyboard_interrupt_returns_without_propagating(self):
        rng = np.random.default_rng(5)
        x0 = rng.normal(size=(5, 2))
        calls = {"n": 0}

        def grad_fn(x):
            calls["n"] += 1
            if calls["n"] == 2:
                raise KeyboardInterrupt
            return x

        result = simplesvgd.update_torch(
            x0,
            grad_fn,
            torch.optim.SGD,
            optimizer_parameters={"lr": 0.1},
            n_iter=100,
            disable_progressbar=True,
        )
        assert result.shape == x0.shape
        assert calls["n"] == 2


class TestDtypePreservation:
    """torch_wrapper used to hardcode torch.FloatTensor (float32), silently
    downcasting float64 input regardless of x0's actual dtype (issue #4)."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_result_dtype_matches_input_dtype(self, dtype):
        rng = np.random.default_rng(7)
        x0 = rng.normal(size=(10, 2)).astype(dtype)

        def grad_fn(x):
            return x

        result = simplesvgd.update_torch(
            x0,
            grad_fn,
            torch.optim.SGD,
            optimizer_parameters={"lr": 0.1},
            n_iter=3,
            disable_progressbar=True,
        )
        assert result.dtype == dtype
