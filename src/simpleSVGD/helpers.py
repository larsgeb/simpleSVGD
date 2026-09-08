"""PyTorch autograd bridge for the SVGD kernel term, used by update_torch."""

from collections.abc import Callable
from typing import Any

import torch as _torch  # ty: ignore[unresolved-import] -- optional extra


def TorchWrapper(
    g_fn: Callable[[Any], Any], kernel: Callable[..., tuple[Any, Any]]
) -> Callable[..., Any]:
    """Build a torch.autograd.Function applying the SVGD kernel term as a custom gradient."""
    class _InternalClass(_torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, input_tensor: Any) -> Any:
            ctx.save_for_backward(input_tensor)
            return input_tensor.type(_torch.FloatTensor)

        @staticmethod
        def backward(ctx: Any, grad_output: Any) -> Any:
            (input_tensor,) = ctx.saved_tensors

            kxy, dxkxy = kernel(input_tensor.numpy(), h=-1)

            K = _torch.from_numpy(kxy).type(_torch.FloatTensor)
            dk = _torch.from_numpy(dxkxy).type(_torch.FloatTensor)

            return grad_output * (
                K
                @ _torch.from_numpy(g_fn(input_tensor.numpy())).type(
                    _torch.FloatTensor
                )
                - dk
            ).type(_torch.FloatTensor)

    return _InternalClass.apply
