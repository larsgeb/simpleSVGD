"""PyTorch autograd bridge for the SVGD kernel term, used by update_torch."""

from collections.abc import Callable
from typing import Any

import numpy.typing as npt
import torch  # ty: ignore[unresolved-import] -- optional extra

from ._typing import FloatDType, KernelFn


def torch_wrapper(
    g_fn: Callable[[npt.NDArray[FloatDType]], npt.NDArray[FloatDType]],
    kernel: KernelFn[FloatDType],
) -> Callable[..., Any]:
    """Build a torch.autograd.Function applying the SVGD kernel term as a custom gradient."""
    class _InternalClass(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, input_tensor: Any) -> Any:  # noqa: ANN401 -- torch context/Tensor types unavailable without installing torch for type-checking
            ctx.save_for_backward(input_tensor)
            return input_tensor

        @staticmethod
        def backward(ctx: Any, grad_output: Any) -> Any:  # noqa: ANN401 -- same as forward()
            (input_tensor,) = ctx.saved_tensors
            dtype = input_tensor.dtype

            kernel_matrix, kernel_grad = kernel(input_tensor.numpy(), -1)

            kernel_matrix_t = torch.from_numpy(kernel_matrix).type(dtype)
            kernel_grad_t = torch.from_numpy(kernel_grad).type(dtype)

            return grad_output * (
                kernel_matrix_t
                @ torch.from_numpy(g_fn(input_tensor.numpy())).type(dtype)
                - kernel_grad_t
            ).type(dtype)

    return _InternalClass.apply
