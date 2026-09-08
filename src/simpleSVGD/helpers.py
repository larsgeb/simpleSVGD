from typing import Any, Callable

import torch as _torch  # ty: ignore[unresolved-import] -- optional extra


def TorchWrapper(
    g_fn: Callable[[Any], Any], kernel: Callable[..., tuple[Any, Any]]
) -> Callable[..., Any]:
    class _internalClass(_torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, input: Any) -> Any:
            ctx.save_for_backward(input)
            return input.type(_torch.FloatTensor)

        @staticmethod
        def backward(ctx: Any, grad_output: Any) -> Any:
            (input,) = ctx.saved_tensors

            kxy, dxkxy = kernel(input.numpy(), h=-1)

            K = _torch.from_numpy(kxy).type(_torch.FloatTensor)
            dk = _torch.from_numpy(dxkxy).type(_torch.FloatTensor)

            return grad_output * (
                K
                @ _torch.from_numpy(g_fn(input.numpy())).type(
                    _torch.FloatTensor
                )
                - dk
            ).type(_torch.FloatTensor)

    return _internalClass.apply
