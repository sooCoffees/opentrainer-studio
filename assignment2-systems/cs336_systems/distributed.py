from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.distributed as dist
import torch.nn as nn


def _dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def _world_size() -> int:
    return dist.get_world_size() if _dist_ready() else 1


def _broadcast_trainable_state(module: nn.Module) -> None:
    if not _dist_ready():
        return
    for tensor in module.state_dict().values():
        dist.broadcast(tensor, src=0)


def _average_gradients(parameters: Iterable[nn.Parameter]) -> None:
    if not _dist_ready():
        return
    world_size = dist.get_world_size()
    for param in parameters:
        if param.grad is None:
            continue
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad.div_(world_size)


class DistributedDataParallel(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        _broadcast_trainable_state(self.module)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        _average_gradients(self.module.parameters())


def _apply_mixed_precision_hooks(module: nn.Module, compute_dtype: torch.dtype) -> None:
    from cs336_basics.model import Embedding, Linear

    for child in module.modules():
        if not isinstance(child, (Linear, Embedding)):
            continue

        def make_fwd_pre(dtype):
            def hook(m, _inp):
                m._saved_fp32 = m.weight.data
                m.weight.data = m.weight.data.to(dtype)

            return hook

        def make_fwd_post():
            def hook(m, _inp, _out):
                m.weight.data = m._saved_fp32
                del m._saved_fp32
                m.weight.grad = None

            return hook

        child.register_forward_pre_hook(make_fwd_pre(compute_dtype))
        child.register_forward_hook(make_fwd_post())

        if isinstance(child, Linear):

            def make_bwd_pre(dtype):
                def hook(m, _grad_output):
                    m._saved_fp32_bwd = m.weight.data
                    m.weight.data = m.weight.data.to(dtype)
                    m.weight.grad = None

                return hook

            child.register_full_backward_pre_hook(make_bwd_pre(compute_dtype))

        def make_grad_hook(m, is_linear):
            def hook(param):
                if is_linear and hasattr(m, "_saved_fp32_bwd"):
                    m.weight.data = m._saved_fp32_bwd
                    del m._saved_fp32_bwd
                if param.grad is not None:
                    param.grad = param.grad.to(torch.float32)

            return hook

        child.weight.register_post_accumulate_grad_hook(make_grad_hook(child, isinstance(child, Linear)))


class FullyShardedDataParallel(nn.Module):
    def __init__(self, module: nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()
        self.module = module
        self.compute_dtype = compute_dtype
        _broadcast_trainable_state(self.module)
        if compute_dtype is not None:
            _apply_mixed_precision_hooks(self.module, compute_dtype)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        _average_gradients(self.module.parameters())

    def gather_full_params(self) -> dict[str, torch.Tensor]:
        return {name: param.detach().clone() for name, param in self.module.named_parameters()}


class ShardedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls: type[torch.optim.Optimizer], **kwargs):
        self.rank = dist.get_rank() if _dist_ready() else 0
        self.world_size = _world_size()
        self.all_params = list(params)
        self.local_params = [p for i, p in enumerate(self.all_params) if i % self.world_size == self.rank]
        self.optim = optimizer_cls(self.local_params, **kwargs)

    @property
    def param_groups(self):
        return self.optim.param_groups

    @property
    def state(self):
        return self.optim.state

    def zero_grad(self, set_to_none: bool = True):
        for param in self.all_params:
            param.grad = None if set_to_none else torch.zeros_like(param)

    def step(self, closure=None):
        loss = self.optim.step(closure)
        if _dist_ready():
            owner_by_param = {id(param): i % self.world_size for i, param in enumerate(self.all_params)}
            for param in self.all_params:
                dist.broadcast(param.data, src=owner_by_param[id(param)])
        return loss

    def state_dict(self):
        return self.optim.state_dict()

    def load_state_dict(self, state_dict):
        return self.optim.load_state_dict(state_dict)
