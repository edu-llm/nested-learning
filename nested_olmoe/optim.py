from __future__ import annotations

import math
from typing import Iterable

import torch


class M3LiteAdamW(torch.optim.Optimizer):
    """A practical multi-timescale AdamW variant.

    This is not the paper's full M3 optimizer: it omits Newton-Schulz
    orthogonalization so it can run safely on adapter parameters of arbitrary
    shape. It preserves the NL test we need here: a fast first moment, a slow
    gradient memory updated at a lower frequency, and Adam-style second moment.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        *,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        slow_beta: float = 0.99,
        slow_interval: int = 16,
        slow_weight: float = 0.25,
    ) -> None:
        if slow_interval < 1:
            raise ValueError("slow_interval must be >= 1")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "slow_beta": slow_beta,
            "slow_interval": slow_interval,
            "slow_weight": slow_weight,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            slow_beta = group["slow_beta"]
            slow_interval = group["slow_interval"]
            slow_weight = group["slow_weight"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("M3LiteAdamW does not support sparse gradients")

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)
                    state["slow_avg"] = torch.zeros_like(param)
                    state["slow_buffer"] = torch.zeros_like(param)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                slow_avg = state["slow_avg"]
                slow_buffer = state["slow_buffer"]
                state["step"] += 1
                step = state["step"]

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                slow_buffer.add_(grad)
                if step % slow_interval == 0:
                    slow_avg.mul_(slow_beta).add_(slow_buffer, alpha=(1.0 - slow_beta) / slow_interval)
                    slow_buffer.zero_()

                if weight_decay != 0.0:
                    param.mul_(1.0 - lr * weight_decay)

                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                step_size = lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                update = exp_avg.add(slow_avg, alpha=slow_weight)
                param.addcdiv_(update, denom, value=-step_size)

        return loss
