# Adapted from ManiFlow (https://github.com/geyan21/ManiFlow_Policy), MIT License, Copyright (c) 2025 Ge Yan.
# See MANIFLOW_LICENSE in this folder.
"""Exponential moving average of the policy weights (the weights that are deployed)."""

from __future__ import annotations

import torch
from torch.nn.modules.batchnorm import _BatchNorm


class EMAModel:
    """EMA with warm-up: decay = 1 - (1 + step / inv_gamma) ** -power, clipped to [min_value, max_value]."""

    def __init__(self, model, update_after_step: int = 0, inv_gamma: float = 1.0, power: float = 2 / 3,
                 min_value: float = 0.0, max_value: float = 0.9999) -> None:
        self.averaged_model = model
        self.averaged_model.eval()
        self.averaged_model.requires_grad_(False)
        self.update_after_step = update_after_step
        self.inv_gamma, self.power = inv_gamma, power
        self.min_value, self.max_value = min_value, max_value
        self.decay = 0.0
        self.optimization_step = 0

    def get_decay(self, optimization_step: int) -> float:
        step = max(0, optimization_step - self.update_after_step - 1)
        value = 1 - (1 + step / self.inv_gamma) ** -self.power
        if step <= 0:
            return 0.0
        return max(self.min_value, min(value, self.max_value))

    @torch.no_grad()
    def step(self, new_model) -> None:
        self.decay = self.get_decay(self.optimization_step)
        for module, ema_module in zip(new_model.modules(), self.averaged_model.modules()):
            for param, ema_param in zip(module.parameters(recurse=False), ema_module.parameters(recurse=False)):
                if isinstance(module, _BatchNorm) or not param.requires_grad:
                    ema_param.copy_(param.to(dtype=ema_param.dtype).data)
                else:
                    ema_param.mul_(self.decay)
                    ema_param.add_(param.data.to(dtype=ema_param.dtype), alpha=1 - self.decay)
        self.optimization_step += 1
