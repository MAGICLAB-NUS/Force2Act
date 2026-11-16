"""Real-time chunking (Black, Galliker and Levine, 2025; arXiv 2506.07339), at inference only.

Each new chunk is sampled as an inpainting of the previous chunk's remaining actions. At every Euler step the
one-step estimate of the clean chunk, x1 = x_t + (1 - t) v_t, is compared with the previous chunk on a
prefix-weighted mask, and the error is pulled back through the network with a vector-Jacobian product:

    v_t <- v_t + min(c * inv_r2, beta) * J^T [W (Y - x1)],   c = (1 - t) / t,   inv_r2 = (t^2 + (1 - t)^2) / (1 - t)^2

t = 0 is noise and t = 1 is data.
"""

from __future__ import annotations

import math

import torch


def prefix_weights(start: int, end: int, total: int, schedule: str = "exp") -> torch.Tensor:
    """1 before start, decaying to 0 at end, 0 after (schedule exp, linear, ones or zeros)."""
    start = min(start, end)
    index = torch.arange(total, dtype=torch.float32)
    if schedule == "ones":
        weights = torch.ones(total)
    elif schedule == "zeros":
        weights = (index < start).float()
    else:
        weights = torch.clamp((start - 1 - index) / (end - start + 1) + 1, 0.0, 1.0)
        if schedule == "exp":
            weights = weights * torch.expm1(weights) / (math.e - 1)
        elif schedule != "linear":
            raise ValueError(f"unknown RTC schedule {schedule!r}")
    return torch.where(index >= end, torch.zeros_like(weights), weights)


def shift_previous_chunk(previous: torch.Tensor, executed: int) -> torch.Tensor:
    """The previous normalized chunk [H, D] on the new chunk's time indices (index k <- old k + executed)."""
    target = torch.zeros_like(previous)
    keep = previous.shape[0] - executed
    if keep > 0:
        target[:keep] = previous[executed:]
    return target


def guided_sample(action_head, steps: int, x0: torch.Tensor, target: torch.Tensor, weights: torch.Tensor,
                  max_guidance_weight: float, **condition) -> torch.Tensor:
    """Euler sampling with guidance toward target [B, H, D]; weights [B, H] (zero rows sample freely)."""
    dt = 1.0 / steps
    x = x0.detach().clone()
    weights = weights.to(x.device, x.dtype)[..., None]
    target = target.to(x.device, x.dtype)
    for i in range(steps):
        t = i / steps
        ti = torch.full((x.shape[0],), t, device=x.device, dtype=x.dtype)
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            velocity = action_head(x_in, ti, **condition)
            estimate = x_in + (1.0 - t) * velocity
            error = (target - estimate) * weights
            correction, = torch.autograd.grad(estimate, x_in, grad_outputs=error)
        inv_r2 = (t ** 2 + (1.0 - t) ** 2) / ((1.0 - t) ** 2)
        c = (1.0 - t) / t if t > 0 else math.inf
        guidance = min(c * inv_r2, max_guidance_weight)
        x = (x + dt * (velocity.detach() + guidance * correction)).detach()
    return x
