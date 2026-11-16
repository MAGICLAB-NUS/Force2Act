"""Relative force: the global policy sees and predicts force in units of its recent magnitude.

    s = max(mean_i |F_i|, s_min)          over the wrench history (0.5 s)
    input:  F_i / s                       output:  F_k = s * u_k

so the policy reasons about the direction and timing of contact rather than its level in newtons; the normal
force level is supplied by MoFE. A checkpoint stores s_min in its normalizer (key KEY); a checkpoint without
the entry uses absolute force.
"""

from __future__ import annotations

import numpy as np
import torch

KEY = "relative_force"


def install(normalizer, floor: float) -> None:
    """Mark a normalizer as relative-force with floor s_min (N)."""
    from force2act.utils.normalizer import SingleFieldLinearNormalizer

    offset = torch.tensor([float(floor), 0.0, 0.0, 0.0, 1.0])  # [s_min, unused x3, window-mean flag]
    entry = SingleFieldLinearNormalizer.create_manual(
        scale=torch.ones(len(offset)), offset=offset,
        input_stats_dict={"min": offset, "max": offset, "mean": offset, "std": torch.ones(len(offset))})
    entry.requires_grad_(False)
    normalizer[KEY] = entry


def floor(normalizer) -> float | None:
    """s_min of a relative-force checkpoint, or None for absolute force."""
    if KEY not in normalizer.params_dict:
        return None
    values = normalizer.params_dict[KEY]["offset"].detach().cpu().tolist()
    if len(values) != 5 or values[4] != 1.0:
        raise ValueError(f"unsupported relative-force setting {values}")
    return float(values[0])


def scale(force_history: torch.Tensor, floor: float) -> torch.Tensor:
    """force_history [B, T, 6] -> s [B, 1]."""
    return force_history[..., :3].norm(dim=-1).mean(dim=1, keepdim=True).clamp_min(floor)


def fit_arrays(force: np.ndarray, episodes: np.ndarray, history: int, offsets: list[int],
               floor: float) -> tuple[np.ndarray, np.ndarray]:
    """Relative inputs [N * history, D] and targets [N * H, D] of every row (episode order), for fitting the
    normalizers; windows are clamped at episode boundaries as in the dataset."""
    inputs, targets = [], []
    for episode in np.unique(episodes):
        rows = np.flatnonzero(episodes == episode)
        t = np.arange(len(rows))
        last = len(rows) - 1
        past = np.clip(t[:, None] + np.arange(1 - history, 1)[None], 0, last)
        future = np.clip(t[:, None] + np.asarray(offsets)[None], 0, last)
        f = torch.from_numpy(force[rows]).float()
        s = scale(f[past], floor)
        inputs.append((f[past] / s[:, None]).reshape(-1, f.shape[1]).numpy())
        targets.append((f[future] / s[:, None]).reshape(-1, f.shape[1]).numpy())
    return np.concatenate(inputs), np.concatenate(targets)
