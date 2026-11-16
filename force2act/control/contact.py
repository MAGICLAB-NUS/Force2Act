"""Contact force split and the per-step normal-force command (numpy only)."""

from __future__ import annotations

from typing import Any

import numpy as np


def contact_split(force: np.ndarray, velocity: np.ndarray, min_speed: float,
                  push_angle: float) -> tuple[np.ndarray, np.ndarray]:
    """(|Ft|, |Fn|) per sample: force [..., 3] split along the tool velocity [..., 3].

    Ft is the force along the motion, Fn the force across it. When the tool is not moving, or pushes (force
    within push_angle of the motion), the whole force is normal and Ft = 0. Training targets, the gates and
    the controller all use this split.
    """
    force, velocity = np.asarray(force, float), np.asarray(velocity, float)
    speed = np.linalg.norm(velocity, axis=-1, keepdims=True)
    direction = np.where(speed >= min_speed, velocity / np.maximum(speed, 1e-9), 0.0)
    along = np.abs((force * direction).sum(-1))
    total = np.linalg.norm(force, axis=-1)
    across = np.sqrt(np.maximum(total ** 2 - along ** 2, 0.0))
    pushing = (speed[..., 0] < min_speed) | (across < np.sin(np.radians(push_angle)) * total)
    return np.where(pushing, 0.0, along), np.where(pushing, total, across)


class NormalForceCommand:
    """Commanded normal force along the contact frame's z, updated at every control step:

        target_k = w * (alpha * Ft_k + beta) + (1 - w) * Fn_policy_k
        Fn_k     = clip(Fn_{k-1} + blend * (target_k - Fn_{k-1}), 0, f_max)

    with Ft_k measured at this step, (alpha, beta) the selected MoFE expert and Fn_policy_k the global
    policy's normal force. On first contact Fn_{-1} is the measured normal force.
    """

    def __init__(self) -> None:
        self.previous: float | None = None
        self.last: dict[str, float] = {}

    def reset(self) -> None:
        self.previous = None

    def step(self, force_world: np.ndarray, velocity_world: np.ndarray, rotation: np.ndarray,
             params: dict[str, Any], policy_normal: float | None = None) -> float:
        force = np.asarray(force_world, float)[:3]
        rotation = np.asarray(rotation, float)
        tangential, _ = contact_split(force, np.asarray(velocity_world, float)[:3],
                                      float(params["min_speed"]), float(params["push_angle"]))
        tangential = float(tangential)
        measured_normal = float(force @ rotation[:, 2])
        maximum = float(params["f_max"])
        law = float(params["alpha"]) * tangential + float(params["beta"])
        weight = 1.0 if policy_normal is None else float(params.get("w", 1.0))
        target = weight * law + (1.0 - weight) * (0.0 if policy_normal is None else float(policy_normal))
        previous = float(np.clip(measured_normal, 0.0, maximum)) if self.previous is None else self.previous
        normal = float(np.clip(previous + float(params.get("blend", 1.0)) * (target - previous), 0.0, maximum))
        self.previous = normal
        self.last = {"tangential": tangential, "measured_normal": measured_normal, "law_normal": law,
                     "policy_normal": policy_normal, "target_normal": target, "commanded_normal": normal}
        return normal
