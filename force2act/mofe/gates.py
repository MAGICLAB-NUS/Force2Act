"""MoFE inputs: the tangential load, the lateral wall force, contact-mode gates and the router's force features.

Gates (which experts are admissible at a step), from the measured force F, tool velocity v and tool axis:

    pushing      moving with F within push_angle of the motion in at least 4 of the last 8 steps
    sticking_1   moving with F more than push_angle from the motion, reaction within 45 deg of the support
                 normal (pressing onto the object; the support normal is the upward vertical)
    sticking_2   as sticking_1, reaction more than 45 deg from the support normal (pressing against a face)
    sliding_*    always
    insertion    |F| >= 1 N within 50 deg of the tool axis and, while moving, motion within 30 deg of it
"""

from __future__ import annotations

import numpy as np

from force2act.control.contact import contact_split

EXPERTS = ["pushing", "sticking_1", "sticking_2", "sliding_1", "sliding_2", "sliding_3", "insertion"]
NUM_INPUTS = 2 + len(EXPERTS)                      # (T, |F_perp|, admissible[7])
UP_COS = float(np.cos(np.radians(45.0)))           # reaction within 45 deg of the support normal
MOTION_ALIGN_COS = float(np.cos(np.radians(30.0)))  # motion within 30 deg of the tool axis
FORCE_ALIGN_COS = float(np.cos(np.radians(50.0)))   # force within 50 deg of the tool axis
PUSH_HISTORY, PUSH_MIN = 8, 4                       # pushing: at least 4 of the last 8 steps


def tool_z_axis(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """Tool axis (local z) of w-x-y-z quaternions [..., 4] -> [N, 3]."""
    q = np.asarray(quaternion_wxyz, float).reshape(-1, 4)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)], axis=1)


def physical_inputs(force: np.ndarray, velocity: np.ndarray, tool_z: np.ndarray, min_speed: float,
                    push_angle: float, gates: bool = True) -> np.ndarray:
    """Consecutive samples of one sequence: force, velocity, tool axis [T, 3] -> [T, 9] = (T, |F_perp|, gates).

    |F_perp| (force across the tool axis) is kept only when the press is along the tool axis (then it is the
    wall load, never the press itself). gates=False makes every expert admissible (ablation)."""
    force, velocity = np.asarray(force, float)[:, :3], np.asarray(velocity, float)[:, :3]
    tangential, _ = contact_split(force, velocity, min_speed, push_angle)
    speed = np.linalg.norm(velocity, axis=1)
    axial = (force * tool_z).sum(1)
    fperp = np.linalg.norm(force - axial[:, None] * tool_z, axis=1)
    fnorm = np.linalg.norm(force, axis=1)
    moving = speed >= min_speed
    motion_along = np.abs((velocity * tool_z).sum(1)) / np.maximum(speed, 1e-9)
    force_along = np.abs(axial) / np.maximum(fnorm, 1e-9)
    along_axis = (fnorm >= 1.0) & (force_along >= FORCE_ALIGN_COS) & (~moving | (motion_along >= MOTION_ALIGN_COS))
    pushing = tangential <= 0.0                  # force within push_angle of the motion, or not moving
    up = force[:, 2] / np.maximum(fnorm, 1e-9) >= UP_COS
    push_move = (pushing & moving).astype(float)
    sustained = np.array([push_move[max(0, i - PUSH_HISTORY + 1):i + 1].sum() >= PUSH_MIN
                          for i in range(len(push_move))])
    admissible = np.stack([sustained, ~pushing & up, ~pushing & ~up, np.ones_like(pushing), np.ones_like(pushing),
                           np.ones_like(pushing), along_axis], axis=1)
    if not gates:
        admissible = np.ones_like(admissible)
    return np.concatenate([np.stack([tangential, np.where(along_axis, fperp, 0.0)], axis=1), admissible],
                          axis=1).astype(np.float32)


def router_features(wrench: np.ndarray, velocity: np.ndarray, min_speed: float, push_angle: float,
                    floor: float = 1.0, include_torque: bool = True) -> np.ndarray:
    """World wrench [..., T, 6] and velocity [..., T, >=3], oldest first -> per-sample features, flattened:
    |Ft| (N), speed (m/s), lever arm M / |F| (m, with torque), F / |F|. |F| is floored at `floor` N."""
    wrench, velocity = np.asarray(wrench, float), np.asarray(velocity, float)[..., :3]
    if wrench.shape[-1] not in ((6,) if include_torque else (3, 6)):
        raise ValueError("MoFE needs a 6-D wrench with torque, or a 3-D force without torque")
    force = wrench[..., :3]
    tangential, _ = contact_split(force, velocity, min_speed, push_angle)
    norm = np.maximum(np.linalg.norm(force, axis=-1, keepdims=True), floor)
    parts = [tangential[..., None], np.linalg.norm(velocity, axis=-1, keepdims=True)]
    if include_torque:
        parts.append(wrench[..., 3:6] / norm)
    per_sample = np.concatenate([*parts, force / norm], axis=-1)
    return per_sample.reshape(*per_sample.shape[:-2], -1).astype(np.float32)
