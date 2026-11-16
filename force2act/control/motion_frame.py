"""Motion frame: the contact frame each step of a chunk is executed in.

Step 0 comes from the sensor: x is the net tool motion (mean measured velocity over velocity_window samples);
in contact, z is the part of the measured force across the motion. When the force lies within push_angle of the
motion (pushing), or the tool does not move, z is the force direction. Later steps turn z with the policy's
predicted force and take x from the planned motion. In contact only z is force-controlled; x and y follow the
planned poses.

Contact is decided once per chunk and is sticky: it starts at the first sample of at least contact_force and
holds until the force has stayed below it for contact_release_steps samples.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import numpy as np

from force2act.utils.rotation import matrix_to_quat


@dataclass(frozen=True)
class ControlConfig:
    contact_force: float = 1.0       # N of measured force that counts as contact
    force_window: int = 3            # latest force samples; the strongest one sets the frame
    contact_release_steps: int = 8   # sticky contact window (samples)
    min_motion: float = 1e-3         # m of chunk displacement that defines x in free space when not moving
    f_max: float = 60.0              # N, limit of the commanded normal force
    handover_steps: int = 3          # steps from the measured to the commanded normal force (without MoFE)
    velocity_window: int = 8         # velocity samples averaged for x
    min_speed: float = 0.005         # m/s of net motion that defines x
    push_angle: float = 15.0         # deg: force this close to the motion is pushing
    min_step_motion: float = 2e-4    # m of planned motion in one step that defines that step's x
    blend: float = 1.0               # per-step update rate of the normal-force command
    w: float = 0.8                   # share of MoFE in the normal-force target (0: global policy only)

    @classmethod
    def from_dict(cls, values: dict[str, Any] | None) -> "ControlConfig":
        names = {f.name: f.type for f in fields(cls)}
        unknown = set(values or {}) - set(names)
        if unknown:
            raise ValueError(f"unknown control settings {sorted(unknown)}")
        config = cls(**{key: type(getattr(cls, key))(value) for key, value in dict(values or {}).items()})
        if (config.force_window < 1 or config.velocity_window < 1 or config.contact_release_steps < 1
                or config.handover_steps < 0):
            raise ValueError("control windows must be positive and handover_steps nonnegative")
        return config


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def _perpendicular(axis: np.ndarray, preferred: np.ndarray) -> np.ndarray:
    for candidate in (preferred, np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])):
        rest = candidate - (candidate @ axis) * axis
        if np.linalg.norm(rest) > 1e-6:
            return _unit(rest)
    raise ValueError("cannot build a perpendicular axis")


def _turn(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation taking unit vector a onto unit vector b."""
    axis, cosine = np.cross(a, b), float(a @ b)
    if cosine < -1.0 + 1e-9:
        perpendicular = _perpendicular(a, np.array([1.0, 0.0, 0.0]))
        return 2.0 * np.outer(perpendicular, perpendicular) - np.eye(3)
    skew = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + skew + skew @ skew / (1.0 + cosine)


def _sensor_frame(force, velocity, displacement, config: ControlConfig):
    """Step-0 axes (x, z) and contact, from the measurements."""
    in_contact = np.linalg.norm(force) >= config.contact_force
    if np.linalg.norm(velocity) >= config.min_speed:
        motion = _unit(velocity)
    elif not in_contact and np.linalg.norm(displacement) >= config.min_motion:
        motion = _unit(displacement)  # free space only: in contact the plan may point into the surface
    else:
        motion = None
    if not in_contact:
        x_axis = motion if motion is not None else np.array([1.0, 0.0, 0.0])
        return x_axis, _perpendicular(x_axis, np.array([0.0, 0.0, 1.0])), False
    if motion is None:
        z_axis = _unit(force)
        return _perpendicular(z_axis, np.array([1.0, 0.0, 0.0])), z_axis, True
    normal = force - (force @ motion) * motion
    if np.linalg.norm(normal) >= np.sin(np.radians(config.push_angle)) * np.linalg.norm(force):
        return motion, _unit(normal), True
    z_axis = _unit(force)  # pushing: compliance along the force
    return _perpendicular(z_axis, motion), z_axis, True


def motion_frame(current_position: np.ndarray, chunk_positions: np.ndarray, measured_forces: np.ndarray,
                 measured_velocities: np.ndarray, predicted_forces: np.ndarray,
                 config: ControlConfig) -> dict[str, Any]:
    """Contact frames, force-control mask and the policy's normal force for one executed chunk.

    current_position [3], chunk_positions [H, 3], measured_forces [T, 3] and measured_velocities [T, 3] (world,
    oldest first; environment on tool), predicted_forces [H, 3] (the policy's world force at each step).
    Returns rotations [H, 3, 3] (columns: local x, y, z in world coordinates) with their quaternions, the mask
    (local z in contact), the policy's normal force per step and, without MoFE, the planned local wrench.
    """
    current = np.asarray(current_position, float)
    positions = np.asarray(chunk_positions, float)
    displacement = positions[-1] - current
    forces = np.asarray(measured_forces, float)
    recent = forces[-config.force_window:]
    force = recent[np.argmax(np.linalg.norm(recent, axis=1))]
    held = forces[-config.contact_release_steps:]
    strongest_held = held[np.argmax(np.linalg.norm(held, axis=1))]
    if np.linalg.norm(force) < config.contact_force <= np.linalg.norm(strongest_held):
        force = strongest_held
    velocity = np.asarray(measured_velocities, float)[-config.velocity_window:].mean(axis=0)
    predicted = np.asarray(predicted_forces, float)
    x0, z0, contact = _sensor_frame(force, velocity, displacement, config)
    steps = np.diff(np.vstack((current, positions)), axis=0)
    rotations = []
    for k, step in enumerate(steps):
        turn = np.eye(3)
        if contact and min(np.linalg.norm(predicted[0]), np.linalg.norm(predicted[k])) >= config.contact_force:
            turn = _turn(_unit(predicted[0]), _unit(predicted[k]))
        if contact:
            z_axis = turn @ z0
            motion = step - (step @ z_axis) * z_axis
            x_axis = _unit(motion) if np.linalg.norm(motion) >= config.min_step_motion else turn @ x0
        else:
            x_axis = _unit(step) if np.linalg.norm(step) >= config.min_step_motion else x0
            z_axis = _perpendicular(x_axis, np.array([0.0, 0.0, 1.0]))
        rotations.append(np.column_stack((x_axis, np.cross(z_axis, x_axis), z_axis)))
    rotations = np.stack(rotations)

    local_force = rotations[0].T @ force
    policy_local = np.einsum("hji,hj->hi", rotations, predicted)  # R_k^T F_k
    policy_normal = np.clip(policy_local[:, 2], 0.0, config.f_max)
    # Without MoFE the push follows the policy's own normal force, handed over from the measured one.
    commanded = np.clip(policy_local[:, 2], 0.0, config.f_max)
    handover_start = float(np.clip(local_force[2], 0.0, config.f_max))
    index = np.arange(1, len(predicted) + 1)
    share = np.minimum(index / config.handover_steps, 1.0) if config.handover_steps else np.ones(len(index))
    wrench = np.zeros((len(predicted), 6))
    if contact:
        wrench[:, 2] = handover_start + share * (commanded - handover_start)
    return {
        "rotations": rotations,
        "quaternions": np.stack([matrix_to_quat(r) for r in rotations]),
        "mask": np.array([False, False, contact, False, False, False]),
        "contact": contact,
        "policy_normal": policy_normal,
        "planned_wrench": wrench,
    }
