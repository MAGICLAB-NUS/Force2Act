"""Small, NumPy-only policy boundary, independent of training frameworks."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class Action:
    """One TCP pose, joint pose, or joint torque command for the selected mode.

    Quaternion order is wxyz. Gripper closure is 0=open, 1=closed; omitted
    fields keep their current targets. Action() holds the current command.
    joint_velocities is rad/s; tcp_velocity is base-frame linear/angular
    velocity (m/s, rad/s). tcp_wrench is the desired sensed environment-on-tool
    wrench at TCP in the force frame (N, N m), following Flexiv RDK.
    Only force-controlled axes use this target. tcp_acceleration is base-frame
    linear/angular acceleration (m/s^2, rad/s^2).
    joint_torques is N m, with gravity compensation configured separately.
    Under hybrid control, force_control_axes (6 bools) selects the axes of the
    force frame that follow tcp_wrench, and force_frame_rotation_wxyz is that
    frame in world coordinates; both stay in effect until an action sets them.
    """

    tcp_position: np.ndarray | None = None
    tcp_quaternion_wxyz: np.ndarray | None = None
    joint_positions: np.ndarray | None = None
    gripper: float | None = None
    joint_velocities: np.ndarray | None = None
    joint_torques: np.ndarray | None = None
    tcp_velocity: np.ndarray | None = None
    tcp_wrench: np.ndarray | None = None
    tcp_acceleration: np.ndarray | None = None
    force_control_axes: np.ndarray | None = None
    force_frame_rotation_wxyz: np.ndarray | None = None

    def __post_init__(self):
        has_tcp = self.tcp_position is not None or self.tcp_quaternion_wxyz is not None
        if has_tcp and self.joint_positions is not None:
            raise ValueError("choose TCP or joint control in one action")
        if self.joint_torques is not None and (has_tcp or self.joint_positions is not None
                                             or self.joint_velocities is not None
                                             or self.tcp_velocity is not None or self.tcp_wrench is not None
                                             or self.tcp_acceleration is not None):
            raise ValueError("joint torque action cannot contain motion or wrench targets")
        if has_tcp and self.joint_velocities is not None:
            raise ValueError("joint_velocities cannot accompany TCP targets")
        if (self.tcp_position is None) != (self.tcp_quaternion_wxyz is None):
            raise ValueError("TCP position and quaternion must be supplied together")
        for name, size in (("tcp_position", 3), ("tcp_quaternion_wxyz", 4), ("joint_positions", 7),
                           ("joint_velocities", 7), ("joint_torques", 7),
                           ("tcp_velocity", 6), ("tcp_wrench", 6), ("tcp_acceleration", 6)):
            value = getattr(self, name)
            if value is not None:
                value = np.asarray(value, dtype=float)
                if value.shape != (size,) or not np.isfinite(value).all():
                    raise ValueError(f"{name} must contain {size} finite values")
                setattr(self, name, value.copy())
        if self.tcp_quaternion_wxyz is not None:
            norm = np.linalg.norm(self.tcp_quaternion_wxyz)
            if norm < 1e-12:
                raise ValueError("TCP quaternion must be nonzero")
            self.tcp_quaternion_wxyz /= norm
        if self.gripper is not None and not 0 <= self.gripper <= 1:
            raise ValueError("gripper closure must be in [0, 1]")


class Policy(Protocol):
    def reset(self) -> None: ...

    def predict(self, observation: dict) -> Action: ...


class HoldPolicy:
    """Runnable example that holds the initialized arm in place."""

    def reset(self):
        pass

    def predict(self, observation):
        return Action()
