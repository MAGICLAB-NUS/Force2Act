"""Rotation helpers: XYZ Euler angles, the continuous 6-D rotation, and roll/pitch/yaw continuity.

Recorded Euler angles lie in [-pi, pi). A tool pointing down sits at roll ~ +/-pi, so consecutive samples
can jump by 2 pi. Each angle of the pose observation is instead mapped into the 2 pi window centred on its
training-set circular mean; the centre is stored in the normalizer (key RPY_CENTER_KEY), so training and
inference apply the same stateless map.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F

RPY_CENTER_KEY = "pose_rpy_center"


# ---------------------------------------------------------------------------------------------- torch
def euler_xyz_to_matrix(euler: torch.Tensor) -> torch.Tensor:
    """Roll-pitch-yaw -> Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    if euler.shape[-1] != 3:
        raise ValueError(f"Euler tensor must end in 3 values, got {euler.shape}")
    roll, pitch, yaw = euler.unbind(dim=-1)
    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    values = (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
              sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
              -sp, cp * sr, cp * cr)
    return torch.stack(values, dim=-1).reshape(*euler.shape[:-1], 3, 3)


def matrix_to_euler_xyz(matrix: torch.Tensor) -> torch.Tensor:
    """Rotation matrix -> canonical XYZ roll-pitch-yaw."""
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"rotation matrix must end in (3, 3), got {matrix.shape}")
    horizontal = torch.sqrt(matrix[..., 0, 0].square() + matrix[..., 1, 0].square())
    pitch = torch.atan2(-matrix[..., 2, 0], horizontal)
    regular_roll = torch.atan2(matrix[..., 2, 1], matrix[..., 2, 2])
    regular_yaw = torch.atan2(matrix[..., 1, 0], matrix[..., 0, 0])
    singular_roll = torch.atan2(-matrix[..., 1, 2], matrix[..., 1, 1])  # pitch = +/- pi/2
    singular = horizontal < 1e-6
    roll = torch.where(singular, singular_roll, regular_roll)
    yaw = torch.where(singular, torch.zeros_like(regular_yaw), regular_yaw)
    return torch.stack((roll, pitch, yaw), dim=-1)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """First two matrix columns (Zhou et al., 2019)."""
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"rotation matrix must end in (3, 3), got {matrix.shape}")
    return torch.cat((matrix[..., :, 0], matrix[..., :, 1]), dim=-1)


def rotation_6d_to_matrix(rotation: torch.Tensor) -> torch.Tensor:
    """Two 3-D vectors -> SO(3) by Gram-Schmidt."""
    if rotation.shape[-1] != 6:
        raise ValueError(f"rotation-6D tensor must end in 6 values, got {rotation.shape}")
    first, second = rotation[..., :3], rotation[..., 3:]
    first_norm = torch.linalg.vector_norm(first, dim=-1, keepdim=True)
    default_first = torch.zeros_like(first)
    default_first[..., 0] = 1.0
    basis_1 = torch.where(first_norm > 1e-6, first / first_norm.clamp_min(1e-6), default_first)
    second_orthogonal = second - (basis_1 * second).sum(dim=-1, keepdim=True) * basis_1
    second_norm = torch.linalg.vector_norm(second_orthogonal, dim=-1, keepdim=True)
    fallback_axis = F.one_hot(basis_1.abs().argmin(dim=-1), num_classes=3).to(rotation.dtype)
    fallback_second = F.normalize(fallback_axis - (fallback_axis * basis_1).sum(dim=-1, keepdim=True) * basis_1,
                                  dim=-1)
    basis_2 = torch.where(second_norm > 1e-6, second_orthogonal / second_norm.clamp_min(1e-6), fallback_second)
    basis_3 = torch.cross(basis_1, basis_2, dim=-1)
    return torch.stack((basis_1, basis_2, basis_3), dim=-1)


def xyzrpy_to_xyz_rotation_6d(pose: torch.Tensor) -> torch.Tensor:
    if pose.shape[-1] != 6:
        raise ValueError(f"xyz+rpy pose must end in 6 values, got {pose.shape}")
    return torch.cat((pose[..., :3], matrix_to_rotation_6d(euler_xyz_to_matrix(pose[..., 3:]))), dim=-1)


def xyz_rotation_6d_to_xyzrpy(pose: torch.Tensor) -> torch.Tensor:
    if pose.shape[-1] != 9:
        raise ValueError(f"xyz+rotation-6D pose must end in 9 values, got {pose.shape}")
    return torch.cat((pose[..., :3], matrix_to_euler_xyz(rotation_6d_to_matrix(pose[..., 3:]))), dim=-1)


# ------------------------------------------------------------------------------------- rpy continuity
def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + math.pi, 2 * math.pi) - math.pi


def rpy_to_continuous(pose: torch.Tensor, params) -> torch.Tensor:
    """Map the angle dimensions of pose into their stored 2 pi window."""
    mask = params["scale"].to(device=pose.device) > 0
    center = params["offset"].to(device=pose.device, dtype=pose.dtype)
    return torch.where(mask, center + wrap_to_pi(pose - center), pose)


def install_rpy_center(normalizer, pose: torch.Tensor, rpy_indices: Sequence[int]) -> torch.Tensor:
    """Fit per-angle circular means, store them in the normalizer, return the mapped poses."""
    from force2act.utils.normalizer import SingleFieldLinearNormalizer

    flat = pose.reshape(-1, pose.shape[-1]).float().cpu()
    indices = list(rpy_indices)
    mask = torch.zeros(flat.shape[-1])
    mask[indices] = 1.0
    center = torch.zeros(flat.shape[-1])
    angles = flat[:, indices]
    center[indices] = torch.atan2(angles.sin().mean(0), angles.cos().mean(0))
    entry = SingleFieldLinearNormalizer.create_manual(
        scale=mask, offset=center,
        input_stats_dict={"min": center - math.pi * mask, "max": center + math.pi * mask,
                          "mean": center, "std": mask.clone()})
    entry.requires_grad_(False)
    normalizer[RPY_CENTER_KEY] = entry
    return rpy_to_continuous(pose, normalizer.params_dict[RPY_CENTER_KEY])


# ---------------------------------------------------------------------------------------------- numpy
def quat_to_matrix(quaternion_wxyz) -> np.ndarray:
    q = np.asarray(quaternion_wxyz, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError(f"invalid quaternion {quaternion_wxyz}")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def quat_to_rpy(quaternion_wxyz) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion_wxyz, dtype=float) / np.linalg.norm(quaternion_wxyz)
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    return np.array([np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)),
                     pitch,
                     np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))])


def rpy_to_quat(rpy) -> np.ndarray:
    roll, pitch, yaw = np.asarray(rpy, dtype=float)
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])


def matrix_to_quat(rotation: np.ndarray) -> np.ndarray:
    m = np.asarray(rotation, dtype=float)
    trace = np.trace(m)
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.asarray(q)
    return q / np.linalg.norm(q)


def rpy_indices(pose_columns: Sequence[int]) -> list[int]:
    """Positions within the pose observation that hold roll/pitch/yaw (state columns 3-5)."""
    return [position for position, column in enumerate(pose_columns) if int(column) in (3, 4, 5)]
