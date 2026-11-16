"""Rigid-pose math. Quaternions are scalar first (w, x, y, z)."""

import numpy as np

def quat_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert a normalized w-x-y-z quaternion to a 3x3 rotation matrix."""
    q = np.asarray(quaternion, dtype=float)
    if q.shape != (4,):
        raise ValueError(f"quaternion must have shape (4,), got {q.shape}")
    norm = np.linalg.norm(q)
    if not np.isfinite(q).all() or norm < 1e-12:
        raise ValueError("quaternion must be non-zero")
    w, x, y, z = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def matrix_to_quat(rotation: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a normalized w-x-y-z quaternion."""
    R = np.asarray(rotation, dtype=float)
    if R.shape != (3, 3):
        raise ValueError(f"rotation_matrix must have shape (3, 3), got {R.shape}")
    trace = np.trace(R)
    if trace > 0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        q = np.array(
            [
                0.25 * scale,
                (R[2, 1] - R[1, 2]) / scale,
                (R[0, 2] - R[2, 0]) / scale,
                (R[1, 0] - R[0, 1]) / scale,
            ]
        )
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            scale = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            q = np.array(
                [
                    (R[2, 1] - R[1, 2]) / scale,
                    0.25 * scale,
                    (R[0, 1] + R[1, 0]) / scale,
                    (R[0, 2] + R[2, 0]) / scale,
                ]
            )
        elif i == 1:
            scale = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            q = np.array(
                [
                    (R[0, 2] - R[2, 0]) / scale,
                    (R[0, 1] + R[1, 0]) / scale,
                    0.25 * scale,
                    (R[1, 2] + R[2, 1]) / scale,
                ]
            )
        else:
            scale = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            q = np.array(
                [
                    (R[1, 0] - R[0, 1]) / scale,
                    (R[0, 2] + R[2, 0]) / scale,
                    (R[1, 2] + R[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    if q[0] < 0:
        q = -q
    return q / np.linalg.norm(q)


def rigid_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Build and validate one 4x4 rigid transform."""
    R = np.asarray(rotation, dtype=float)
    p = np.asarray(translation, dtype=float)
    if R.shape != (3, 3) or p.shape != (3,):
        raise ValueError("rotation and translation must have shapes (3, 3) and (3,)")
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-5) or not np.isclose(
        np.linalg.det(R), 1.0, atol=1e-5
    ):
        raise ValueError("rotation must be a proper orthonormal matrix")
    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = p
    return transform


def quaternion_slerp(q0: np.ndarray, q1: np.ndarray, fraction: float) -> np.ndarray:
    """Shortest-path interpolation for w-x-y-z quaternions."""
    a = np.asarray(q0, dtype=float) / np.linalg.norm(q0)
    b = np.asarray(q1, dtype=float) / np.linalg.norm(q1)
    dot = float(a @ b)
    if dot < 0:
        b, dot = -b, -dot
    if dot > 0.9995:
        q = a + fraction * (b - a)
        return q / np.linalg.norm(q)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    return (np.sin((1 - fraction) * theta) * a + np.sin(fraction * theta) * b) / np.sin(theta)


