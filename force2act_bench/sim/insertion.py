"""Geometric insertion progress, expressed at the plug tip (commanded TCP)."""

import numpy as np

from .geometry import quat_to_matrix


def insertion_state(config, position, quaternion):
    """Entrance and orientation are in scene/base coordinates, lengths in m.

    Success is a pose criterion, not an electrical connection or latch state.
    Depth is signed along the desired tool +z axis; over-insertion also fails.
    """
    rotation = quat_to_matrix(config["quaternion_wxyz"])
    displacement = np.asarray(position) - np.asarray(config["entrance_position"])
    local = rotation.T @ displacement
    relative = rotation.T @ quat_to_matrix(quaternion)
    angle = float(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1)))
    # A circular peg has no yaw constraint, but its insertion axis must align.
    if config.get("axially_symmetric", False):
        angle = float(np.arccos(np.clip(relative[2, 2], -1, 1)))
    lateral = float(np.linalg.norm(local[:2]))
    depth = float(local[2])
    return dict(depth_m=depth, lateral_error_m=lateral, rotation_error_rad=angle,
                success=bool(abs(depth - config["depth"]) <= config["depth_tolerance"]
                             and lateral <= config["lateral_tolerance"]
                             and angle <= config["rotation_tolerance"]))
