"""Controller settings in SI units; independent of Genesis and Flexiv RDK."""

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Mapping

import numpy as np
import yaml


CONTROL_MODES = ("position", "joint_impedance", "cartesian_impedance",
                 "cartesian_motion_force", "joint_torque", "floating")


@dataclass(frozen=True)
class ControllerConfig:
    """RDK-style gains. Defaults are simulation estimates, not factory values.

    RDK damping parameters are ratios, not the position PD velocity gains.
    Optional nominal gains/limits come from the particular robot's info().
    """

    mode: str = "position"
    position_kp: tuple = (800, 800, 600, 500, 300, 240, 160)
    position_kv: tuple = (55, 55, 42, 36, 24, 20, 14)
    gripper_kp: float = 400.0
    gripper_kv: float = 20.0
    ik_position_tolerance_m: float = 2e-4
    ik_rotation_tolerance_rad: float = 2e-3
    joint_stiffness: tuple = (800, 800, 600, 500, 300, 240, 160)
    joint_damping_ratio: tuple = (0.7,) * 7
    cartesian_stiffness: tuple = (2000, 2000, 2000, 100, 100, 100)
    cartesian_damping_ratio: tuple = (0.7,) * 6
    nullspace_stiffness: tuple = (20,) * 7
    nullspace_damping_ratio: tuple = (0.7,) * 7
    nullspace_posture: tuple | None = None  # None records posture at reset.
    force_control_axes: tuple = (False,) * 6
    force_control_frame: str = "base"  # base or current TCP; origin is always TCP.
    force_frame_rotation_wxyz: tuple = (1, 0, 0, 0)
    force_axis_max_linear_velocity: tuple = (1, 1, 1)
    passive_force_control: bool = True
    force_feedback_gain: tuple = (0.5,) * 6  # Active force P gain; estimated.
    max_command_wrench: tuple = (100, 100, 100, 20, 20, 20)
    gravity_compensation: bool = True
    torque_rate_limit: tuple | None = None  # N m/s; None disables slew limiting.
    command_delay_s: float = 0.0  # Integer multiple of the physics period.
    inertia_regularization: float = 1e-6
    joint_stiffness_nominal: tuple | None = None
    cartesian_stiffness_nominal: tuple | None = None
    joint_effort_limits: tuple | None = None
    joint_velocity_limits: tuple | None = None
    joint_position_lower: tuple | None = None
    joint_position_upper: tuple | None = None
    joint_armature: tuple | None = None  # Reflected motor inertia, kg m^2.
    joint_viscous_damping: tuple | None = None  # Passive joint friction, N m s/rad.
    joint_friction_loss: tuple | None = None  # Coulomb friction, N m.
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.mode not in CONTROL_MODES:
            raise ValueError(f"controller.mode must be one of {CONTROL_MODES}")
        if self.force_control_frame not in ("base", "tcp"):
            raise ValueError("force_control_frame must be base or tcp")
        defaults = {item.name: item.default for item in fields(self)}
        for name, size in (("position_kp", 7), ("position_kv", 7),
                           ("joint_stiffness", 7), ("joint_damping_ratio", 7),
                           ("cartesian_stiffness", 6), ("cartesian_damping_ratio", 6),
                           ("nullspace_stiffness", 7), ("nullspace_damping_ratio", 7),
                           ("nullspace_posture", 7), ("force_feedback_gain", 6),
                           ("max_command_wrench", 6), ("torque_rate_limit", 7),
                           ("force_axis_max_linear_velocity", 3),
                           ("joint_stiffness_nominal", 7), ("cartesian_stiffness_nominal", 6),
                           ("joint_effort_limits", 7), ("joint_velocity_limits", 7),
                           ("joint_position_lower", 7), ("joint_position_upper", 7),
                           ("joint_armature", 7), ("joint_viscous_damping", 7), ("joint_friction_loss", 7)):
            value = getattr(self, name)
            if value is None:
                if defaults[name] is not None:
                    raise ValueError(f"controller.{name} must not be null")
                continue
            array = np.asarray(value, dtype=float)
            if array.shape != (size,) or not np.isfinite(array).all():
                raise ValueError(f"controller.{name} must contain {size} finite values")
            if "damping_ratio" in name and np.any((array < 0.3) | (array > 0.8)):
                raise ValueError(f"{name} must be in RDK range [0.3, 0.8]")
            if not any(word in name for word in ("posture", "position")) and np.any(array < 0):
                raise ValueError(f"{name} must be nonnegative")
            if name in ("max_command_wrench", "torque_rate_limit", "joint_effort_limits",
                        "joint_velocity_limits", "position_kp", "position_kv") and np.any(array <= 0):
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, tuple(array.tolist()))
        for name in ("gripper_kp", "gripper_kv", "ik_position_tolerance_m", "ik_rotation_tolerance_rad"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"controller.{name} must be positive and finite")
        frame_quaternion = np.asarray(self.force_frame_rotation_wxyz, dtype=float)
        if frame_quaternion.shape != (4,) or not np.isfinite(frame_quaternion).all() or np.linalg.norm(frame_quaternion) < 1e-12:
            raise ValueError("force_frame_rotation_wxyz must be a finite nonzero quaternion")
        object.__setattr__(self, "force_frame_rotation_wxyz", tuple((frame_quaternion / np.linalg.norm(frame_quaternion)).tolist()))
        if np.any((np.asarray(self.force_axis_max_linear_velocity) < .005) | (np.asarray(self.force_axis_max_linear_velocity) > 2)):
            raise ValueError("force_axis_max_linear_velocity must be in [0.005, 2.0] m/s")
        if len(self.force_control_axes) != 6 or any(type(x) is not bool for x in self.force_control_axes):
            raise ValueError("force_control_axes must contain six booleans")
        object.__setattr__(self, "force_control_axes", tuple(self.force_control_axes))
        if self.mode != "cartesian_motion_force" and any(self.force_control_axes):
            raise ValueError("force_control_axes requires cartesian_motion_force mode")
        if self.mode == "position" and self.torque_rate_limit is not None:
            raise ValueError("torque_rate_limit requires a torque-based controller")
        for name in ("gravity_compensation", "passive_force_control"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if not np.isfinite(self.command_delay_s) or self.command_delay_s < 0:
            raise ValueError("command_delay_s must be finite and nonnegative")
        delay_ticks = self.command_delay_s / 0.001
        if not np.isclose(delay_ticks, round(delay_ticks), atol=1e-9, rtol=0):
            raise ValueError("command_delay_s must be an integer multiple of 0.001 s")
        if not np.isfinite(self.inertia_regularization) or self.inertia_regularization <= 0:
            raise ValueError("inertia_regularization must be positive and finite")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("controller.metadata must be a mapping")
        for space in ("joint", "cartesian"):
            nominal = getattr(self, f"{space}_stiffness_nominal")
            if nominal is not None and np.any(np.asarray(getattr(self, f"{space}_stiffness")) > nominal):
                raise ValueError(f"{space}_stiffness exceeds the imported RDK nominal stiffness")
        if (self.joint_position_lower is None) != (self.joint_position_upper is None):
            raise ValueError("joint position lower and upper limits must be supplied together")
        if self.joint_position_lower is not None and np.any(
                np.asarray(self.joint_position_lower) >= self.joint_position_upper):
            raise ValueError("joint position lower limits must be below upper limits")


class _ControllerLoader(yaml.SafeLoader):
    """Reject duplicate keys, including duplicates hidden by YAML merges."""


def _unique_mapping(loader, node):
    loader.flatten_mapping(node)
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if not isinstance(key, str):
            raise ValueError("controller YAML keys must be strings")
        if key in result:
            raise ValueError(f"duplicate controller YAML key: {key}")
        result[key] = loader.construct_object(value_node)
    return result


_ControllerLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def load_controller(path: str | Path | None = None) -> ControllerConfig:
    """Load one YAML file relative to cwd, or the fixed defaults for None.

    Files must explicitly declare mode. Every other omitted field uses the
    ControllerConfig default, independent of the robot or previous controller.
    """
    if path is None:
        return ControllerConfig()
    if not isinstance(path, (str, Path)):
        raise TypeError("controller must be a YAML file path or None; mappings and ControllerConfig inputs are unsupported")
    path = Path(path).expanduser().resolve()
    try:
        value = yaml.load(path.read_text(encoding="utf-8"), Loader=_ControllerLoader)
        if not isinstance(value, Mapping):
            raise ValueError("controller config must be a mapping")
        unknown = set(value) - {item.name for item in fields(ControllerConfig)}
        if unknown:
            raise ValueError(f"unknown controller settings: {sorted(unknown)}")
        if "mode" not in value:
            raise ValueError("controller YAML must explicitly declare mode")
        return ControllerConfig(**value)
    except (ValueError, TypeError, yaml.YAMLError) as error:
        raise ValueError(f"{path}: {error}") from error


def export_robot_info(info, path, *, mode="joint_impedance", group=None):
    """Export an existing RDK RobotInfo, without connecting/enabling/moving a robot.

    RDK 1.x uses vectors; 2.x can use maps indexed by JointGroup. In that case
    pass the single-arm group explicitly. Load the output as controller YAML.
    """
    def vector(name):
        value = getattr(info, name)
        if isinstance(value, Mapping):
            if group is None:
                raise ValueError("RDK group-mapped RobotInfo requires an explicit group")
            value = value[group]
        return tuple(float(x) for x in value)

    config = ControllerConfig(
        mode=mode, joint_stiffness=vector("K_q_nom"),
        cartesian_stiffness=vector("K_x_nom"),
        joint_stiffness_nominal=vector("K_q_nom"),
        cartesian_stiffness_nominal=vector("K_x_nom"),
        joint_effort_limits=vector("tau_max"), joint_velocity_limits=vector("dq_max"),
        joint_position_lower=vector("q_min"), joint_position_upper=vector("q_max"),
        metadata={"source": "Flexiv RDK RobotInfo; damping ratio uses API default 0.7",
                  **{key: str(getattr(info, key, ""))
                     for key in ("serial_num", "model_name", "software_ver")}},
    )
    # safe_dump emits plain sequences, readable by any YAML implementation.
    values = asdict(config)
    values = {key: list(value) if isinstance(value, tuple) else value for key, value in values.items()}
    Path(path).write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    return config
