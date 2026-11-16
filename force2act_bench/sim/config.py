"""Two-level YAML configuration: scene geometry references one robot YAML."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple
import math

import numpy as np
import yaml

from .visual_marks import VisualMarksConfig, marks_config

PACKAGE_DIR = Path(__file__).resolve().parent


def _resource_directories(name: str) -> Tuple[Path, ...]:
    return (PACKAGE_DIR / name,)

def _finite_tuple(
    value: Sequence[Any], length: int, name: str
) -> Tuple[float, ...]:
    result = tuple(float(item) for item in value)
    if len(result) != length or not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain {length} finite values")
    return result


@dataclass(frozen=True)
class BoxConfig:
    """One axis-aligned or Euler-rotated rigid box in the Genesis scene."""

    name: str
    center: Tuple[float, float, float]
    size: Tuple[float, float, float]
    euler_deg: Tuple[float, float, float]
    fixed: bool
    friction: float
    color: Tuple[float, float, float, float]
    contact_feedback: bool = True
    density: float = 200.0
    mass: Optional[float] = None
    urdf: Optional[Path] = None
    collision: bool = True  # false: drawn only (e.g. a flat mark nothing can bump into)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("box name must not be empty")
        if any(value <= 0.0 for value in self.size):
            raise ValueError(f"box {self.name!r} size values must be positive")
        if not math.isfinite(self.friction) or self.friction < 0.0:
            raise ValueError(
                f"box {self.name!r} friction must be finite and nonnegative"
            )
        if not math.isfinite(self.density) or self.density <= 0.0:
            raise ValueError(f"box {self.name!r} density must be positive")
        if self.mass is not None and (not math.isfinite(self.mass) or self.mass <= 0.0):
            raise ValueError(f"box {self.name!r} mass must be positive and finite")
        if self.urdf is not None and not self.urdf.is_file():
            raise FileNotFoundError(f"box {self.name!r} URDF not found: {self.urdf}")
        if any(value < 0.0 or value > 1.0 for value in self.color):
            raise ValueError(f"box {self.name!r} color must lie in [0, 1]")

    @property
    def top_z(self) -> float:
        if any(abs(value) > 1e-12 for value in self.euler_deg[:2]):
            raise ValueError("top_z is defined only for boxes without roll/pitch")
        return self.center[2] + self.size[2] / 2.0


@dataclass(frozen=True)
class InitialStateConfig:
    """Explicit initialization, with an optional move to a target TCP pose."""

    source: str = "explicit"
    gripper_state: float = 0.0
    joint_positions: Optional[Tuple[float, ...]] = None
    tcp_position: Optional[Tuple[float, float, float]] = None
    tcp_quaternion_wxyz: Optional[Tuple[float, float, float, float]] = None
    move_duration_s: float = 0.0
    settle_steps: int = 300
    position_tolerance_m: float = 0.005
    rotation_tolerance_rad: float = 0.02

    def __post_init__(self) -> None:
        if self.source != "explicit":
            raise ValueError("standalone scenes require explicit initial_state")
        if self.source == "explicit":
            if self.joint_positions is None or len(self.joint_positions) != 7:
                raise ValueError(
                    "explicit initial_state.joint_positions must contain seven values"
                )
        if self.joint_positions is not None and (
            len(self.joint_positions) != 7
            or not all(math.isfinite(value) for value in self.joint_positions)
        ):
            raise ValueError("initial_state.joint_positions must contain seven finite values")
        if (self.tcp_position is None) != (self.tcp_quaternion_wxyz is None):
            raise ValueError(
                "initial_state TCP position and quaternion must be provided together"
            )
        if self.tcp_position is not None:
            _finite_tuple(self.tcp_position, 3, "initial_state.tcp_position")
            _finite_tuple(self.tcp_quaternion_wxyz, 4, "initial_state.tcp_quaternion_wxyz")
            if not all(math.isfinite(value) for value in self.tcp_position):
                raise ValueError("initial_state.tcp_position must be finite")
            quaternion_norm = math.sqrt(
                sum(value * value for value in self.tcp_quaternion_wxyz or ())
            )
            if quaternion_norm < 1e-8:
                raise ValueError("initial_state.tcp_quaternion_wxyz must be nonzero")
        if not math.isfinite(self.move_duration_s) or self.move_duration_s < 0.0:
            raise ValueError("initial_state.move_duration_s must be nonnegative")
        if not math.isfinite(self.gripper_state) or self.gripper_state < 0.0:
            raise ValueError("initial_state.gripper_state must be finite and nonnegative")
        if self.settle_steps < 0:
            raise ValueError("initial_state.settle_steps must be nonnegative")
        tolerances = (self.position_tolerance_m, self.rotation_tolerance_rad)
        if not all(math.isfinite(value) and value > 0.0 for value in tolerances):
            raise ValueError("initial-state tolerances must be positive")


@dataclass(frozen=True)
class RobotConfig:
    """Task-selected robot asset and its flange-to-commanded-TCP transform."""

    urdf: Path
    flange_to_tcp_position: Tuple[float, float, float]
    flange_to_tcp_rotation: Tuple[Tuple[float, float, float], ...]
    tool_friction: float = 0.15
    joint_names: Tuple[str, ...] = tuple(f"joint{i}" for i in range(1, 8))
    flange_link: str = "flange"
    tool_links: Tuple[str, ...] = ("tool_link",)
    joint_effort_limits: Tuple[float, ...] = (123, 123, 64, 64, 39, 39, 39)
    gripper_joint_names: Tuple[str, ...] = ()
    gripper_open_position: float = 0.04
    joint_kp: Tuple[float, ...] = (800, 800, 600, 500, 300, 240, 160)
    joint_kv: Tuple[float, ...] = (55, 55, 42, 36, 24, 20, 14)

    def __post_init__(self) -> None:
        if not self.urdf.is_file():
            raise FileNotFoundError(f"robot URDF not found: {self.urdf}")
        _finite_tuple(self.flange_to_tcp_position, 3, "robot.flange_to_tcp_position")
        if not math.isfinite(self.tool_friction) or self.tool_friction < 0:
            raise ValueError("robot.tool_friction must be finite and nonnegative")
        if len(self.joint_names) != 7 or len(set(self.joint_names)) != 7:
            raise ValueError("robot.joint_names must contain seven unique arm joints")
        if not self.flange_link or not self.tool_links:
            raise ValueError("robot flange and tool links must be nonempty")
        if len(self.joint_effort_limits) != 7 or min(self.joint_effort_limits) <= 0:
            raise ValueError("robot.joint_effort_limits must contain seven positive values")
        for name in ("joint_kp", "joint_kv", "joint_effort_limits"):
            _finite_tuple(getattr(self, name), 7, f"robot.{name}")
            if min(getattr(self, name)) <= 0:
                raise ValueError(f"robot.{name} must contain positive values")
        if not math.isfinite(self.gripper_open_position) or self.gripper_open_position <= 0:
            raise ValueError("robot.gripper_open_position must be positive and finite")
        rotation = np.asarray(self.flange_to_tcp_rotation, dtype=float)
        if rotation.shape != (3, 3):
            raise ValueError("robot.flange_to_tcp_rotation must be 3x3")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
            raise ValueError("robot.flange_to_tcp_rotation must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
            raise ValueError("robot.flange_to_tcp_rotation must be proper")
        if not math.isfinite(self.tool_friction) or self.tool_friction < 0.0:
            raise ValueError("robot.tool_friction must be finite and nonnegative")


@dataclass(frozen=True)
class WristCameraConfig:
    """Scene-specific transform from commanded TCP to wrist-camera coordinates.

    ``camera_from_tcp`` uses OpenCV camera axes (x right, y down, z forward).
    Keeping this transform in the scene config lets datasets collected with
    different physical camera mounts share the same Genesis environment code.
    """

    camera_from_tcp_position: Tuple[float, float, float]
    camera_from_tcp_rotation: Tuple[Tuple[float, float, float], ...]

    def __post_init__(self) -> None:
        rotation = np.asarray(self.camera_from_tcp_rotation, dtype=float)
        if rotation.shape != (3, 3):
            raise ValueError("wrist_camera.camera_from_tcp_rotation must be 3x3")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
            raise ValueError(
                "wrist_camera.camera_from_tcp_rotation must be orthonormal"
            )
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
            raise ValueError("wrist_camera.camera_from_tcp_rotation must be proper")

    @property
    def camera_from_tcp(self) -> np.ndarray:
        transform = np.eye(4)
        transform[:3, :3] = np.asarray(
            self.camera_from_tcp_rotation, dtype=float
        )
        transform[:3, 3] = np.asarray(self.camera_from_tcp_position, dtype=float)
        return transform


@dataclass(frozen=True)
class ViewerConfig:
    """Initial interactive viewer framing, also used by the video camera."""

    camera_pos: Tuple[float, float, float] = (1.25, -1.35, 1.05)
    camera_lookat: Tuple[float, float, float] = (0.35, 0.0, 0.42)
    camera_up: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    camera_fov: float = 42.0

    def __post_init__(self) -> None:
        for name in ("camera_pos", "camera_lookat", "camera_up"):
            value = getattr(self, name)
            if len(value) != 3 or not all(math.isfinite(item) for item in value):
                raise ValueError(f"viewer.{name} must contain three finite values")
        direction = np.asarray(self.camera_lookat) - np.asarray(self.camera_pos)
        up = np.asarray(self.camera_up)
        if np.linalg.norm(direction) < 1e-8:
            raise ValueError("viewer.camera_pos and camera_lookat must differ")
        if np.linalg.norm(np.cross(direction, up)) < 1e-8:
            raise ValueError("viewer.camera_up must not be parallel to the view direction")
        if not math.isfinite(self.camera_fov) or not 0.0 < self.camera_fov < 180.0:
            raise ValueError("viewer.camera_fov must lie between 0 and 180 degrees")


@dataclass(frozen=True)
class GenesisSceneConfig:
    """Task geometry, cameras and self-contained initialization for one scene."""

    task_name: Optional[str]
    robot: RobotConfig
    robot_base_position: Tuple[float, float, float]
    wrist_camera: Optional[WristCameraConfig]
    viewer: ViewerConfig
    table: BoxConfig
    objects: Tuple[BoxConfig, ...]
    initial_state: InitialStateConfig
    sensor_cameras: Mapping[str, Any]
    config_path: Path
    insertion: Optional[Mapping[str, Any]] = None
    visual_marks: Optional[VisualMarksConfig] = None
    # Sponge normal force (N) needed to erase a mark, and how long it must be held (wiping_marks.py).
    visual_marks_min_force: float = 0.0
    visual_marks_min_force_hold_s: float = 0.0
    # Wider erasing: marks within erase_margin_m beyond the sponge edges are erased too, and
    # erase_depth_m (if set) replaces erase_distance_m as the allowed distance from the sponge face.
    visual_marks_erase_margin_m: float = 0.0
    visual_marks_erase_depth_m: Optional[float] = None
    # Mark centres uniform within +-this angle of the object's front (default: one column,
    # +-20% of front_half_angle_deg). Must be below 18 deg.
    visual_marks_centre_spread_deg: Optional[float] = None
    # One vertical ink line instead of random marks
    # {length_m, width_m, centre_height_m, lateral_jitter_m}.
    visual_marks_line: Optional[Mapping[str, float]] = None

    def __post_init__(self) -> None:
        names = [self.table.name, *(item.name for item in self.objects)]
        if len(set(names)) != len(names):
            raise ValueError("scene box names must be unique")
        if self.visual_marks is not None:
            if self.visual_marks.target not in {body.name for body in self.objects}:
                raise ValueError("visual_marks.target must name a scene object")
            if self.visual_marks.sponge_link not in self.robot.tool_links:
                raise ValueError("visual_marks.sponge_link must be one of robot.tool_links")
        for name in ("visual_marks_min_force", "visual_marks_min_force_hold_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name.replace('visual_marks_', 'visual_marks.')} must be finite and nonnegative")
        if not math.isfinite(self.visual_marks_erase_margin_m) or not 0.0 <= self.visual_marks_erase_margin_m <= 0.05:
            raise ValueError("visual_marks.erase_margin_m must be in [0, 0.05]")
        depth = self.visual_marks_erase_depth_m
        if depth is not None and (not math.isfinite(depth) or not 0.0 < depth <= 0.03):
            raise ValueError("visual_marks.erase_depth_m must be in (0, 0.03]")
        line = self.visual_marks_line
        if line is not None:
            allowed = {"length_m", "width_m", "centre_height_m", "lateral_jitter_m"}
            if set(line) - allowed or not {"length_m", "width_m", "centre_height_m"} <= set(line):
                raise ValueError("visual_marks.line needs length_m, width_m, centre_height_m (optional lateral_jitter_m)")
            if not all(math.isfinite(float(v)) for v in line.values()):
                raise ValueError("visual_marks.line values must be finite")
            if not (0.0 < line["length_m"] <= 0.15 and 0.0 < line["width_m"] <= 0.03
                    and 0.0 <= line.get("lateral_jitter_m", 0.0) <= 0.03):
                raise ValueError("visual_marks.line: length_m (0, 0.15], width_m (0, 0.03], lateral_jitter_m [0, 0.03]")
        spread = self.visual_marks_centre_spread_deg
        if spread is not None and (not math.isfinite(spread) or not 0.0 < spread < 18.0):
            raise ValueError("visual_marks.mark_centre_spread_deg must be in (0, 18)")
        if self.insertion is not None:
            target = next((body for body in self.objects
                           if body.name == self.insertion["target"]), None)
            if target is None or not target.fixed or not target.contact_feedback:
                raise ValueError("insertion target must be fixed and have contact_feedback")
            _finite_tuple(self.insertion["entrance_position"], 3, "insertion.entrance_position")
            q = _finite_tuple(self.insertion["quaternion_wxyz"], 4, "insertion.quaternion_wxyz")
            if np.linalg.norm(q) < 1e-8:
                raise ValueError("insertion quaternion must be nonzero")
            for key in ("depth", "depth_tolerance", "lateral_tolerance", "rotation_tolerance"):
                value = float(self.insertion[key])
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"insertion.{key} must be positive and finite")


def _parse_box(
    values: Mapping[str, Any],
    default_name: Optional[str] = None,
    config_directory: Optional[Path] = None,
) -> BoxConfig:
    name = str(values.get("name", default_name or ""))
    urdf = values.get("urdf")
    if urdf is not None:
        urdf = Path(str(urdf)).expanduser()
        if not urdf.is_absolute():
            if config_directory is None:
                raise ValueError(f"relative URDF for {name!r} needs a config directory")
            urdf = config_directory / urdf
        urdf = urdf.resolve()
    return BoxConfig(
        name=name,
        center=_finite_tuple(values["center"], 3, f"{name}.center"),
        size=_finite_tuple(values["size"], 3, f"{name}.size"),
        euler_deg=_finite_tuple(
            values.get("euler_deg", (0.0, 0.0, 0.0)), 3, f"{name}.euler_deg"
        ),
        fixed=bool(values.get("fixed", True)),
        friction=float(values.get("friction", 0.15)),
        color=_finite_tuple(
            values.get("color", (0.7, 0.7, 0.7, 1.0)), 4, f"{name}.color"
        ),
        contact_feedback=bool(values.get("contact_feedback", True)),
        density=float(values.get("density", 200.0)),
        mass=None if values.get("mass") is None else float(values["mass"]),
        urdf=urdf,
        collision=bool(values.get("collision", True)),
    )


def available_scenes() -> Tuple[str, ...]:
    return tuple(sorted({path.stem for directory in _resource_directories("scenes")
                         for path in directory.glob("*.yaml")}))


def _yaml(path: Path) -> dict:
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict) or values.get("schema_version") != 1:
        raise ValueError(f"{path}: expected a mapping with schema_version: 1")
    return values


def load_scene(scene: str | Path = "flip", robot: str | Path | None = None) -> GenesisSceneConfig:
    """Resolve paths relative to their owning YAML, independent of working directory.

    Built-in names or custom YAML paths are accepted. A robot override changes
    the asset and TCP definition; the caller must supply a compatible initial pose.
    """
    path = Path(scene).expanduser()
    if not path.is_file() and str(scene) in available_scenes():
        path = next(directory / f"{scene}.yaml" for directory in _resource_directories("scenes")
                    if (directory / f"{scene}.yaml").is_file())
    path = path.resolve()
    values = _yaml(path)
    reference = robot if robot is not None else values["robot"]
    if not isinstance(reference, (str, Path)):
        raise ValueError("scene.robot must reference a robot YAML file")
    robot_path = Path(reference).expanduser()
    named_robot = next((directory / f"{robot}.yaml" for directory in _resource_directories("robots")
                        if robot is not None and (directory / f"{robot}.yaml").is_file()), None)
    if named_robot is not None:
        robot_path = named_robot
    elif not robot_path.is_absolute():
        robot_path = (Path.cwd() if robot is not None else path.parent) / robot_path
    robot_values = _yaml(robot_path)
    robot_values.pop("schema_version")
    robot_values["urdf"] = (robot_path.parent / robot_values["urdf"]).resolve()
    for key in ("joint_names", "tool_links", "gripper_joint_names", "flange_to_tcp_position",
                "joint_effort_limits", "joint_kp", "joint_kv"):
        if key in robot_values:
            robot_values[key] = tuple(robot_values[key])
    camera = values.get("wrist_camera")
    marks = None if values.get("visual_marks") is None else dict(values["visual_marks"])
    marks_min_force = 0.0 if marks is None else float(marks.pop("min_force", 0.0))
    marks_hold_s = 0.0 if marks is None else float(marks.pop("min_force_hold_s", 0.0))
    marks_margin = 0.0 if marks is None else float(marks.pop("erase_margin_m", 0.0))
    marks_depth = None if marks is None else marks.pop("erase_depth_m", None)
    marks_spread = None if marks is None else marks.pop("mark_centre_spread_deg", None)
    marks_line = None if marks is None else marks.pop("line", None)
    return GenesisSceneConfig(
        task_name=values.get("task_name"),
        robot=RobotConfig(**robot_values),
        robot_base_position=_finite_tuple(
            values.get("robot_base_position", (0.0, 0.0, 0.0)), 3,
            "scene.robot_base_position",
        ),
        wrist_camera=None if camera is None else WristCameraConfig(**camera),
        viewer=ViewerConfig(**values.get("viewer", {})),
        table=_parse_box(values["table"], "table", path.parent),
        objects=tuple(_parse_box(item, config_directory=path.parent) for item in values["objects"]),
        initial_state=InitialStateConfig(**values["initial_state"]),
        sensor_cameras=values.get("sensor_cameras", {}),
        config_path=path,
        insertion=values.get("insertion"),
        visual_marks=marks_config(marks, path.parent),
        visual_marks_min_force=marks_min_force,
        visual_marks_min_force_hold_s=marks_hold_s,
        visual_marks_erase_margin_m=marks_margin,
        visual_marks_erase_depth_m=None if marks_depth is None else float(marks_depth),
        visual_marks_centre_spread_deg=None if marks_spread is None else float(marks_spread),
        visual_marks_line=None if marks_line is None else {k: float(v) for k, v in marks_line.items()},
    )
