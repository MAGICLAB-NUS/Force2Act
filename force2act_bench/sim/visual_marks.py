"""Surface-attached marker ink. Reads contacts, writes only rasterizer nodes.

The NumPy ink model is independent of Genesis. The small renderer adapter is
for the project's pinned Genesis 0.3.7 rasterizer, including non-debug cameras.
"""

from dataclasses import dataclass
from pathlib import Path
import secrets

import numpy as np
import yaml

from .geometry import quat_to_matrix, rigid_transform


@dataclass(frozen=True)
class VisualMarksConfig:
    target: str
    geometry: Path
    sponge_link: str = "tool_link"
    sponge_collision_index: int = 2
    seed: int | None = None
    mark_count: tuple = (3, 4)
    height_range_m: tuple = (.095, .122)
    front_half_angle_deg: float = 12.
    stroke_length_m: tuple = (.016, .024)
    stroke_width_m: tuple = (.002, .003)
    erase_distance_m: float = .0008

    def __post_init__(self):
        object.__setattr__(self, "geometry", Path(self.geometry).resolve())
        if not self.target or not self.sponge_link:
            raise ValueError("visual_marks target and sponge_link must be nonempty")
        if not Path(self.geometry).is_file():
            raise FileNotFoundError(f"visual_marks geometry not found: {self.geometry}")
        for name in ("sponge_collision_index", "seed"):
            value = getattr(self, name)
            if value is None and name == "seed":
                continue
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError(f"visual_marks.{name} must be a nonnegative integer")
        for name in ("mark_count", "height_range_m", "stroke_length_m", "stroke_width_m"):
            value = tuple(getattr(self, name))
            if len(value) != 2 or not np.isfinite(value).all() or not 0 < value[0] <= value[1]:
                raise ValueError(f"visual_marks.{name} must be a positive ordered pair")
            object.__setattr__(self, name, value)
        if any(isinstance(x, bool) or int(x) != x for x in self.mark_count) or self.mark_count[1] > 32:
            raise ValueError("visual_marks.mark_count must contain integers from 1 to 32")
        object.__setattr__(self, "mark_count", tuple(map(int, self.mark_count)))
        if not np.isfinite(self.front_half_angle_deg) or not 0 < self.front_half_angle_deg < 90:
            raise ValueError("visual_marks.front_half_angle_deg must be between 0 and 90")
        if not np.isfinite(self.erase_distance_m) or not 0 < self.erase_distance_m <= .005:
            raise ValueError("visual_marks.erase_distance_m must be in (0, 0.005]")


def marks_config(values, directory):
    if values is None:
        return None
    values = dict(values)
    values["geometry"] = (Path(directory) / values["geometry"]).resolve()
    return VisualMarksConfig(**values)


class SurfaceInk:
    """Short wavy ribbons on the polygonal surface of a vase of revolution."""

    def __init__(self, config, profile, radial_segments, facing_angle, seed):
        self.config = config
        self.profile = np.asarray(profile, dtype=float)
        if (self.profile.ndim != 2 or self.profile.shape[1] != 2
                or len(self.profile) < 2 or not np.isfinite(self.profile).all()
                or np.any(np.diff(self.profile[:, 0]) <= 0) or np.any(self.profile[:, 1] <= 0)):
            raise ValueError("visual_marks requires an increasing z/radius vase profile")
        if not isinstance(radial_segments, int) or radial_segments < 3:
            raise ValueError("visual_marks requires at least three radial segments")
        self.radial_segments = radial_segments
        if not self.profile[0, 0] < config.height_range_m[0] <= config.height_range_m[1] < self.profile[-1, 0]:
            raise ValueError("visual_marks height range must lie within the vase profile")
        self.seed = int(seed)
        self.facing_angle = float(facing_angle)
        rng = np.random.default_rng(self.seed)  # Never consumes the physics RNG.
        self.mark_count = int(rng.integers(config.mark_count[0], config.mark_count[1] + 1))
        reference_radius = np.interp(np.mean(config.height_range_m), self.profile[:, 0], self.profile[:, 1])
        half_span = np.deg2rad(config.front_half_angle_deg) * reference_radius
        heights = np.linspace(*config.height_range_m, self.mark_count)
        quads, centres = [], []
        for height in heights:
            length = rng.uniform(*config.stroke_length_m)
            width = rng.uniform(*config.stroke_width_m)
            angle = rng.uniform(-.4, .4)
            count = max(12, int(np.ceil(length / .0006)))
            t = np.linspace(-length/2, length/2, count + 1)
            phase = rng.uniform(0., 2*np.pi)
            u = rng.uniform(-half_span*.2, half_span*.2) + t*np.cos(angle)
            z = height + rng.uniform(-.001, .001) + t*np.sin(angle) + .0015*np.sin(2*np.pi*t/length + phase)
            u = np.clip(u, -half_span + width, half_span - width)
            z = np.clip(z, self.profile[0, 0] + width, self.profile[-1, 0] - width)
            path = np.column_stack((u, z))
            tangent = np.gradient(path, axis=0)
            tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-12)
            side = np.column_stack((-tangent[:, 1], tangent[:, 0])) * width/2
            edges = np.stack((path - side, path + side), axis=1)
            for i in range(count):
                uv = np.array([edges[i, 0], edges[i+1, 0], edges[i+1, 1], edges[i, 1]])
                quads.append(self.surface(uv[:, 0]/reference_radius + facing_angle, uv[:, 1], offset=.00015))
                mid = (path[i] + path[i+1])/2
                centres.append(self.surface(mid[0]/reference_radius + facing_angle, mid[1]))
        self.quads = np.asarray(quads)
        self.points = np.asarray(centres)
        self.remaining = np.ones(len(self.points), dtype=bool)

    def surface(self, theta, z, offset=0.):
        theta, z = np.broadcast_arrays(theta, z)
        r = np.interp(z, self.profile[:, 0], self.profile[:, 1])
        # Match the actual 64-sided OBJ, rather than floating above a circle.
        step = 2*np.pi / self.radial_segments
        mid = (np.floor(theta/step) + .5) * step
        radius = r * np.cos(step/2) / np.cos(theta-mid)
        points = np.stack((radius*np.cos(theta), radius*np.sin(theta), z), axis=-1)
        index = np.clip(np.searchsorted(self.profile[:, 0], z, side="right") - 1, 0, len(self.profile)-2)
        slope = np.diff(self.profile[:, 1])[index] / np.diff(self.profile[:, 0])[index]
        normal = np.stack((np.cos(mid), np.sin(mid), -slope*np.cos(step/2)), axis=-1)
        normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
        return points + offset*normal

    def erase(self, vase_from_sponge, half_size, *, contact):
        if not contact:
            return False
        pose = np.asarray(vase_from_sponge)
        local = (self.points - pose[:3, 3]) @ pose[:3, :3]
        half = np.asarray(half_size)
        covered = ((np.abs(local[:, 0]) <= half[0]) & (np.abs(local[:, 1]) <= half[1])
                   & (np.abs(local[:, 2] - half[2]) <= self.config.erase_distance_m))
        removed = covered & self.remaining
        self.remaining[removed] = False
        return bool(removed.any())

    def mesh_arrays(self):
        vertices = self.quads[self.remaining].reshape(-1, 3)
        base = 4*np.arange(np.count_nonzero(self.remaining))[:, None]
        faces = (base[:, None, :] + np.array([[0, 1, 2], [0, 2, 3]])).reshape(-1, 3)
        return vertices, faces

    def state(self):
        return dict(seed=self.seed, mark_count=self.mark_count, segments_total=len(self.remaining),
                    segments_remaining=int(self.remaining.sum()), erased_fraction=float(1-self.remaining.mean()))


class InkRenderer:
    """A non-marker rasterizer mesh, absent from every physics entity/geom list."""

    def __init__(self, scene):
        self.scene = scene
        self.context = scene.visualizer.context
        self.mesh = None

    def replace(self, ink, pose):
        import trimesh
        from genesis.ext import pyrender
        with self.scene.visualizer.viewer_lock:
            self.clear()
            vertices, faces = ink.mesh_arrays()
            if not len(faces):
                return
            material = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=(.015, .018, .025, 1.), metallicFactor=0., roughnessFactor=.8,
                alphaMode="OPAQUE", doubleSided=True)
            self.mesh = pyrender.Mesh.from_trimesh(
                trimesh.Trimesh(vertices=vertices, faces=faces, process=False), material=material,
                name=f"wiping_ink_{id(self)}", poses=np.asarray(pose)[None], is_marker=False)
            self.context.add_external_node(self.mesh)

    def move(self, pose):
        if self.mesh is None:
            return
        # Genesis 0.3.7 uses instanced model buffers for external rasterizer nodes.
        # Its debug updater rejects arbitrary mesh names; keep this adapter local.
        with self.scene.visualizer.viewer_lock:
            poses = np.asarray(pose, dtype=np.float32)[None]
            self.mesh._bounds = None
            self.mesh.primitives[0].poses = poses
            node = self.context.external_nodes[self.mesh.name]
            buffer_id = self.context._scene.get_buffer_id(node, "model")
            if buffer_id >= 0:  # GPU buffers are allocated lazily at first render.
                self.context._external_node_buffer[buffer_id] = poses.transpose((0, 2, 1))

    def clear(self):
        if self.mesh is not None:
            with self.scene.visualizer.viewer_lock:
                buffer_id = self.mesh.primitives[0].get_buffer_id("model")
                self.context._external_node_buffer.pop(buffer_id, None)
                self.context.clear_external_node(self.mesh)
                self.mesh = None


class WipingMarks:
    def __init__(self, sim, config):
        self.sim, self.config = sim, config
        self.target = sim.entities[config.target]
        self.link = sim.robot.get_link(config.sponge_link)
        if config.sponge_collision_index >= len(self.link.geoms):
            raise ValueError("visual_marks sponge_collision_index is outside the tool's collision geoms")
        self.sponge = self.link.geoms[config.sponge_collision_index]
        if self.sponge.type != sim.gs.GEOM_TYPE.BOX:
            raise ValueError("visual_marks requires a box sponge collision geometry")
        self.half_size = np.asarray(self.sponge.data[:3], dtype=float)/2
        geometry = yaml.safe_load(config.geometry.read_text(encoding="utf-8"))["vase"]
        self.profile, self.radial_segments = geometry["profile_z_radius"], geometry["radial_segments"]
        self.renderer = InkRenderer(sim.scene)
        self.ink = None

    @staticmethod
    def _array(value):
        return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)

    def target_pose(self):
        return rigid_transform(quat_to_matrix(self._array(self.target.get_quat())), self._array(self.target.get_pos()))

    def sponge_pose(self):
        return rigid_transform(quat_to_matrix(self._array(self.sponge.get_quat())), self._array(self.sponge.get_pos()))

    def reset(self):
        pose = self.target_pose()
        tcp, _ = self.sim.get_tcp_pose()
        direction = (np.asarray(tcp) - pose[:3, 3]) @ pose[:3, :3]
        angle = np.arctan2(direction[1], direction[0])
        seed = secrets.randbits(63) if self.config.seed is None else self.config.seed
        self.ink = SurfaceInk(self.config, self.profile, self.radial_segments, angle, seed)
        self.renderer.replace(self.ink, pose)

    def update(self):
        contacts = self.sim.robot.get_contacts(with_entity=self.target)
        a = self._array(contacts["geom_a"]).reshape(-1) == self.sponge.idx
        b = self._array(contacts["geom_b"]).reshape(-1) == self.sponge.idx
        force = np.where(a[:, None], self._array(contacts["force_a"]).reshape(-1, 3),
                         self._array(contacts["force_b"]).reshape(-1, 3))
        active = (a | b) & (np.linalg.norm(force, axis=1) > 1e-4)
        pose = self.target_pose()
        changed = self.ink.erase(np.linalg.inv(pose) @ self.sponge_pose(), self.half_size,
                                 contact=bool(active.any()))
        if changed:
            self.renderer.replace(self.ink, pose)
        else:
            self.renderer.move(pose)

    def sync_pose(self):
        if self.ink is not None:
            self.renderer.move(self.target_pose())

    def state(self):
        return dict(target=self.config.target, **self.ink.state())
