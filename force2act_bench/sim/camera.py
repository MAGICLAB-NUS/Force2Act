"""Scene-defined pinhole/fisheye cameras with dataset intrinsics.

Fisheye RGB is sampled from six perspective views, preserving rays beyond the
front hemisphere. Depth for this projection is distance along the camera ray.
"""

import numpy as np

from .geometry import rigid_transform

OPENGL_FROM_OPENCV = np.diag([1., -1., -1., 1.])

# Each matrix maps a cube face's OpenCV optical coordinates into rig coordinates.
_CUBE_ROTATIONS = np.array([
    np.eye(3),
    [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
    [[0, 0, -1], [0, 1, 0], [1, 0, 0]],
    [[1, 0, 0], [0, 0, 1], [0, -1, 0]],
    [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
    [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
], dtype=float)


def _genesis_pose(transform):
    gl = transform @ OPENGL_FROM_OPENCV
    position = gl[:3, 3]
    return dict(pos=tuple(position), lookat=tuple(position - gl[:3, 2]),
                up=tuple(gl[:3, 1]))


class _CubeCamera:
    """Expose Genesis camera attachment/render operations for one optical rig."""

    def __init__(self, cameras):
        self.cameras = cameras

    def attach(self, link, transform):
        for camera, rotation in zip(self.cameras, _CUBE_ROTATIONS):
            face = rigid_transform(rotation, np.zeros(3))
            camera.attach(link, transform @ OPENGL_FROM_OPENCV @ face @ OPENGL_FROM_OPENCV)

    def move_to_attach(self):
        for camera in self.cameras:
            camera.move_to_attach()

    def render(self, *, rgb=True, depth=False, **kwargs):
        results = [camera.render(rgb=rgb, depth=depth, **kwargs) for camera in self.cameras]
        return (np.stack([result[0] for result in results]),
                np.stack([result[1] for result in results]) if depth else None)


class Camera:
    def __init__(self, values, size, wrist_camera):
        self.size = tuple(size)
        width, height = self.size
        self.projection = values.get("projection", "pinhole")
        if self.projection not in ("pinhole", "opencv_fisheye"):
            raise ValueError("camera projection must be pinhole or opencv_fisheye")
        self.mount = values.get("mount", "base")
        if self.mount == "tcp":
            if wrist_camera is None:
                raise ValueError("TCP camera requires scene.wrist_camera")
            self.transform = np.linalg.inv(wrist_camera.camera_from_tcp)
        elif self.mount == "base":
            if "base_from_camera" in values:
                self.transform = np.asarray(values["base_from_camera"], dtype=float)
            else:
                position = np.asarray(values["position"], dtype=float)
                forward = np.asarray(values["lookat"], dtype=float) - position
                right = np.cross(forward, values.get("up", [0, 0, 1]))
                if min(np.linalg.norm(forward), np.linalg.norm(right)) < 1e-8:
                    raise ValueError("invalid camera view direction or up")
                forward /= np.linalg.norm(forward)
                right /= np.linalg.norm(right)
                self.transform = rigid_transform(
                    np.column_stack((right, np.cross(forward, right), forward)), position)
        else:
            raise ValueError("camera mount must be base or tcp")
        if self.transform.shape != (4, 4) or not np.allclose(self.transform[3], [0, 0, 0, 1]):
            raise ValueError("camera transform must be homogeneous 4x4")
        rigid_transform(self.transform[:3, :3], self.transform[:3, 3])
        if "intrinsics" in values:
            self.intrinsics = np.asarray(values["intrinsics"], dtype=float).copy()
            native_width, native_height = values["native_size"]
            self.intrinsics[0] *= width / native_width
            self.intrinsics[1] *= height / native_height
        else:
            fov = float(values["fov"])
            if not 0 < fov < 180:
                raise ValueError("camera fov must lie in (0, 180)")
            focal = height / (2 * np.tan(np.deg2rad(fov) / 2))
            self.intrinsics = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
        if (self.intrinsics.shape != (3, 3) or not np.isfinite(self.intrinsics).all()
                or min(self.intrinsics[0, 0], self.intrinsics[1, 1]) <= 0):
            raise ValueError("camera intrinsics must be finite 3x3 with positive focal lengths")
        self.render_size = self.size
        self.fov = float(np.rad2deg(2 * np.arctan(height / (2 * self.intrinsics[1, 1]))))
        # Genesis renders centred fx=fy pinholes. Map their pixels to the
        # exported dataset intrinsics without needing OpenCV as a dependency.
        yy, xx = np.indices((height, width), dtype=float)
        fx, fy = self.intrinsics[0, 0], self.intrinsics[1, 1]
        cx, cy = self.intrinsics[0, 2], self.intrinsics[1, 2]
        self.source_x = (xx - cx) * fy / fx + width / 2
        self.source_y = yy - cy + height / 2
        self.valid_pixels = np.ones((height, width), dtype=bool)
        if self.projection == "opencv_fisheye":
            self._fisheye_map(values, xx, yy)

    def _fisheye_map(self, values, xx, yy):
        if "intrinsics" not in values or "native_size" not in values:
            raise ValueError("fisheye cameras require intrinsics and native_size")
        self.distortion = np.asarray(values.get("distortion", [0, 0, 0, 0]), dtype=float)
        if self.distortion.shape != (4,) or not np.isfinite(self.distortion).all():
            raise ValueError("fisheye distortion must contain four finite coefficients")
        scale = values.get("render_scale", 2)
        if isinstance(scale, bool) or int(scale) != scale or not 1 <= scale <= 8:
            raise ValueError("fisheye render_scale must be an integer in [1, 8]")
        side = max(self.size) * int(scale)
        # Overlap cube faces to keep interpolation away from their last pixel.
        self.render_size, self.fov = (side, side), 92.0
        face_focal = side / (2 * np.tan(np.deg2rad(self.fov) / 2))
        fx, fy = self.intrinsics[0, 0], self.intrinsics[1, 1]
        cx, cy = self.intrinsics[0, 2], self.intrinsics[1, 2]
        x, y = (xx - cx) / fx, (yy - cy) / fy
        radius = np.hypot(x, y)
        theta = np.minimum(radius, 1.8)
        k1, k2, k3, k4 = self.distortion
        for _ in range(15):
            t2 = theta * theta
            distorted = theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
            derivative = 1 + t2 * (3*k1 + t2 * (5*k2 + t2 * (7*k3 + t2 * 9*k4)))
            theta = np.clip(theta - (distorted - radius) / np.maximum(derivative, .05), 0, np.pi)
        t2 = theta * theta
        recovered = theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
        derivative = 1 + t2 * (3*k1 + t2 * (5*k2 + t2 * (7*k3 + t2 * 9*k4)))
        self.valid_pixels &= (np.abs(recovered - radius) < 1e-6) & (derivative > 0)
        if "aperture" in values:
            aperture = values["aperture"]
            center = np.asarray(aperture["center"], dtype=float)
            native_width, native_height = values["native_size"]
            aperture_radius = float(aperture["radius"])
            if center.shape != (2,) or not np.isfinite(center).all() or not np.isfinite(aperture_radius) or aperture_radius <= 0:
                raise ValueError("fisheye aperture requires a finite centre and positive radius")
            self.valid_pixels &= ((xx * native_width / self.size[0] - center[0])**2
                                  + (yy * native_height / self.size[1] - center[1])**2 <= aperture_radius**2)
        sinc = np.divide(np.sin(theta), radius, out=np.ones_like(radius), where=radius > 1e-12)
        rays = np.stack((x*sinc, y*sinc, np.cos(theta)), axis=-1)
        axes = _CUBE_ROTATIONS[:, :, 2]
        self.source_face = np.argmax(rays @ axes.T, axis=-1)
        face_rays = np.einsum("...j,...jk->...k", rays, _CUBE_ROTATIONS[self.source_face])
        self.source_x = face_focal * face_rays[..., 0] / face_rays[..., 2] + side / 2
        self.source_y = face_focal * face_rays[..., 1] / face_rays[..., 2] + side / 2
        self.depth_scale = 1 / face_rays[..., 2]

    def create_genesis_camera(self, scene):
        """Create a perspective camera or a cube rig before scene.build()."""
        kwargs = dict(model="pinhole", res=self.render_size, fov=self.fov,
                      GUI=False, near=.01, far=5., debug=False)
        if self.projection == "pinhole":
            return scene.add_camera(**kwargs, **self.genesis_pose())
        return _CubeCamera([
            scene.add_camera(**kwargs, **_genesis_pose(self.pose() @ rigid_transform(rotation, np.zeros(3))))
            for rotation in _CUBE_ROTATIONS
        ])

    def pose(self, base_from_tcp=None):
        transform = self.transform.copy()
        if self.mount == "tcp" and base_from_tcp is not None:
            transform = base_from_tcp @ transform
        return transform

    def genesis_pose(self):
        return _genesis_pose(self.pose())

    def rectify(self, value, nearest=False):
        cube = self.projection == "opencv_fisheye"
        if cube and (value.ndim not in (3, 4) or value.shape[0] != 6):
            raise ValueError("fisheye rendering requires six cube faces")
        height, width = value.shape[1:3] if cube else value.shape[:2]
        if (width, height) != self.render_size:
            raise RuntimeError(
                f"Genesis rendered {(width, height)}, expected {self.render_size}. "
                "Genesis 0.3.7's OSMesa backend may return a 32x32 buffer. "
                "Use a working EGL or desktop OpenGL context for camera observations.")
        x, y = self.source_x, self.source_y
        if nearest:
            xi, yi = np.rint(x).astype(int), np.rint(y).astype(int)
            valid = self.valid_pixels & (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
            output = np.zeros((self.size[1], self.size[0], *value.shape[3:]) if cube else value.shape, dtype=value.dtype)
            output[valid] = (value[self.source_face[valid], yi[valid], xi[valid]] if cube
                             else value[yi[valid], xi[valid]])
            if cube and value.ndim == 3:
                output *= self.depth_scale
            return output
        x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
        valid = self.valid_pixels & (x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1)
        x0c, x1c = np.clip(x0, 0, width - 1), np.clip(x0 + 1, 0, width - 1)
        y0c, y1c = np.clip(y0, 0, height - 1), np.clip(y0 + 1, 0, height - 1)
        wx, wy = x - x0, y - y0
        if value.ndim == (4 if cube else 3):
            wx, wy, valid = wx[..., None], wy[..., None], valid[..., None]
        def sample(yi, xi):
            return value[self.source_face, yi, xi] if cube else value[yi, xi]
        output = (sample(y0c, x0c) * (1 - wx) * (1 - wy)
                  + sample(y0c, x1c) * wx * (1 - wy)
                  + sample(y1c, x0c) * (1 - wx) * wy
                  + sample(y1c, x1c) * wx * wy)
        return np.where(valid, output, 0).astype(value.dtype)
