"""Wiping ink, erased only under a sustained normal press of the sponge.

Scene settings under visual_marks: min_force is the sponge's normal force on the target (the summed contact
force along the sponge-face normal; friction drag excluded) and min_force_hold_s how long that force must stay
at or above min_force before the marks under the pad are erased, so an impact spike or a light brush does not
wipe. erase_margin_m widens the erased area beyond the pad edges, and erase_depth_m (default:
erase_distance_m) is how far from the pad face a mark may be and still be erased.

mark_centre_spread_deg spreads random marks sideways (centres uniform within +-mark_centre_spread_deg of the
object's front). line replaces the random marks with one continuous vertical ink line on the front, facing
the TCP at reset: {length_m, width_m, centre_height_m, lateral_jitter_m}. The erased fraction is the share
of the line wiped off.
"""

import dataclasses
import secrets

import numpy as np

from .visual_marks import SurfaceInk, WipingMarks

# SurfaceInk places each mark's centre within +-20% of front_half_angle_deg.
CENTRE_FRACTION = 0.2
# Segment length along the line (m), as in SurfaceInk strokes.
LINE_SEGMENT_M = 0.0006


class LineInk(SurfaceInk):
    """One straight vertical ink line on the vase front, built like SurfaceInk strokes."""

    def __init__(self, config, profile, radial_segments, facing_angle, seed, *,
                 length_m, width_m, centre_height_m, lateral_jitter_m=0.0):
        self.config = config
        self.profile = np.asarray(profile, dtype=float)
        self.radial_segments = int(radial_segments)
        self.seed = int(seed)
        self.facing_angle = float(facing_angle)
        self.mark_count = 1
        rng = np.random.default_rng(self.seed)
        reference_radius = np.interp(centre_height_m, self.profile[:, 0], self.profile[:, 1])
        u = rng.uniform(-lateral_jitter_m, lateral_jitter_m) if lateral_jitter_m > 0 else 0.0
        count = max(12, int(np.ceil(length_m / LINE_SEGMENT_M)))
        z = np.linspace(centre_height_m - length_m / 2, centre_height_m + length_m / 2, count + 1)
        half = width_m / 2
        quads, centres = [], []
        for i in range(count):
            # Same corner order as SurfaceInk strokes (path - side, path + side).
            uv = np.array([[u + half, z[i]], [u + half, z[i + 1]], [u - half, z[i + 1]], [u - half, z[i]]])
            quads.append(self.surface(uv[:, 0] / reference_radius + facing_angle, uv[:, 1], offset=.00015))
            centres.append(self.surface(u / reference_radius + facing_angle, (z[i] + z[i + 1]) / 2))
        self.quads = np.asarray(quads)
        self.points = np.asarray(centres)
        self.remaining = np.ones(len(self.points), dtype=bool)


class ForceGatedWipingMarks(WipingMarks):
    def __init__(self, sim, config, min_force=0.0, hold_s=0.0, update_period_s=0.01,
                 erase_margin=0.0, erase_depth=None, centre_spread_deg=None, line=None):
        super().__init__(sim, config)
        self.line = None if line is None else dict(line)
        self.centre_spread_deg = None if centre_spread_deg is None else float(centre_spread_deg)
        self.erase_margin = float(erase_margin)
        self.erase_depth = float(config.erase_distance_m if erase_depth is None else erase_depth)
        self.min_force = float(min_force)
        self.hold_s = float(hold_s)
        # update() runs every 10 physics steps (see Simulator._advance).
        self.hold_updates = max(1, int(round(self.hold_s / update_period_s))) if self.hold_s > 0 else 0
        self.sponge_force = 0.0
        self.pressed_updates = 0

    def make_ink(self, facing_angle, seed):
        """The configured ink (line, spread marks, or random marks) for this seed.

        Used by reset() and by Simulator.mirror_visual_marks(), so a mirroring
        renderer always rebuilds exactly the ink the physics process erases.
        """
        if self.line is not None:
            return LineInk(self.config, self.profile, self.radial_segments, facing_angle, seed, **self.line)
        config = self.config
        if self.centre_spread_deg is not None:
            config = dataclasses.replace(
                config, front_half_angle_deg=self.centre_spread_deg / CENTRE_FRACTION)
        return SurfaceInk(config, self.profile, self.radial_segments, facing_angle, seed)

    def reset(self):
        # WipingMarks.reset, building the ink through make_ink().
        pose = self.target_pose()
        tcp, _ = self.sim.get_tcp_pose()
        direction = (np.asarray(tcp) - pose[:3, 3]) @ pose[:3, :3]
        angle = np.arctan2(direction[1], direction[0])
        seed = secrets.randbits(63) if self.config.seed is None else self.config.seed
        self.ink = self.make_ink(angle, seed)
        self.renderer.replace(self.ink, pose)
        self.sponge_force, self.pressed_updates = 0.0, 0

    def update(self):
        contacts = self.sim.robot.get_contacts(with_entity=self.target)
        a = self._array(contacts["geom_a"]).reshape(-1) == self.sponge.idx
        b = self._array(contacts["geom_b"]).reshape(-1) == self.sponge.idx
        force = np.where(a[:, None], self._array(contacts["force_a"]).reshape(-1, 3),
                         self._array(contacts["force_b"]).reshape(-1, 3))
        touching = (a | b) & (np.linalg.norm(force, axis=1) > 1e-4)
        sponge = self.sponge_pose()
        face_normal = sponge[:3, 2]
        averaged = getattr(self.sim, "_step_wrench", None)
        if averaged is not None:
            # Simulator averages the tool wrench over the step (wrench_samples > 1); a
            # single contact sample flickers to zero while the sponge slides. The TCP z
            # axis is the sponge-face normal.
            self.sponge_force = abs(float(averaged[2]))
            pressed = self.sponge_force >= self.min_force
        else:
            self.sponge_force = (abs(float(force[touching].sum(axis=0) @ face_normal))
                                 if touching.any() else 0.0)
            pressed = touching.any() and self.sponge_force >= self.min_force
        self.pressed_updates = self.pressed_updates + 1 if pressed else 0
        pose = self.target_pose()
        changed = self._erase(np.linalg.inv(pose) @ sponge,
                              contact=self.pressed_updates >= max(1, self.hold_updates))
        if changed:
            self.renderer.replace(self.ink, pose)
        else:
            self.renderer.move(pose)

    def _erase(self, vase_from_sponge, *, contact):
        """SurfaceInk.erase with a widened footprint and depth tolerance."""
        if not contact:
            return False
        ink = self.ink
        local = (ink.points - vase_from_sponge[:3, 3]) @ vase_from_sponge[:3, :3]
        half = self.half_size
        covered = ((np.abs(local[:, 0]) <= half[0] + self.erase_margin)
                   & (np.abs(local[:, 1]) <= half[1] + self.erase_margin)
                   & (np.abs(local[:, 2] - half[2]) <= self.erase_depth))
        removed = covered & ink.remaining
        ink.remaining[removed] = False
        return bool(removed.any())

    def state(self):
        return dict(super().state(), min_force=self.min_force, min_force_hold_s=self.hold_s,
                    sponge_normal_force=self.sponge_force, erase_margin_m=self.erase_margin,
                    erase_depth_m=self.erase_depth)
