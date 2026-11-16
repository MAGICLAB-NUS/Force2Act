"""Genesis scene, robot control, cameras and the TCP contact wrench."""

import ctypes.util
import os
import sys
from dataclasses import replace

import numpy as np

from .camera import Camera, OPENGL_FROM_OPENCV, _genesis_pose
from .config import load_scene
from .control_config import load_controller
from .controller import ArmController, tcp_jacobian
from .geometry import matrix_to_quat, quat_to_matrix, quaternion_slerp, rigid_transform
from .action import Action
from .insertion import insertion_state


def as_numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


class Simulator:
    physics_dt = 0.001
    control_dt = 0.01
    _backend = None

    def __init__(self, scene="flip", *, robot=None, viewer=True, gpu=True,
                 cameras=False, image_size=(640, 360), seed=0, controller=None, wrench_samples=1):
        self.config = load_scene(scene, robot)
        self.controller_config = load_controller(controller)
        self.controller = None
        # Tool wrench samples per step() (1 = only the last physics step, as before).
        # Genesis contact forces alternate between physics steps while a tool slides,
        # so a single sample often reads zero mid-contact; averaging a few spread over
        # the step gives a steady force for haptics, force limits, ink, and data.
        self.wrench_samples = int(wrench_samples)
        if self.wrench_samples < 1:
            raise ValueError("wrench_samples must be a positive integer")
        self._step_wrench = None
        self._controller_active = False
        self.viewer_enabled = viewer
        self.closed = False
        self.time = 0.0
        self._steps = 0
        self.image_size = tuple(image_size)
        if len(self.image_size) != 2 or any(int(v) != v or v <= 0 for v in self.image_size):
            raise ValueError("image_size must contain two positive integers (width, height)")
        self.camera_configs = {
            name: Camera(values, self.image_size, self.config.wrist_camera)
            for name, values in self.config.sensor_cameras.items()
        } if cameras else {}
        # Imports stay local so configuration, tests and policies work without Genesis.
        # Genesis creates a GL context even for physics-only scenes. On Linux,
        # CPU runs can use Mesa when no interactive viewer is requested.
        if sys.platform == "linux" and not viewer and not gpu and ctypes.util.find_library("OSMesa"):
            os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
        import genesis as gs

        self.gs = gs
        backend = gs.gpu if gpu else gs.cpu
        if not gs._initialized:
            gs.init(backend=backend, seed=seed, precision="32", logging_level="warning")
            Simulator._backend = backend
        elif Simulator._backend != backend:
            raise RuntimeError("use one Genesis backend per process; restart to change CPU/GPU")
        view = self.config.viewer
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.physics_dt, substeps=1),
            rigid_options=gs.options.RigidOptions(
                enable_collision=True, enable_self_collision=False, enable_joint_limit=True,
                constraint_timeconst=0.01, iterations=50, noslip_iterations=2, use_gjk_collision=True),
            viewer_options=gs.options.ViewerOptions(
                res=(1280, 720), camera_pos=view.camera_pos, camera_lookat=view.camera_lookat,
                camera_up=view.camera_up, camera_fov=view.camera_fov, max_FPS=60),
            vis_options=gs.options.VisOptions(
                show_world_frame=False, show_link_frame=False, show_cameras=False,
                background_color=(0.68, 0.72, 0.76), ambient_light=(0.32, 0.32, 0.32)),
            show_viewer=viewer)
        self.visual_marks = None
        try:
            self._build()
            if self.config.visual_marks is not None:
                # Wiping ink, erased where the sponge presses on the target.
                from .wiping_marks import ForceGatedWipingMarks
                self.visual_marks = ForceGatedWipingMarks(
                    self, self.config.visual_marks, self.config.visual_marks_min_force,
                    self.config.visual_marks_min_force_hold_s, 10 * self.physics_dt,
                    erase_margin=self.config.visual_marks_erase_margin_m,
                    erase_depth=self.config.visual_marks_erase_depth_m,
                    centre_spread_deg=self.config.visual_marks_centre_spread_deg,
                    line=self.config.visual_marks_line)
            self._initial_state = self.scene.get_state()
            # Scene placements, so evaluation can shift fixed targets per trial.
            self._base_object_positions = {name: as_numpy(entity.get_pos()).copy()
                                           for name, entity in self.entities.items()}
            self._base_object_quaternions = {name: as_numpy(entity.get_quat()).copy()
                                             for name, entity in self.entities.items()}
            self._target_offset = np.zeros(3)
            self._tilted_cameras = set()
            self.reset()
        except BaseException:
            self.close()
            raise

    def _build(self):
        gs = self.gs
        self.entities = {}
        self.contact_entities = []
        for body in (self.config.table, *self.config.objects):
            if body.urdf is None:
                morph = gs.morphs.Box(pos=body.center, size=body.size,
                                      euler=body.euler_deg, fixed=body.fixed, collision=body.collision)
                surface = gs.surfaces.Default(color=body.color)
            else:
                morph = gs.morphs.URDF(file=str(body.urdf), pos=body.center,
                                       euler=body.euler_deg, fixed=body.fixed, convexify=False)
                surface = None
            entity = self.scene.add_entity(morph, material=gs.materials.Rigid(rho=body.density),
                                           surface=surface)
            self.entities[body.name] = entity
            if body.contact_feedback and body.collision:
                self.contact_entities.append(entity)
        robot = self.config.robot
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(file=str(robot.urdf), pos=self.config.robot_base_position,
                           fixed=True, merge_fixed_links=True,
                           links_to_keep=list(dict.fromkeys((robot.flange_link, *robot.tool_links))),
                           convexify=False),
            material=gs.materials.Rigid(
                gravity_compensation=float(self.controller_config.gravity_compensation)))
        # Camera builds a pinhole camera or, for opencv_fisheye, a
        # six-face cube rig that rectify() resamples into the fisheye image.
        self.cameras = {
            name: camera.create_genesis_camera(self.scene)
            for name, camera in self.camera_configs.items()
        }
        self.scene.build()
        # URDF inertial values take precedence over material density in Genesis.
        # Apply an explicit scene mass after build; Genesis scales the link's
        # inertia tensor by the same ratio, preserving consistent dynamics.
        for body in (self.config.table, *self.config.objects):
            if body.mass is not None:
                self.entities[body.name].set_mass(body.mass)
        self.flange = self.robot.get_link(robot.flange_link)
        self.tool_links = [self.robot.get_link(name) for name in robot.tool_links]
        joints = [self.robot.get_joint(name) for name in robot.joint_names]
        self.arm_dofs = [joint.dofs_idx_local[0] for joint in joints]
        self.arm_qs = [joint.qs_idx_local[0] for joint in joints]
        self.gripper_dofs = [self.robot.get_joint(name).dofs_idx_local[0]
                             for name in robot.gripper_joint_names]
        self.robot.set_dofs_kp(np.asarray(robot.joint_kp), self.arm_dofs)
        self.robot.set_dofs_kv(np.asarray(robot.joint_kv), self.arm_dofs)
        effort = np.asarray(robot.joint_effort_limits)
        self.arm_effort_limits = effort.copy()
        self.robot.set_dofs_force_range(-effort, effort, self.arm_dofs)
        if self.gripper_dofs:
            self.robot.set_dofs_kp(np.full(len(self.gripper_dofs), 400.), self.gripper_dofs)
            self.robot.set_dofs_kv(np.full(len(self.gripper_dofs), 20.), self.gripper_dofs)
        for body in (self.config.table, *self.config.objects):
            if body.collision:
                self.entities[body.name].set_friction(body.friction)
        for link in self.tool_links:
            link.set_friction(robot.tool_friction)
        self.flange_from_tcp = rigid_transform(np.asarray(robot.flange_to_tcp_rotation),
                                               np.asarray(robot.flange_to_tcp_position))
        for name, camera in self.camera_configs.items():
            if camera.mount == "tcp":
                self.cameras[name].attach(
                    self.flange, self.flange_from_tcp @ camera.transform @ OPENGL_FROM_OPENCV)
                self.cameras[name].move_to_attach()

    def _require_open(self):
        if self.closed:
            raise RuntimeError("simulator is closed")

    def reset(self):
        """Restore all bodies and the scene's arm/hand initialization. Return lowdim state."""
        self._require_open()
        self.scene.reset(self._initial_state)
        self._target_offset = np.zeros(3)  # the reset puts every body back in place
        for name in self._tilted_cameras:
            self.cameras[name].set_pose(**self.camera_configs[name].genesis_pose())
        self._tilted_cameras = set()
        # Initialization deliberately uses the existing collision-aware IK
        # position path; activate torque-level Cartesian control only after the
        # requested start pose has settled.
        self._controller_active = False
        initial = self.config.initial_state
        joints = np.asarray(initial.joint_positions, dtype=float)
        self.robot.set_dofs_position(joints, self.arm_dofs)
        self.robot.control_dofs_position(joints, self.arm_dofs)
        self._set_gripper(initial.gripper_state, initialize=True)
        position, quaternion = self.get_tcp_pose()
        target_position, target_quaternion = position, quaternion
        if initial.tcp_position is not None:
            target_position = np.asarray(initial.tcp_position)
            target_quaternion = np.asarray(initial.tcp_quaternion_wxyz)
            updates = max(1, round(initial.move_duration_s / self.control_dt))
            for index in range(1, updates + 1):
                fraction = index / updates
                self.set_tcp_pose(position + fraction * (target_position - position),
                                  quaternion_slerp(quaternion, target_quaternion, fraction))
                self._advance(round(self.control_dt / self.physics_dt))
        self._advance(initial.settle_steps)
        actual_position, actual_quaternion = self.get_tcp_pose()
        position_error = np.linalg.norm(actual_position - target_position)
        relative = quat_to_matrix(target_quaternion).T @ quat_to_matrix(actual_quaternion)
        rotation_error = np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1))
        if position_error > initial.position_tolerance_m or rotation_error > initial.rotation_tolerance_rad:
            raise RuntimeError(f"initial TCP error: {position_error:.6f} m, {rotation_error:.6f} rad")
        self._steps = 0
        self.time = 0.0
        if self.controller is None:
            self.controller = ArmController(self, self.controller_config)
        else:
            self.controller.reset()
        self._controller_active = True
        if self.visual_marks is not None:
            self.visual_marks.reset()
        return self.observe()

    def set_tcp_pose(self, position, quaternion_wxyz):
        """Submit to the Cartesian controller, or use IK during initialization."""
        self._require_open()
        if self._controller_active:
            self.controller.submit(Action(
                tcp_position=position, tcp_quaternion_wxyz=quaternion_wxyz
            ))
            return
        self.robot.control_dofs_position(
            self._solve_tcp_joints(position, quaternion_wxyz), self.arm_dofs
        )

    def _solve_tcp_joints(self, position, quaternion_wxyz):
        action = Action(tcp_position=position, tcp_quaternion_wxyz=quaternion_wxyz)
        base_from_tcp = rigid_transform(quat_to_matrix(action.tcp_quaternion_wxyz), action.tcp_position)
        base_from_flange = base_from_tcp @ np.linalg.inv(self.flange_from_tcp)
        qpos, error = self.robot.inverse_kinematics(
            link=self.flange, pos=base_from_flange[:3, 3], quat=matrix_to_quat(base_from_flange[:3, :3]),
            init_qpos=self.robot.get_qpos(), respect_joint_limit=True,
            max_samples=1, max_solver_iters=200, pos_tol=2e-4, rot_tol=2e-3, return_error=True)
        error = as_numpy(error).reshape(-1)
        if np.linalg.norm(error[:3]) > 2e-3 or np.linalg.norm(error[3:]) > 2e-2:
            raise RuntimeError(f"IK did not converge: {error}")
        return as_numpy(qpos).reshape(-1)[self.arm_qs].copy()

    def _set_gripper(self, closure, initialize=False):
        self._gripper = float(closure)
        if self.gripper_dofs:
            positions = np.full(len(self.gripper_dofs),
                                self.config.robot.gripper_open_position * (1 - np.clip(closure, 0, 1)))
            if initialize:
                self.robot.set_dofs_position(positions, self.gripper_dofs)
            self.robot.control_dofs_position(positions, self.gripper_dofs)

    def get_tcp_pose(self):
        self._require_open()
        base_from_flange = rigid_transform(quat_to_matrix(as_numpy(self.flange.get_quat())),
                                           as_numpy(self.flange.get_pos()))
        transform = base_from_flange @ self.flange_from_tcp
        return transform[:3, 3].copy(), matrix_to_quat(transform[:3, :3])

    def get_tcp_wrench(self):
        """[Fx,Fy,Fz,Tx,Ty,Tz], N / N·m, environment-on-tool, TCP frame/origin.

        This is a gravity-compensated contact measurement: it sums contacts on
        robot.tool_links with bodies marked contact_feedback. It does not include
        tool inertia, sensor noise, or contacts on other arm links.
        """
        self._require_open()
        position, quaternion = self.get_tcp_pose()
        force, torque = np.zeros(3), np.zeros(3)
        indices = [link.idx for link in self.tool_links]
        for entity in self.contact_entities:
            contacts = self.robot.get_contacts(with_entity=entity)
            is_a = np.isin(as_numpy(contacts["link_a"]).reshape(-1), indices)
            is_b = np.isin(as_numpy(contacts["link_b"]).reshape(-1), indices)
            keep = is_a | is_b
            points = as_numpy(contacts["position"]).reshape(-1, 3)[keep]
            forces = np.where(is_a[keep, None],
                              as_numpy(contacts["force_a"]).reshape(-1, 3)[keep],
                              as_numpy(contacts["force_b"]).reshape(-1, 3)[keep])
            force += forces.sum(axis=0)
            torque += np.cross(points - position, forces).sum(axis=0)
        rotation = quat_to_matrix(quaternion)
        return np.concatenate((rotation.T @ force, rotation.T @ torque))

    def _tool_wrench_all_contacts(self):
        """get_tcp_wrench() with one contact query instead of one per contact entity."""
        position, quaternion = self.get_tcp_pose()
        if not hasattr(self, "_contact_link_idx"):
            self._contact_link_idx = np.asarray(
                [link.idx for entity in self.contact_entities for link in entity.links])
            self._tool_link_idx = np.asarray([link.idx for link in self.tool_links])
        contacts = self.robot.get_contacts()
        link_a = as_numpy(contacts["link_a"]).reshape(-1)
        link_b = as_numpy(contacts["link_b"]).reshape(-1)
        is_a = np.isin(link_a, self._tool_link_idx) & np.isin(link_b, self._contact_link_idx)
        is_b = np.isin(link_b, self._tool_link_idx) & np.isin(link_a, self._contact_link_idx)
        keep = is_a | is_b
        force, torque = np.zeros(3), np.zeros(3)
        if keep.any():
            points = as_numpy(contacts["position"]).reshape(-1, 3)[keep]
            forces = np.where(is_a[keep, None], as_numpy(contacts["force_a"]).reshape(-1, 3)[keep],
                              as_numpy(contacts["force_b"]).reshape(-1, 3)[keep])
            force = forces.sum(axis=0)
            torque = np.cross(points - position, forces).sum(axis=0)
        rotation = quat_to_matrix(quaternion)
        return np.concatenate((rotation.T @ force, rotation.T @ torque))

    def _advance(self, count):
        sample_at = set()
        if self.wrench_samples > 1:
            # Evenly spread over this advance, always including its last physics step.
            sample_at = {count - 1 - round(i * (count - 1) / (self.wrench_samples - 1))
                         for i in range(min(self.wrench_samples, count))}
        samples = []
        for index in range(count):
            if self._controller_active:
                self.controller.update()
            self._steps += 1
            self.scene.step(update_visualizer=self.viewer_enabled and self._steps % 10 == 0)
            if index in sample_at:
                samples.append(self._tool_wrench_all_contacts())
                self._step_wrench = np.mean(samples, axis=0)
            if self.visual_marks is not None and self._controller_active and self._steps % 10 == 0:
                self.visual_marks.update()
        self.time += count * self.physics_dt
        for name, camera in self.camera_configs.items():
            if camera.mount == "tcp":
                self.cameras[name].move_to_attach()

    def step(self, action=None, *, dt=None, images=False, depth=False):
        """Apply a command, advance dt (default 10 ms), then return an observation.

        dt must be a positive integer multiple of the 1 ms physics step.
        For DROID 15 Hz, alternate 66/67 ms periods in the caller.
        """
        self._require_open()
        duration = self.control_dt if dt is None else float(dt)
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("dt must be positive and finite")
        steps = round(duration / self.physics_dt)
        if steps < 1 or not np.isclose(steps * self.physics_dt, duration, atol=1e-10, rtol=0):
            raise ValueError("dt must be an integer multiple of physics_dt")
        if action is not None:
            if not isinstance(action, Action):
                raise TypeError("step expects an Action")
            # Force-controlled axes and their frame (world, w-x-y-z; default: world axes) update the live
            # controller configuration and stay in effect until the next action that sets them.
            axes = getattr(action, "force_control_axes", None)
            frame_rotation = getattr(action, "force_frame_rotation_wxyz", None)
            values = vars(action).copy()
            values.pop("force_control_axes", None)
            values.pop("force_frame_rotation_wxyz", None)
            action = Action(**values)
            if axes is not None:
                axes = tuple(bool(value) for value in np.asarray(axes).reshape(6))
                rotation = (1.0, 0.0, 0.0, 0.0) if frame_rotation is None else tuple(
                    float(value) for value in np.asarray(frame_rotation, dtype=float).reshape(4))
                self.controller_config = replace(
                    self.controller_config, force_control_axes=axes,
                    force_frame_rotation_wxyz=rotation,
                )
                self.controller.config = self.controller_config
            self.controller.submit(action)
            if action.gripper is not None:
                self._set_gripper(action.gripper)
        self._advance(steps)
        return self.observe(images=images, depth=depth)

    def capture(self, name, *, depth=False):
        """Return RGB uint8 HWC; with depth=True return (RGB, depth_m)."""
        self._require_open()
        if name not in self.cameras:
            raise KeyError(f"camera {name!r} unavailable; initialize with cameras=True")
        camera = self.camera_configs[name]
        # Camera.render() already acquires Genesis's viewer lock internally.
        # Taking that non-reentrant lock here too deadlocks whenever the viewer
        # and policy camera are both enabled.
        if camera.mount == "tcp":
            self.cameras[name].move_to_attach()
        if self.visual_marks is not None:
            self.visual_marks.sync_pose()
        rendered = self.cameras[name].render(rgb=True, depth=depth)
        rgb = rendered[0] if isinstance(rendered, (tuple, list)) else rendered
        rgb = camera.rectify(as_numpy(rgb))
        if depth:
            return rgb, camera.rectify(as_numpy(rendered[1]), nearest=True)
        return rgb

    def observe(self, *, images=False, depth=False):
        self._require_open()
        if depth and not images:
            raise ValueError("depth=True requires images=True")
        if images and not self.cameras:
            raise RuntimeError("initialize with cameras=True to request images")
        position, quaternion = self.get_tcp_pose()
        gripper = self._gripper
        if self.gripper_dofs:
            positions = as_numpy(self.robot.get_dofs_position(self.gripper_dofs))
            gripper = float(np.clip(1 - positions.mean() / self.config.robot.gripper_open_position, 0, 1))
        observation = dict(time=self.time, tcp_position=position, tcp_quaternion_wxyz=quaternion,
                           joint_positions=as_numpy(self.robot.get_dofs_position(self.arm_dofs)).copy(),
                           gripper=gripper, tcp_wrench=(
                               self.get_tcp_wrench() if self._step_wrench is None else self._step_wrench.copy()))
        if self.controller is not None:
            status = self.controller.status()
            rotation = quat_to_matrix(quaternion)
            measured = observation["tcp_wrench"]
            measured_base = np.r_[rotation @ measured[:3], rotation @ measured[3:]]
            flange_rotation = quat_to_matrix(as_numpy(self.flange.get_quat()))
            offset = flange_rotation @ np.asarray(
                self.config.robot.flange_to_tcp_position
            )
            jacobian = tcp_jacobian(
                as_numpy(self.robot.get_jacobian(self.flange))[:, self.arm_dofs],
                offset,
            )
            commanded_torque = np.asarray(
                status["commanded_joint_torques"], dtype=float
            )
            # Least-squares diagnostic only: nullspace torque can appear in this
            # reconstruction, but it reliably reveals clipping/sign/scale errors.
            wrench_estimate = np.linalg.pinv(jacobian.T) @ commanded_torque
            status.update(
                force_control_axes=np.asarray(
                    self.controller.config.force_control_axes, dtype=bool
                ).copy(),
                target_wrench=np.asarray(
                    self.controller.wrench_target, dtype=float
                ).copy(),
                measured_wrench_base=measured_base,
                applied_wrench_estimate_base=wrench_estimate,
            )
            observation["controller"] = status
        if self.config.insertion is not None:
            insertion = dict(self.config.insertion)
            insertion["entrance_position"] = (np.asarray(insertion["entrance_position"], float)
                                              + self._target_offset)
            observation["insertion"] = insertion_state(insertion, position, quaternion)
        if self.visual_marks is not None:
            observation["visual_marks"] = self.visual_marks.state()
        if images:
            observation["rgb"], observation["depth"] = {}, {}
            observation["camera_intrinsics"], observation["base_from_camera"] = {}, {}
            base_from_tcp = rigid_transform(quat_to_matrix(quaternion), position)
            for name, camera in self.camera_configs.items():
                image = self.capture(name, depth=depth)
                observation["rgb"][name] = image[0] if depth else image
                if depth:
                    observation["depth"][name] = image[1]
                observation["camera_intrinsics"][name] = camera.intrinsics.copy()
                observation["base_from_camera"][name] = camera.pose(base_from_tcp)
        return observation

    def get_object_pose(self, name):
        self._require_open()
        entity = self.entities[name]
        return as_numpy(entity.get_pos()).copy(), as_numpy(entity.get_quat()).copy()

    def get_contact_report(self):
        """Per-pair contact points and forces from the last physics step, for diagnostics.

        Keys are "tool|<body>" (force on the tool) for every scene body, and
        "<object>|<body>" (force on the object) for every movable object against
        every other body. Each value holds world-frame "points" and "forces" lists,
        one row per Genesis contact; Genesis gives no contact normals.
        """
        self._require_open()
        tool_indices = [link.idx for link in self.tool_links]

        def pair(entity, other, side_indices=None):
            contacts = entity.get_contacts(with_entity=other)
            link_a = as_numpy(contacts["link_a"]).reshape(-1)
            link_b = as_numpy(contacts["link_b"]).reshape(-1)
            if side_indices is None:
                side_indices = [link.idx for link in entity.links]
            is_a = np.isin(link_a, side_indices)
            keep = is_a | np.isin(link_b, side_indices)
            forces = np.where(is_a[keep, None],
                              as_numpy(contacts["force_a"]).reshape(-1, 3)[keep],
                              as_numpy(contacts["force_b"]).reshape(-1, 3)[keep])
            points = as_numpy(contacts["position"]).reshape(-1, 3)[keep]
            return {"points": points.tolist(), "forces": forces.tolist()}

        report = {}
        for name, entity in self.entities.items():
            report[f"tool|{name}"] = pair(self.robot, entity, tool_indices)
        movable = [body.name for body in self.config.objects if not body.fixed]
        for name in movable:
            for other, entity in self.entities.items():
                if other != name:
                    report[f"{name}|{other}"] = pair(self.entities[name], entity)
        return report

    def set_target_offset(self, names, offset):
        """Shift scene bodies (e.g. a fixed socket and its mount) by a world offset [m]
        from their scene placement, and the insertion success target with them when
        the target is among them. Call after reset(); reset() clears the offset."""
        self._require_open()
        offset = np.asarray(offset, dtype=float).reshape(3)
        for name in names:
            if name not in self.entities:
                raise KeyError(f"unknown scene object {name!r}; available: {sorted(self.entities)}")
            self.entities[name].set_pos(self._base_object_positions[name] + offset)
        target = (self.config.insertion or {}).get("target")
        self._target_offset = offset.copy() if target in names else np.zeros(3)

    def set_incline(self, degrees, names, pivot, axis=(0.0, 1.0, 0.0), cameras=()):
        """Rotate scene bodies (e.g. the table, everything on it) rigidly by `degrees` about
        the world `axis` through the world point `pivot`, from their scene placement, and the
        listed base-mounted `cameras` with them (their image is then the same at every
        incline). Call after reset(); reset() puts them back. A negative angle about +y
        raises +x."""
        self._require_open()
        axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
        half = np.radians(float(degrees)) / 2.0
        rotation_quaternion = np.r_[np.cos(half), np.sin(half) * axis]
        rotation = quat_to_matrix(rotation_quaternion)
        pivot = np.asarray(pivot, dtype=float).reshape(3)
        for name in names:
            if name not in self.entities:
                raise KeyError(f"unknown scene object {name!r}; available: {sorted(self.entities)}")
            position = pivot + rotation @ (self._base_object_positions[name] - pivot)
            quaternion = matrix_to_quat(rotation @ quat_to_matrix(self._base_object_quaternions[name]))
            self.entities[name].set_pos(position)
            self.entities[name].set_quat(quaternion)
        tilt = rigid_transform(rotation, pivot - rotation @ pivot)
        for name in cameras:
            camera = self.camera_configs.get(name) if hasattr(self, "camera_configs") else None
            if camera is None:
                continue  # images off: nothing to move
            if camera.mount != "base" or camera.projection != "pinhole":
                raise ValueError(f"camera {name!r}: only base-mounted pinhole cameras can tilt")
            self.cameras[name].set_pose(**_genesis_pose(tilt @ camera.transform))
            self._tilted_cameras.add(name)

    def object_contacts(self, name, *, entity=None):
        """Robot contacts on one object from the last physics step: "tool_points" (world) and
        "tool_forces" (on the tool) for the tool links, and "other_links", the number of
        contacts with any other robot link."""
        self._require_open()
        entity = self.entities[name] if entity is None else entity
        contacts = self.robot.get_contacts(with_entity=entity)
        link_a = as_numpy(contacts["link_a"]).reshape(-1)
        link_b = as_numpy(contacts["link_b"]).reshape(-1)
        robot_links = [link.idx for link in self.robot.links]
        tool = [link.idx for link in self.tool_links]
        robot_a = np.isin(link_a, robot_links)
        robot_side = np.where(robot_a, link_a, link_b)
        is_tool = np.isin(robot_side, tool)
        forces = np.where(robot_a[:, None], as_numpy(contacts["force_a"]).reshape(-1, 3),
                          as_numpy(contacts["force_b"]).reshape(-1, 3))
        points = as_numpy(contacts["position"]).reshape(-1, 3)
        return {"tool_points": points[is_tool], "tool_forces": forces[is_tool],
                "other_links": int((~is_tool).sum())}

    def set_object_mass(self, name, mass):
        """Set one movable object's total mass and consistently scaled inertia."""
        self._require_open()
        if name not in self.entities:
            raise KeyError(f"unknown scene object {name!r}; available: {sorted(self.entities)}")
        value = float(mass)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("object mass must be positive and finite")
        self.entities[name].set_mass(value)

    def close(self):
        if self.closed:
            return
        if getattr(self, "visual_marks", None) is not None:
            self.visual_marks.renderer.clear()
        viewer = self.scene.viewer if self.viewer_enabled else None
        if viewer is not None and viewer.is_alive():
            viewer.stop()
        self.scene.destroy()
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
