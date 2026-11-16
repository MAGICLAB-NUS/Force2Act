"""Readable torque controllers with RDK-style parameters, evaluated at 1 kHz.

These implement documented interface semantics, not Flexiv's proprietary law.
Genesis provides rigid-body dynamics and gravity compensation exactly once.
"""

from collections import deque

import numpy as np

from .geometry import matrix_to_quat, quat_to_matrix


def as_numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def rotation_error(target, current):
    """Shortest SO(3) log error expressed in base axes, including at 180 degrees."""
    quaternion = matrix_to_quat(quat_to_matrix(target) @ quat_to_matrix(current).T)
    sine = np.linalg.norm(quaternion[1:])
    if sine < 1e-12:
        return 2 * quaternion[1:]
    return quaternion[1:] * (2 * np.arctan2(sine, quaternion[0]) / sine)


def tcp_jacobian(flange_jacobian, offset_in_base):
    """Shift a base-frame flange Jacobian to the commanded TCP origin."""
    result = np.asarray(flange_jacobian, dtype=float).copy()
    result[:3] += np.cross(result[3:].T, offset_in_base).T
    return result


def damping(stiffness, ratios, inertia):
    """Axis approximation D_i = 2 zeta_i sqrt(K_i M_i), SI units."""
    return 2 * np.asarray(ratios) * np.sqrt(np.asarray(stiffness) * np.maximum(inertia, 0))


class ArmController:
    def __init__(self, simulator, config, *, activate=True):
        self.sim = simulator
        self.config = config
        self.dofs = simulator.arm_dofs
        limits = config.joint_effort_limits or simulator.config.robot.joint_effort_limits
        self.effort = np.asarray(limits, dtype=float)
        # Honor both configured and URDF actuator limits.
        self.effort = np.minimum(self.effort, simulator.arm_effort_limits)
        lower, upper = simulator.robot.get_dofs_limit(self.dofs)
        self.lower = as_numpy(lower).astype(float)
        self.upper = as_numpy(upper).astype(float)
        if config.joint_position_lower is not None:
            self.lower = np.maximum(self.lower, config.joint_position_lower)
            self.upper = np.minimum(self.upper, config.joint_position_upper)
        if np.any(self.lower >= self.upper):
            raise ValueError("imported joint limits do not overlap the robot model")
        delay = config.command_delay_s / simulator.physics_dt
        if not np.isclose(delay, round(delay), atol=1e-9, rtol=0):
            raise ValueError("command_delay_s must be an integer multiple of physics_dt")
        self.delay_steps = round(delay)
        self.reset(activate=activate)

    def reset(self, *, activate=True):
        self.pending = deque()
        self.tick = 0
        self.q_target = as_numpy(self.sim.robot.get_dofs_position(self.dofs)).copy()
        self.dq_target = np.zeros(7)
        self.position_target, self.quaternion_target = self.sim.get_tcp_pose()
        self.twist_target = np.zeros(6)
        self.acceleration_target = np.zeros(6)
        self.wrench_target = np.zeros(6)
        self.torque_target = np.zeros(7)
        self.nullspace_target = np.asarray(
            self.config.nullspace_posture if self.config.nullspace_posture is not None else self.q_target
        ).copy()
        self._check_joints(self.nullspace_target)
        self.last_torque = np.zeros(7)
        self.saturated = False
        if activate:
            self.activate()

    def activate(self):
        """Apply limits and the initial hold only after replacement validation."""
        self.sim.robot.set_dofs_force_range(-self.effort, self.effort, self.dofs)
        if self.config.mode == "position":
            self.sim.robot.control_dofs_position(self.q_target, self.dofs)
        else:
            self.sim.robot.control_dofs_force(self.last_torque, self.dofs)

    def _check_joints(self, q):
        if np.any(q < self.lower) or np.any(q > self.upper):
            raise ValueError("joint target exceeds configured/URDF position limits")

    def submit(self, action):
        """Validate now, apply after the configured simulated transport delay."""
        mode = self.config.mode
        cartesian = mode in ("cartesian_impedance", "cartesian_motion_force")
        command = {}
        has_pose = action.joint_positions is not None or action.tcp_position is not None
        if mode in ("joint_torque", "floating") and (has_pose or action.joint_velocities is not None):
            raise ValueError(f"{mode} does not accept motion targets")
        if action.joint_torques is not None and mode != "joint_torque":
            raise ValueError("joint_torques requires joint_torque mode")
        if action.tcp_wrench is not None and not cartesian:
            raise ValueError("tcp_wrench requires a Cartesian controller")
        if (action.tcp_velocity is not None or action.tcp_acceleration is not None) and not cartesian:
            raise ValueError("tcp_velocity/tcp_acceleration requires a Cartesian controller")
        if cartesian and (action.joint_positions is not None or action.joint_velocities is not None):
            raise ValueError("Cartesian controllers require TCP actions; configure nullspace_posture for arm posture")
        if mode == "position" and action.joint_velocities is not None:
            raise ValueError("joint_velocities requires joint_impedance mode")
        if has_pose:
            if cartesian:
                command.update(position_target=action.tcp_position.copy(),
                               quaternion_target=action.tcp_quaternion_wxyz.copy(),
                               twist_target=np.zeros(6) if action.tcp_velocity is None else action.tcp_velocity.copy(),
                               acceleration_target=np.zeros(6) if action.tcp_acceleration is None else action.tcp_acceleration.copy())
            else:
                q = (action.joint_positions.copy() if action.joint_positions is not None else
                     self.sim._solve_tcp_joints(action.tcp_position, action.tcp_quaternion_wxyz))
                self._check_joints(q)
                command.update(q_target=q, dq_target=np.zeros(7))
        if action.joint_velocities is not None:
            if self.config.joint_velocity_limits is not None and np.any(
                    np.abs(action.joint_velocities) > self.config.joint_velocity_limits):
                raise ValueError("joint_velocities exceeds imported RDK dq_max")
            command["dq_target"] = action.joint_velocities.copy()
        for field, target in (("tcp_velocity", "twist_target"), ("tcp_wrench", "wrench_target"),
                              ("tcp_acceleration", "acceleration_target"),
                              ("joint_torques", "torque_target")):
            value = getattr(action, field)
            if value is not None:
                command[target] = value.copy()
        if command:
            self.pending.append((self.tick + self.delay_steps, command))

    def _mass_matrix(self):
        matrix = as_numpy(self.sim.robot.get_mass_mat()).reshape(self.sim.robot.n_dofs, self.sim.robot.n_dofs)
        return matrix[np.ix_(self.dofs, self.dofs)].astype(float)

    def _cartesian_torque(self, q, dq, mass):
        config = self.config
        position, quaternion = self.sim.get_tcp_pose()
        rotation = quat_to_matrix(quaternion)
        flange_rotation = quat_to_matrix(as_numpy(self.sim.flange.get_quat()))
        offset = flange_rotation @ np.asarray(self.sim.config.robot.flange_to_tcp_position)
        jacobian = tcp_jacobian(as_numpy(self.sim.robot.get_jacobian(self.sim.flange))[:, self.dofs], offset)
        twist = jacobian @ dq
        error = np.concatenate((self.position_target - position,
                                rotation_error(self.quaternion_target, quaternion)))
        # Impedance axes are always base axes. Force axes can follow current TCP.
        inverse_mass_jt = np.linalg.solve(mass, jacobian.T)
        mobility = jacobian @ inverse_mass_jt
        task_mass = np.linalg.inv(mobility + config.inertia_regularization * np.eye(6))
        gains = np.asarray(config.cartesian_stiffness)
        velocity_gains = damping(gains, config.cartesian_damping_ratio, np.diag(task_mass))
        wrench = (gains * error + velocity_gains * (self.twist_target - twist)
                  + task_mass @ self.acceleration_target)
        if config.mode == "cartesian_motion_force":
            frame = np.eye(6)
            force_rotation = quat_to_matrix(config.force_frame_rotation_wxyz)
            if config.force_control_frame == "tcp":
                force_rotation = rotation @ force_rotation
            frame[:3, :3] = force_rotation
            frame[3:, 3:] = force_rotation
            axes = np.asarray(config.force_control_axes)
            motion_projector = frame @ np.diag(~axes) @ frame.T
            # Delete force-axis reference components BEFORE gain/inertia
            # coupling, then project the motion output into its subspace.
            wrench = motion_projector @ (
                gains * (motion_projector @ error)
                + velocity_gains * (motion_projector @ (self.twist_target - twist))
                + task_mass @ (motion_projector @ self.acceleration_target))
            local_wrench = frame.T @ wrench
            measured = self.sim.get_tcp_wrench()  # environment on tool, TCP axes
            measured_base = np.concatenate((rotation @ measured[:3], rotation @ measured[3:]))
            target = self.wrench_target
            # Official target is SENSED wrench: +Z commands a push towards -Z.
            force_output = -target.copy()
            if not config.passive_force_control:
                force_output -= np.asarray(config.force_feedback_gain) * (target - frame.T @ measured_base)
            # Remove pose springs on force axes; keep dissipative velocity damping.
            local_damping = frame.T @ np.diag(velocity_gains) @ frame
            local_twist = frame.T @ twist
            force_output -= local_damping @ (local_twist * axes)
            if not config.passive_force_control:
                # Approximate contact-loss velocity protection: remove drive that
                # would accelerate further beyond the configured velocity bound.
                limits = np.asarray(config.force_axis_max_linear_velocity)
                outward = (np.abs(local_twist[:3]) >= limits) & (force_output[:3] * local_twist[:3] > 0)
                force_output[:3][outward] = -np.diag(local_damping)[:3][outward] * local_twist[:3][outward]
            local_wrench[axes] = force_output[axes]
            wrench = frame @ local_wrench
        # This is an OUTPUT bound, not RDK SetMaxContactWrench's contact regulator.
        wrench = np.clip(wrench, -np.asarray(config.max_command_wrench), config.max_command_wrench)
        torque = jacobian.T @ wrench
        null_gains = np.asarray(config.nullspace_stiffness)
        null_damping = damping(null_gains, config.nullspace_damping_ratio, np.diag(mass))
        null_torque = null_gains * (self.nullspace_target - q) - null_damping * dq
        # Dynamically consistent torque projector; task leakage near singularities
        # is bounded by the explicit regularization rather than an unstable inverse.
        projector = np.eye(7) - jacobian.T @ task_mass @ inverse_mass_jt.T
        return torque + projector @ null_torque

    def update(self):
        while self.pending and self.pending[0][0] <= self.tick:
            _, command = self.pending.popleft()
            for name, value in command.items():
                setattr(self, name, value)
            if self.config.mode == "position":
                self.sim.robot.control_dofs_position(self.q_target, self.dofs)
        self.tick += 1
        if self.config.mode == "position":
            return
        q = as_numpy(self.sim.robot.get_dofs_position(self.dofs))
        dq = as_numpy(self.sim.robot.get_dofs_velocity(self.dofs))
        mode = self.config.mode
        if mode == "joint_torque":
            torque = self.torque_target.copy()
        elif mode == "floating":
            torque = np.zeros(7)
        else:
            mass = self._mass_matrix()
            if mode == "joint_impedance":
                gains = np.asarray(self.config.joint_stiffness)
                velocity_gains = damping(gains, self.config.joint_damping_ratio, np.diag(mass))
                torque = gains * (self.q_target - q) + velocity_gains * (self.dq_target - dq)
            else:
                torque = self._cartesian_torque(q, dq, mass)
        if not np.isfinite(torque).all():
            raise RuntimeError("controller computed a nonfinite torque")
        self.saturated = bool(np.any(np.abs(torque) > self.effort))
        torque = np.clip(torque, -self.effort, self.effort)
        if self.config.torque_rate_limit is not None:
            delta = np.asarray(self.config.torque_rate_limit) * self.sim.physics_dt
            torque = np.clip(torque, self.last_torque - delta, self.last_torque + delta)
        self.last_torque = torque.copy()
        self.sim.robot.control_dofs_force(torque, self.dofs)

    def status(self):
        torque = (as_numpy(self.sim.robot.get_dofs_control_force(self.dofs)).copy()
                  if self.config.mode == "position" else self.last_torque.copy())
        saturated = (bool(np.any(np.abs(torque) >= self.effort - 1e-6))
                     if self.config.mode == "position" else self.saturated)
        return dict(mode=self.config.mode, commanded_joint_torques=torque,
                    torque_saturated=saturated, command_delay_s=self.config.command_delay_s)
