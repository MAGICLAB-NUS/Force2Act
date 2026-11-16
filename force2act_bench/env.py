"""Force2Act-Bench environment: one task, one simulator, episodes on demand.

    env = BenchEnv("pivot_block", control="hybrid")
    obs = env.reset(env.task.conditions[0], seed=1050)
    while not env.done:
        obs = env.step(policy.act(obs))
    print(env.result())

Observations: time, tcp_position and tcp_quaternion_wxyz (world), tcp_wrench (environment on tool, TCP axes,
N and Nm), joint_positions and, when rendered, rgb {camera: uint8 H x W x 3}. Actions: force2act_bench.Action
with a TCP pose target and, under hybrid control, tcp_wrench, force_control_axes and force_frame_rotation_wxyz.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from force2act_bench.sim import Action, Simulator
from force2act_bench.sim.geometry import quat_to_matrix
from force2act_bench.tasks import FREQUENCY_HZ, IMAGE_SIZE, TASKS, Condition, Task

CONTROLLERS = Path(__file__).resolve().parent / "controllers"


@dataclass
class EpisodeResult:
    task: str
    condition: str
    trial: int
    seed: int
    success: bool
    success_time_s: float | None
    steps: int
    error: str | None = None
    max_tilt_deg: float | None = None
    max_insertion_depth_mm: float | None = None
    final_lateral_error_mm: float | None = None
    erased_fraction: float | None = None
    object_shift_mm: float | None = None
    push_mm: float | None = None
    contact_violation: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def control_dt(index: int, frequency_hz: float = FREQUENCY_HZ) -> float:
    """Duration of control step `index` on the 1 ms physics clock (66 / 67 ms at 15 Hz)."""
    previous_tick = round(index / frequency_hz / Simulator.physics_dt)
    next_tick = round((index + 1) / frequency_hz / Simulator.physics_dt)
    return (next_tick - previous_tick) * Simulator.physics_dt


def tilt_degrees(initial_quaternion: np.ndarray, quaternion: np.ndarray) -> float:
    """Tilt of the object's z axis from its initial direction."""
    cosine = float(np.clip(abs(np.dot(quat_to_matrix(initial_quaternion)[:, 2], quat_to_matrix(quaternion)[:, 2])),
                           0.0, 1.0))
    return math.degrees(math.acos(cosine))


def target_offset(seed: int, distance: float) -> np.ndarray:
    """Target displacement of `distance` m in a direction drawn from the episode seed."""
    rng = np.random.default_rng([int(seed), 1729])
    angle = rng.uniform(0.0, 2.0 * math.pi)
    return np.array([distance * math.cos(angle), distance * math.sin(angle), 0.0])


class BenchEnv:
    def __init__(self, task: str | Task, control: str = "hybrid", images: bool = True, gpu: bool = False,
                 seed: int = 0) -> None:
        self.task = TASKS[task] if isinstance(task, str) else task
        controller = Path(control) if str(control).endswith(".yaml") else CONTROLLERS / f"{control}.yaml"
        if not controller.is_file():
            raise ValueError(f"unknown control {control!r}: use position, impedance, hybrid or a controller YAML")
        self.images = images
        self.sim = Simulator(self.task.scene, viewer=False, gpu=gpu, cameras=images, image_size=IMAGE_SIZE,
                             seed=seed, controller=str(controller), wrench_samples=self.task.wrench_samples)
        self.done = True

    @property
    def camera(self) -> str:
        return self.task.camera

    def close(self) -> None:
        self.sim.close()

    # ---------------------------------------------------------------------------------------------- episode
    def reset(self, condition: Condition | str, seed: int, trial: int = 1) -> dict[str, Any]:
        """Start an episode of `condition` with `seed`; returns the first observation (rendered)."""
        task, sim = self.task, self.sim
        if isinstance(condition, str):
            condition = next(c for c in task.conditions if c.name == condition)
        self.condition, self.seed, self.trial = condition, int(seed), int(trial)
        kind = task.success["type"]
        np.random.seed(self.seed)
        if condition.mass_kg is not None:
            sim.set_object_mass(task.object, condition.mass_kg)
        if kind == "wiping" and sim.visual_marks is not None:
            sim.visual_marks.config = replace(sim.visual_marks.config, seed=self.seed)  # the ink line of this seed
        sim.reset()
        if condition.mass_kg is not None:
            sim.set_object_mass(task.object, condition.mass_kg)
        if task.target_objects:
            sim.set_target_offset(list(task.target_objects), target_offset(self.seed, condition.offset_m or 0.0))
        if task.incline is not None:
            incline = task.incline
            sim.set_incline(float(condition.incline_deg), list(incline["bodies"]), incline["pivot"], incline["axis"],
                            cameras=incline["cameras"])
        success = task.success
        self._state: dict[str, Any] = {"max_tilt": 0.0, "final_tilt": 0.0}
        if kind == "push":
            position, quaternion = sim.get_object_pose(task.object)
            _, frame = sim.get_object_pose(success["frame"])
            self._state.update(
                position=position, quaternion=quaternion,
                axis=quat_to_matrix(frame) @ np.asarray(success["push_axis"], dtype=float),
                half_height=next(body.size[2] for body in sim.config.objects if body.name == task.object) / 2)
        if kind in ("object_tilt", "wiping"):
            self._state["position"], self._state["quaternion"] = sim.get_object_pose(task.object)
        self.index, self.done, self.success_time, self.error = -1, False, None, None
        self.observation = sim.observe(images=self.images)
        return self.observation

    def step(self, action: Action, images: bool = True) -> dict[str, Any]:
        """Apply `action` for one control step (1/15 s); images=False skips rendering."""
        if self.done:
            raise RuntimeError("the episode is over: call reset()")
        self.index += 1
        observation = self.sim.step(action, dt=control_dt(self.index), images=self.images and images)
        self.observation = observation
        achieved = self._check(observation)
        if achieved and self.task.success["type"] != "wiping":
            self.success_time = float(observation["time"])
        if achieved or self.index + 1 >= self.task.max_steps or self._state.get("violation"):
            self.done = True
        return observation

    def _check(self, observation: dict[str, Any]) -> bool:
        task, sim, state, success = self.task, self.sim, self._state, self.task.success
        kind = success["type"]
        if kind == "object_tilt":
            _, quaternion = sim.get_object_pose(task.object)
            state["final_tilt"] = tilt_degrees(state["quaternion"], quaternion)
            state["max_tilt"] = max(state["max_tilt"], state["final_tilt"])
            return state["final_tilt"] >= success["target_degrees"] - success["tolerance_degrees"]
        if kind == "wiping":
            position, quaternion = sim.get_object_pose(task.object)
            state["final_tilt"] = tilt_degrees(state["quaternion"], quaternion)
            state["max_tilt"] = max(state["max_tilt"], state["final_tilt"])
            state["shift"] = 1000.0 * float(np.linalg.norm((np.asarray(position) - state["position"])[:2]))
            marks = observation.get("visual_marks") or sim.visual_marks.state()
            state["erased"] = float(marks["erased_fraction"])
            if (self.success_time is None and state["erased"] >= success["min_erased_fraction"]
                    and state["shift"] <= success["max_object_shift_mm"]):
                self.success_time = float(observation["time"])
            return state["erased"] >= 1.0  # judged at the end; stops early only once the line is gone
        if kind == "push":
            position, quaternion = sim.get_object_pose(task.object)
            state["push_mm"] = 1000.0 * float((np.asarray(position) - state["position"]) @ state["axis"])
            state["final_tilt"] = tilt_degrees(state["quaternion"], quaternion)
            state["max_tilt"] = max(state["max_tilt"], state["final_tilt"])
            # Only the top face may be touched: a tool contact more than top_margin below the top plane is on a
            # side, and no other robot link may touch.
            contacts = sim.object_contacts(task.object)
            up = quat_to_matrix(quaternion)[:, 2]
            depth = (np.asarray(position) + state["half_height"] * up - contacts["tool_points"]) @ up
            if contacts["other_links"]:
                state["violation"] = "arm"
            elif len(depth) and float(depth.max()) > success["top_margin_mm"] / 1000.0:
                state["violation"] = "side"
            elif state["final_tilt"] > success["max_tilt_degrees"]:
                state["violation"] = "tipped"
            return state.get("violation") is None and state["push_mm"] >= success["min_push_mm"]
        insertion = observation["insertion"]
        depth_mm = 1000.0 * float(insertion["depth_m"])
        state["max_depth"] = max(state.get("max_depth", -math.inf), depth_mm)
        state["lateral"] = 1000.0 * float(insertion["lateral_error_m"])
        return depth_mm >= success["min_depth_mm"] and state["lateral"] <= success["max_lateral_mm"]

    def result(self) -> EpisodeResult:
        kind, state, success = self.task.success["type"], self._state, self.task.success
        passed = self.success_time is not None and self.error is None
        if kind == "wiping":
            passed = (self.error is None and state.get("erased") is not None
                      and state["erased"] >= success["min_erased_fraction"]
                      and state["shift"] <= success["max_object_shift_mm"])
        return EpisodeResult(
            task=self.task.name, condition=self.condition.name, trial=self.trial, seed=self.seed, success=passed,
            success_time_s=self.success_time if passed else None, steps=self.index + 1, error=self.error,
            max_tilt_deg=state["max_tilt"] if kind in ("object_tilt", "wiping", "push") else None,
            max_insertion_depth_mm=state.get("max_depth"), final_lateral_error_mm=state.get("lateral"),
            erased_fraction=state.get("erased"), object_shift_mm=state.get("shift"), push_mm=state.get("push_mm"),
            contact_violation=state.get("violation"))
