"""Force2Act at run time: the global force policy once per chunk, MoFE and the compliance command every step.

    agent = Force2ActAgent.from_pretrained("hf://k3rth1v5n/Force2Act-Sim/pivot_block")
    agent.reset()
    command = agent.act(position, quaternion_wxyz, wrench_world, image)   # every control step

Each call takes the measured tool pose, the measured wrench (environment on tool, world axes) and the camera
frame, and returns the next pose target, the force-controlled axes in a per-step contact frame and the commanded
wrench in that frame.
"""

from __future__ import annotations

from collections import deque
import posixpath
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from force2act.control.contact import NormalForceCommand
from force2act.control.motion_frame import ControlConfig, motion_frame
from force2act.mofe.law import ExpertSelector, MoFELaw
from force2act.policy.build import load_policy, resolve
from force2act.policy.policy import Force2ActPolicy
from force2act.policy.rtc import prefix_weights, shift_previous_chunk
from force2act.utils.rotation import quat_to_matrix, quat_to_rpy, rpy_to_quat


@dataclass
class Command:
    position: np.ndarray                 # [3] m, world
    quaternion_wxyz: np.ndarray          # [4]
    wrench: np.ndarray                   # [6] in the contact frame (N, Nm); only force_axes are used
    force_axes: np.ndarray               # [6] bool, axes of the contact frame under force control
    frame_quaternion_wxyz: np.ndarray    # [4] contact frame (columns: motion, y, push) in world coordinates
    info: dict[str, Any]


def image_tensor(image: np.ndarray) -> torch.Tensor:
    """HWC (or CHW) RGB, uint8 or float -> float32 CHW in [0, 1]."""
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"image must be HWC or CHW, received {image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.moveaxis(image, 0, -1)
    if np.issubdtype(image.dtype, np.integer):
        image = image.astype(np.float32) / 255.0
    else:
        image = image.astype(np.float32)
        if image.max(initial=0.0) > 1.0:
            image /= 255.0
    return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)


def _padded(history: deque, length: int) -> list:
    values = list(history)
    return [values[0]] * (length - len(values)) + values


class Force2ActAgent:
    def __init__(self, policy: Force2ActPolicy, config: dict[str, Any], mofe: MoFELaw | None = None,
                 instruction: str | None = None, actions_per_chunk: int | None = None) -> None:
        self.policy, self.config, self.mofe = policy, config, mofe
        self.device = policy.device
        data, inference = config["data"], config["inference"]
        self.fps = float(data["fps"])
        self.obs_steps, self.wrench_steps = int(data["obs_steps"]), int(data["wrench_steps"])
        self.instruction = instruction or config["instruction"]
        self.action_steps = policy.action_steps
        self.actions_per_chunk = int(actions_per_chunk or self.action_steps)
        self.control = ControlConfig.from_dict(inference["control"])
        if self.control.w > 0 and mofe is None:
            raise ValueError("control.w > 0 needs a MoFE model")
        rtc = inference.get("rtc") or {}
        self.rtc = bool(rtc.get("enabled", False))
        self.rtc_max_guidance_weight = float(rtc.get("max_guidance_weight", 5.0))
        self.rtc_weights = prefix_weights(start=int(rtc.get("frozen_steps", self.obs_steps)),
                                          end=policy.horizon - self.action_steps, total=policy.horizon,
                                          schedule=str(rtc.get("schedule", "exp")))
        hysteresis = inference.get("hysteresis") or {}
        self.selector = ExpertSelector(float(hysteresis.get("margin", 0.2)), int(hysteresis.get("hold", 3)))
        self._images: deque = deque(maxlen=self.obs_steps)
        self._poses: deque = deque(maxlen=self.obs_steps)
        self._forces: deque = deque(maxlen=self.wrench_steps)
        self._velocities: deque = deque(maxlen=self.wrench_steps)
        self._normal = NormalForceCommand()
        self.reset()

    @classmethod
    def from_pretrained(cls, path: str, mofe: str | None = "auto", device: str = "cuda",
                        **kwargs) -> "Force2ActAgent":
        """path: a policy directory or hf://<owner>/<repo>/<task>. mofe: "auto" (the MoFE named in the policy's
        config, a path relative to the policy), a directory / hf:// path, or None for the global policy alone
        (requires control.w = 0)."""
        policy, config = load_policy(path, device)
        if mofe == "auto":
            mofe = config["inference"].get("mofe")
            if mofe and not str(mofe).startswith(("hf://", "/", "~")):
                # relative to the policy location: ../mofe next to the task folders, locally or on the Hub
                mofe = ("hf://" + posixpath.normpath(posixpath.join(str(path)[len("hf://"):], mofe))
                        if str(path).startswith("hf://") else str(resolve(path) / mofe))
        law = MoFELaw(mofe, device) if mofe else None
        return cls(policy, config, law, **kwargs)

    # -------------------------------------------------------------------------------------------- state
    def reset(self) -> None:
        """Start a new episode."""
        self._pending: list = []
        self._images.clear()
        self._poses.clear()
        self._forces.clear()
        self._velocities.clear()
        self._previous_tcp = None
        self._normal.reset()
        self.selector.reset()
        self._previous_chunk = None
        self._previous_scale = None

    def needs_image(self) -> bool:
        """MoFE reads the camera at every step; the global policy only at the frames of its next chunk."""
        return self.mofe is not None or len(self._pending) < self.obs_steps

    def _record_motion(self, position: np.ndarray, quaternion: np.ndarray) -> None:
        """World [linear, angular] tool velocity by finite difference at the control rate."""
        position = np.asarray(position, dtype=float)
        rotation = quat_to_matrix(quaternion)
        velocity = np.zeros(6)
        if self._previous_tcp is not None:
            previous_position, previous_rotation = self._previous_tcp
            velocity[:3] = (position - previous_position) * self.fps
            delta = rotation @ previous_rotation.T
            angle = np.arccos(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0))
            if angle > 1e-9:
                axis = np.array([delta[2, 1] - delta[1, 2], delta[0, 2] - delta[2, 0],
                                 delta[1, 0] - delta[0, 1]]) / (2.0 * np.sin(angle))
                velocity[3:] = axis * angle * self.fps
        self._previous_tcp = (position.copy(), rotation)
        self._velocities.append(velocity.astype(np.float32))

    # ----------------------------------------------------------------------------------------- one step
    def act(self, position: np.ndarray, quaternion_wxyz: np.ndarray, wrench_world: np.ndarray,
            image: np.ndarray | None) -> Command:
        """One control step. image: HWC RGB (uint8), required when needs_image() is True."""
        wrench_world = np.asarray(wrench_world, dtype=float)
        rpy = quat_to_rpy(quaternion_wxyz)
        state = np.concatenate((position, rpy, wrench_world)).astype(np.float32)
        self._poses.append(state[:6].copy())
        self._forces.append(state[6:].copy())
        self._record_motion(position, quaternion_wxyz)
        if image is not None:
            self._images.append(np.asarray(image).copy())
        elif not self._pending:
            raise ValueError("a camera frame is needed at this step (needs_image() was True)")
        if not self._pending:
            self._plan_chunk()
        return self._step(*self._pending.pop(0), wrench_world, image)

    @torch.no_grad()
    def _plan_chunk(self) -> None:
        images = _padded(self._images, self.obs_steps)
        poses = np.stack(_padded(self._poses, self.obs_steps)).astype(np.float32)
        forces = np.stack(_padded(self._forces, self.wrench_steps)).astype(np.float32)
        velocities = np.stack(_padded(self._velocities, self.wrench_steps)).astype(np.float32)
        obs = {"image": torch.stack([image_tensor(v) for v in np.stack(images)])[None].to(self.device),
               "pose": torch.from_numpy(poses)[None].to(self.device),
               "force": torch.from_numpy(forces)[None].to(self.device),
               "instruction": [self.instruction]}
        rtc = None
        if self.rtc and self._previous_chunk is not None:
            rtc = {"target": shift_previous_chunk(self._previous_chunk, self.action_steps)[None].to(self.device),
                   "weights": self.rtc_weights[None].to(self.device),
                   "max_guidance_weight": self.rtc_max_guidance_weight}
            if self._previous_scale is not None:
                rtc["scale"] = self._previous_scale[None].to(self.device)
        # Guidance needs gradients through the network, so no inference_mode then.
        with (torch.no_grad() if rtc is not None else torch.inference_mode()):
            result = self.policy.predict_action(obs, rtc=rtc)
            self._previous_chunk = result["normalized_chunk"][0].detach().clone()
            self._previous_scale = result["scale"][0].detach().clone() if "scale" in result else None
        actions = result["action"][0].detach().float().cpu().numpy().astype(np.float64)
        wrenches = result["wrench"][0].detach().float().cpu().numpy()
        frame = motion_frame(current_position=poses[-1, :3], chunk_positions=actions[:, :3],
                             measured_forces=forces[:, :3], measured_velocities=velocities[:, :3],
                             predicted_forces=wrenches[:, :3], config=self.control)
        actions[:, 3:6] = np.remainder(actions[:, 3:6] + np.pi, 2 * np.pi) - np.pi
        count = min(self.actions_per_chunk, len(actions))
        planned = np.asarray(frame["planned_wrench"], dtype=np.float64)[:count].copy()
        mask = frame["mask"]
        planned[:, ~mask] = 0.0
        compliance = (frame["contact"] and self.mofe is not None)
        policy_normal = np.asarray(frame["policy_normal"], float).reshape(-1)
        self._pending = [(actions[k], planned[k], mask.copy(), frame["quaternions"][k], compliance,
                          policy_normal[k]) for k in range(count)]

    def _step(self, target: np.ndarray, wrench: np.ndarray, mask: np.ndarray, frame_quaternion: np.ndarray,
              compliance: bool, policy_normal: float, wrench_world: np.ndarray, image: np.ndarray | None) -> Command:
        info: dict[str, Any] = {}
        if compliance and mask[2]:
            if image is None:
                raise ValueError("MoFE needs the camera frame at every step")
            # MoFE on this step's frame and force history; the tool axis from the latest pose.
            quaternion = rpy_to_quat(self._poses[-1][3:6]).astype(np.float32).astype(np.float64)
            out = self.mofe(image_tensor(image), np.stack(self._forces).astype(np.float32),
                            np.stack(self._velocities).astype(np.float32), quaternion)
            expert = self.selector(out["probs"])
            params = {"alpha": float(out["alpha"][expert]), "beta": float(out["beta"][expert]),
                      "w": self.control.w, "blend": self.control.blend, "f_max": self.control.f_max,
                      "min_speed": self.control.min_speed, "push_angle": self.control.push_angle}
            wrench = np.array(wrench, dtype=float)
            velocity = self._velocities[-1][:3]
            wrench[2] = self._normal.step(wrench_world[:3], velocity, quat_to_matrix(frame_quaternion), params,
                                          policy_normal)
            info = {"expert": ExpertSelector.name(expert), "probs": out["probs"], **self._normal.last}
        else:
            self._normal.reset()
        wrench = np.asarray(wrench, dtype=float)
        return Command(position=np.asarray(target[:3], float), quaternion_wxyz=rpy_to_quat(target[3:6]),
                       wrench=wrench, force_axes=np.asarray(mask, dtype=bool).copy(),
                       frame_quaternion_wxyz=np.asarray(frame_quaternion, float).copy(), info=info)
