"""Force2Act on Force2Act-Bench (hybrid control)."""

from __future__ import annotations

from typing import Any

import numpy as np

from force2act_bench.sim import Action
from force2act_bench.sim.geometry import quat_to_matrix


class Force2ActPolicy:
    def __init__(self, checkpoint: str, camera: str, mofe: str | None = "auto", device: str = "cuda",
                 instruction: str | None = None) -> None:
        import torch

        from force2act.agent import Force2ActAgent

        self.torch = torch
        self.agent = Force2ActAgent.from_pretrained(checkpoint, mofe=mofe, device=device, instruction=instruction)
        self.camera = camera

    def reset(self, seed: int) -> None:
        self.torch.manual_seed(seed)
        self.agent.reset()

    def needs_image(self) -> bool:
        return self.agent.needs_image()

    def act(self, observation: dict[str, Any]) -> Action:
        quaternion = observation["tcp_quaternion_wxyz"]
        rotation = quat_to_matrix(quaternion)
        tool_wrench = np.asarray(observation["tcp_wrench"], dtype=float)  # environment on tool, TCP axes
        world_wrench = np.concatenate((rotation @ tool_wrench[:3], rotation @ tool_wrench[3:]))
        image = observation.get("rgb", {}).get(self.camera)
        command = self.agent.act(observation["tcp_position"], quaternion, world_wrench, image)
        wrench = np.asarray(command.wrench, dtype=float)
        action = Action(tcp_position=command.position, tcp_quaternion_wxyz=command.quaternion_wxyz,
                        tcp_wrench=wrench if np.isfinite(wrench).all() else None)
        action.force_control_axes = command.force_axes.copy()
        action.force_frame_rotation_wxyz = command.frame_quaternion_wxyz.copy()
        return action
