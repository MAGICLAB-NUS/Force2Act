"""LeRobot (v3) demonstrations as training windows for the global force policy.

Each sample at frame t holds the last obs_steps camera frames and poses, the last wrench_steps wrenches and tool
velocities, and the targets of the chunk: the measured pose one frame after each chunk step (where the tool
actually went) and the measured wrench at each chunk step. The chunk starts at the oldest observed frame.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class DemonstrationDataset(Dataset):
    def __init__(self, root: str, image_key: str, pose_key: str, wrench_key: str, velocity_key: str, fps: int,
                 obs_steps: int, wrench_steps: int, horizon: int, instruction: str | None = None,
                 repo_id: str = "local/force2act", video_backend: str = "pyav") -> None:
        super().__init__()
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        if horizon < obs_steps:
            raise ValueError("horizon must be at least obs_steps")
        self.image_key, self.pose_key, self.wrench_key, self.velocity_key = image_key, pose_key, wrench_key, velocity_key
        self.obs_steps, self.wrench_steps, self.horizon, self.instruction = obs_steps, wrench_steps, horizon, instruction
        dt = 1.0 / fps
        observation = [(i - (obs_steps - 1)) * dt for i in range(obs_steps)]
        wrench = [(i - (wrench_steps - 1)) * dt for i in range(wrench_steps)]
        chunk = [(i - (obs_steps - 1)) * dt for i in range(horizon)]
        label = [offset + dt for offset in chunk]          # measured pose one frame after each chunk step
        wrench_window = sorted(set(wrench + chunk))
        pose_window = sorted(set(observation + label))
        self._wrench_history = [wrench_window.index(x) for x in wrench]
        self._wrench_target = [wrench_window.index(x) for x in chunk]
        self._pose_observed = [pose_window.index(x) for x in observation]
        self._pose_label = [pose_window.index(x) for x in label]
        self.dataset = LeRobotDataset(repo_id=repo_id, root=root, video_backend=video_backend, delta_timestamps={
            image_key: observation, pose_key: pose_window, wrench_key: wrench_window, velocity_key: wrench_window})

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "DemonstrationDataset":
        data = config["data"]
        return cls(root=data["root"], image_key=data["image_key"], pose_key=data["pose_key"],
                   wrench_key=data["wrench_key"], velocity_key=data["velocity_key"], fps=int(data["fps"]),
                   obs_steps=int(data["obs_steps"]), wrench_steps=int(data["wrench_steps"]),
                   horizon=int(data["horizon"]),
                   instruction=config["instruction"] if config["policy"].get("language_model") else None,
                   repo_id=data.get("repo_id", "local/force2act"))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[index]
        image = torch.as_tensor(sample[self.image_key], dtype=torch.float32)
        pose = torch.as_tensor(sample[self.pose_key], dtype=torch.float32)[:, :6]
        wrench = torch.as_tensor(sample[self.wrench_key], dtype=torch.float32)[:, :6]
        velocity = torch.as_tensor(sample[self.velocity_key], dtype=torch.float32)
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.shape[0] != self.obs_steps or pose.shape[0] != len(set(self._pose_observed + self._pose_label)):
            raise ValueError(f"unexpected window shapes {tuple(image.shape)}, {tuple(pose.shape)}")
        result = {
            "obs": {"image": image, "pose": pose[self._pose_observed], "force": wrench[self._wrench_history],
                    "velocity": velocity[self._wrench_history]},
            "action": pose[self._pose_label],
            "wrench_target": wrench[self._wrench_target],
        }
        if self.instruction is not None:
            result["obs"]["instruction"] = self.instruction
        return result

    def episode_index(self) -> np.ndarray:
        return np.asarray(self.dataset.hf_dataset["episode_index"], dtype=np.int64)
