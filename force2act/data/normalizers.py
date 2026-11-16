"""Fit the policy's normalizers on the training frames."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from force2act.policy import relative_force
from force2act.utils.normalizer import SingleFieldLinearNormalizer
from force2act.utils.rotation import install_rpy_center, rpy_indices, xyzrpy_to_xyz_rotation_6d


def fit_normalizers(policy, dataset, train_indices: list[int], config: dict[str, Any]) -> None:
    data = config["data"]
    rows = dataset.dataset.hf_dataset.select(train_indices)
    pose = torch.as_tensor(np.stack(rows[data["pose_key"]]), dtype=torch.float32)[:, :6]
    wrench = torch.as_tensor(np.stack(rows[data["wrench_key"]]), dtype=torch.float32)[:, :6]
    action = pose.clone()  # pose labels: the measured pose
    observed = install_rpy_center(policy.normalizer, pose, rpy_indices(range(6)))
    policy.normalizer["image"] = SingleFieldLinearNormalizer.create_identity()
    policy.normalizer["pose"] = SingleFieldLinearNormalizer.create_fit(observed, mode="limits")
    force_input = wrench_target = wrench
    floor = config["policy"].get("relative_force_floor")
    if floor is not None:
        obs_steps = int(data["obs_steps"])
        inputs, targets = relative_force.fit_arrays(
            wrench.numpy(), np.asarray(rows["episode_index"]), int(data["wrench_steps"]),
            list(range(1 - obs_steps, int(data["horizon"]) + 1 - obs_steps)), float(floor))
        force_input, wrench_target = torch.from_numpy(inputs), torch.from_numpy(targets)
        relative_force.install(policy.normalizer, float(floor))
    policy.normalizer["force"] = SingleFieldLinearNormalizer.create_fit(force_input, mode="gaussian")
    policy.normalizer["action"] = SingleFieldLinearNormalizer.create_fit(xyzrpy_to_xyz_rotation_6d(action),
                                                                         mode="limits")
    policy.normalizer["wrench_target"] = SingleFieldLinearNormalizer.create_fit(wrench_target, mode="gaussian")
