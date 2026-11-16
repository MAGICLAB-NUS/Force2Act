"""MoFE at inference: expert probabilities and every expert's (alpha, beta) for one step, and the hysteresis that
selects the expert to command."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from force2act.mofe.gates import EXPERTS, physical_inputs, router_features, tool_z_axis
from force2act.mofe.model import MoFE

WEIGHTS = "model.safetensors"
CONFIG = "config.yaml"


def build_mofe(config: dict[str, Any], pretrained: bool = True) -> MoFE:
    return MoFE(int(config["feature_dim"]), int(config["history"]), image_size=int(config["image_size"]),
                offset_bound=float(config["offset_bound"]), max_force=float(config["max_force"]), pretrained=pretrained,
                router_sees_direction=bool(config.get("router_sees_direction", False)),
                one_per_case=bool(config.get("one_per_case", False)))


def save_mofe(model: MoFE, config: dict[str, Any], directory: str | Path) -> Path:
    from safetensors.torch import save_model

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    save_model(model, str(directory / WEIGHTS))
    (directory / CONFIG).write_text(yaml.safe_dump(config, sort_keys=False))
    return directory


class MoFELaw:
    """A trained MoFE (directory with model.safetensors and config.yaml)."""

    def __init__(self, path: str | Path, device: str | torch.device = "cuda") -> None:
        from safetensors.torch import load_model

        from force2act.policy.build import resolve

        directory = resolve(path)
        self.config = yaml.safe_load((directory / CONFIG).read_text())
        self.device = torch.device(device)
        self.model = build_mofe(self.config, pretrained=False)
        load_model(self.model, str(directory / WEIGHTS), strict=True)
        self.model.to(self.device).eval()
        self.history = int(self.config["history"])
        self.include_torque = bool(self.config["include_torque"])
        self.min_speed, self.push_angle = float(self.config["min_speed"]), float(self.config["push_angle"])
        self.gates = bool(self.config.get("gates", True))

    def _padded(self, wrench: np.ndarray, velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The last `history` samples, the oldest repeated while the history is shorter."""
        wrench, velocity = np.asarray(wrench, float), np.asarray(velocity, float)
        if wrench.ndim == 1:
            wrench = wrench[None]
        velocity = velocity.reshape(len(velocity), -1)
        wrench, velocity = wrench[-self.history:], velocity[-self.history:]
        if len(wrench) < self.history:
            wrench = np.concatenate((np.repeat(wrench[:1], self.history - len(wrench), 0), wrench))
        if len(velocity) < self.history:
            velocity = np.concatenate((np.repeat(velocity[:1], self.history - len(velocity), 0), velocity))
        return wrench, velocity

    @torch.no_grad()
    def __call__(self, image: torch.Tensor, wrench: np.ndarray, velocity: np.ndarray,
                 tool_quaternion_wxyz: np.ndarray) -> dict[str, Any]:
        """image [3, H, W] RGB in [0, 1]; world wrench [T, 6] and tool velocity [T, >=3], oldest first; the current
        tool orientation. Returns the expert probabilities (0 where the gate is closed), every expert's
        (alpha, beta) and the measured tangential load T."""
        wrench, velocity = self._padded(wrench, velocity)
        features = router_features(wrench if self.include_torque else wrench[:, :3], velocity, self.min_speed,
                                   self.push_angle, include_torque=self.include_torque)
        tool_z = np.repeat(tool_z_axis(np.asarray(tool_quaternion_wxyz, float)), len(wrench), axis=0)
        x = physical_inputs(wrench[:, :3], velocity[:, :3], tool_z, self.min_speed, self.push_angle, self.gates)[-1]
        image_t = torch.as_tensor(image)[None].to(self.device)
        features_t = torch.from_numpy(features)[None].to(self.device)
        xt = torch.from_numpy(x)[None].to(self.device)
        logits = self.model.logits(image_t, features_t, xt).float()
        alpha, beta = self.model.expert_alpha_beta(xt)
        probs = torch.softmax(logits, -1)[0].cpu().numpy()
        return {"probs": probs, "alpha": alpha[0].cpu().numpy(), "beta": beta[0].cpu().numpy(),
                "tangential": float(x[0]), "lateral": float(x[1])}


class ExpertSelector:
    """Hysteresis on the router's choice: keep the current expert until another one beats it by `margin` in
    probability for `hold` consecutive steps; switch at once when the current expert's gate closes."""

    def __init__(self, margin: float = 0.2, hold: int = 3) -> None:
        self.margin, self.hold = float(margin), int(hold)
        self.reset()

    def reset(self) -> None:
        self.current: int | None = None
        self.streak = 0

    def __call__(self, probs: np.ndarray) -> int:
        probs = np.asarray(probs, float)
        best = int(probs.argmax())
        if self.current is None or probs[self.current] <= 0.0:
            self.current, self.streak = best, 0
        elif best != self.current and probs[best] - probs[self.current] > self.margin:
            self.streak += 1
            if self.streak >= self.hold:
                self.current, self.streak = best, 0
        else:
            self.streak = 0
        return self.current

    @staticmethod
    def name(index: int) -> str:
        return EXPERTS[index]
