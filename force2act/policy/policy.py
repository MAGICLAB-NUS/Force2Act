"""Global force policy of Force2Act: a flow-matching policy over pose and wrench chunks.

Given the last two camera frames and poses, the last 0.5 s of wrench and the instruction, the policy samples a
chunk of H future poses (xyz + 6-D rotation) and wrenches (relative to the recent force magnitude):

    x_t = (1 - t) x_0 + t x_1,   x_0 ~ N(0, I),   x_1 = normalized chunk
    loss = || v(x_t, t | obs) - (x_1 - x_0) ||^2,   t = 0.999 * Beta(1, 1.5)

Sampling integrates v from t = 0 (noise) to 1 with 10 Euler steps, optionally with real-time chunking.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta

from force2act.policy import relative_force
from force2act.policy.action_head import ActionHead
from force2act.policy.encoders import ObservationEncoder
from force2act.policy.rtc import guided_sample
from force2act.utils.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from force2act.utils.rotation import xyz_rotation_6d_to_xyzrpy, xyzrpy_to_xyz_rotation_6d

POSE_DIM = 9      # xyz + 6-D rotation (internal action representation)
WRENCH_DIM = 6


class Force2ActPolicy(nn.Module):
    def __init__(self, obs_encoder: ObservationEncoder, obs_steps: int, horizon: int, action_steps: int,
                 sampling_steps: int = 10, n_layer: int = 12, n_head: int = 12, n_emb: int = 768,
                 mlp_ratio: float = 4.0, dropout: float = 0.1, qk_norm: bool = True,
                 language_model: str | None = "t5-small", pretrained_language: bool = True) -> None:
        super().__init__()
        self.obs_encoder = obs_encoder
        self.action_dim = POSE_DIM + WRENCH_DIM
        self.action_head = ActionHead(
            action_dim=self.action_dim, horizon=horizon, cond_dim=obs_encoder.width, cond_len=obs_encoder.num_tokens,
            n_layer=n_layer, n_head=n_head, n_emb=n_emb, mlp_ratio=mlp_ratio, dropout=dropout, qk_norm=qk_norm,
            language_model=language_model, pretrained_language=pretrained_language)
        self.normalizer = LinearNormalizer()
        for key in ("image", "pose", "force", "action", "wrench_target"):
            self.normalizer[key] = SingleFieldLinearNormalizer.create_identity()
        self.obs_steps, self.horizon, self.action_steps = obs_steps, horizon, action_steps
        self.sampling_steps = int(sampling_steps)
        self.obs_horizons = {"image": obs_steps, "pose": obs_steps, "force": obs_encoder.force_steps}

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def force_floor(self) -> float | None:
        """s_min of relative force, or None for an absolute-force policy."""
        return relative_force.floor(self.normalizer)

    # ------------------------------------------------------------------------------------------ helpers
    def _relative(self, obs: dict) -> tuple[dict, torch.Tensor | None]:
        obs = {key: value for key, value in obs.items() if key != "velocity"}
        floor = self.force_floor
        if floor is None:
            return obs, None
        scale = relative_force.scale(obs["force"], floor)
        return {**obs, "force": obs["force"] / scale[:, None]}, scale

    def _condition(self, nobs: dict, batch_size: int):
        encoder_inputs = {key: value[:, :self.obs_horizons[key]].to(self.device)
                          for key, value in nobs.items() if torch.is_tensor(value)}
        tokens = self.obs_encoder(encoder_inputs).reshape(batch_size, -1, self.obs_encoder.width)
        instruction = nobs.get("instruction") if self.action_head.language_conditioned else None
        if self.action_head.language_conditioned and instruction is None:
            raise ValueError("the policy is instruction-conditioned: the observation needs 'instruction'")
        return tokens, instruction

    @staticmethod
    def sample_t(batch_size: int, device) -> torch.Tensor:
        """pi0's flow-time distribution: 0.999 * Beta(1, 1.5)."""
        beta = Beta(torch.tensor([1.0], device=device), torch.tensor([1.5], device=device))
        return (0.999 * beta.sample((batch_size, 1, 1))).reshape(-1)

    @torch.no_grad()
    def _sample(self, x: torch.Tensor, **condition) -> torch.Tensor:
        dt = 1.0 / self.sampling_steps
        for i in range(self.sampling_steps):
            t = torch.full((x.shape[0],), i * dt, device=x.device)
            x = x + dt * self.action_head(x, t, **condition)
        return x

    # ---------------------------------------------------------------------------------------- inference
    def predict_action(self, obs: dict, rtc: dict | None = None) -> dict[str, torch.Tensor]:
        """obs: image [B, T, 3, H, W] in [0, 1], pose [B, T, 6] (xyz, rpy), force [B, Tf, 6] (world wrench),
        instruction (list of str). rtc: {"target", "weights", "max_guidance_weight", "scale"} from the previous
        chunk. Returns the executed chunk (action [B, n, 6], wrench [B, n, 6]) and the state for the next one."""
        obs, scale = self._relative(obs)
        if rtc is not None and scale is not None and "scale" in rtc:
            rtc = {**rtc, "target": self._rescale_target(rtc["target"], rtc["scale"], scale)}
        nobs = self.normalizer.normalize(obs)
        batch = next(v for v in nobs.values() if torch.is_tensor(v)).shape[0]
        vis_cond, instruction = self._condition(nobs, batch)
        x0 = torch.randn((batch, self.horizon, self.action_dim), device=self.device, dtype=self.dtype)
        if rtc is not None:
            sample = guided_sample(self.action_head, self.sampling_steps, x0, rtc["target"], rtc["weights"],
                                   float(rtc["max_guidance_weight"]), vis_cond=vis_cond, lang_cond=instruction)
        else:
            sample = self._sample(x0, vis_cond=vis_cond, lang_cond=instruction)
        poses = xyz_rotation_6d_to_xyzrpy(self.normalizer["action"].unnormalize(sample[..., :POSE_DIM]))
        wrenches = self.normalizer["wrench_target"].unnormalize(sample[..., POSE_DIM:])
        if scale is not None:
            wrenches = scale[:, None] * wrenches
        start, end = self.obs_steps - 1, self.obs_steps - 1 + self.action_steps
        result = {"action": poses[:, start:end], "wrench": wrenches[:, start:end], "normalized_chunk": sample}
        if scale is not None:
            result["scale"] = scale
        return result

    def _rescale_target(self, target, previous_scale, scale):
        """Express the previous chunk's normalized relative wrench in the current force scale."""
        wrench = self.normalizer["wrench_target"].unnormalize(target[..., POSE_DIM:])
        absolute = previous_scale.to(wrench)[:, None] * wrench
        rescaled = self.normalizer["wrench_target"].normalize(absolute / scale[:, None])
        return torch.cat((target[..., :POSE_DIM], rescaled), dim=-1)

    # ----------------------------------------------------------------------------------------- training
    def compute_loss(self, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
        obs, scale = self._relative(batch["obs"])
        wrench_target = batch["wrench_target"]
        if scale is not None:
            wrench_target = wrench_target / scale[:, None]
        nobs = self.normalizer.normalize(obs)
        x1 = self.normalizer["action"].normalize(xyzrpy_to_xyz_rotation_6d(batch["action"]))
        x1 = torch.cat((x1, self.normalizer["wrench_target"].normalize(wrench_target)), dim=-1).to(self.device)
        batch_size = x1.shape[0]
        vis_cond, instruction = self._condition(nobs, batch_size)
        t = self.sample_t(batch_size, self.device)
        x0 = torch.randn_like(x1)
        xt = (1.0 - t[:, None, None]) * x0 + t[:, None, None] * x1
        v = self.action_head(xt, t, vis_cond=vis_cond, lang_cond=instruction)
        error = F.mse_loss(v.float(), (x1 - x0).float(), reduction="none")
        loss = error.mean()
        return loss, {"loss": loss.item(), "loss_pose": error[..., :POSE_DIM].mean().item(),
                      "loss_wrench": error[..., POSE_DIM:].mean().item()}
