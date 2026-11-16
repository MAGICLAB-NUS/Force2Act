"""Observation encoder of the global force policy: image, wrench and pose tokens.

    image  [B, T, 3, H, W]  -> CLIP ViT-B/16, every patch token of every frame   [B, T * N, D]
    force  [B, Tf, 6]       -> per-sample MLP                                     [B, Tf, D]
    pose   [B, T, 6]        -> linear projection                                  [B, T, D]

The tokens are concatenated in that order and cross-attended by the action head.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ImageEncoder(nn.Module):
    """A timm ViT returning all tokens (no pooling). Images are RGB in [0, 1] (or [0, 255]), resized to the
    model's input size with bilinear interpolation; there is no further pixel normalization."""

    def __init__(self, model_name: str = "vit_base_patch16_clip_224.openai", image_size: int = 224,
                 width: int = 768, pretrained: bool = True, frozen: bool = False) -> None:
        super().__init__()
        import timm

        if not model_name.startswith("vit"):
            raise ValueError(f"only ViT image encoders are supported, got {model_name!r}")
        self.vit = timm.create_model(model_name=model_name, pretrained=pretrained, global_pool="", num_classes=0)
        if frozen:
            if not pretrained:
                raise ValueError("a frozen image encoder must be pretrained")
            self.vit.requires_grad_(False)
        self.image_size = int(image_size)
        features = int(self.vit.num_features)
        self.projection = nn.Identity() if features == width else nn.Linear(features, width)
        self.width = int(width)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """image [B, T, 3, H, W] (or [B, T, H, W, 3]) -> tokens [B, T * N, width]."""
        if image.max() > 1.0:
            image = image / 255.0
        if image.shape[-1] == 3:
            image = image.permute(0, 1, 4, 2, 3)
        batch, steps = image.shape[:2]
        image = image.reshape(batch * steps, *image.shape[2:])
        if tuple(image.shape[2:]) != (self.image_size, self.image_size):
            image = F.interpolate(image, size=(self.image_size, self.image_size), mode="bilinear",
                                  align_corners=False)
        device = next(self.parameters()).device
        tokens = self.projection(self.vit(image.to(device)).to(device))
        return tokens.reshape(batch, -1, self.width)


class ForceEncoder(nn.Module):
    """Per-sample MLP over the wrench history: [B, Tf, F] -> [B, Tf, D]."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256, n_layer: int = 2,
                 dropout: float = 0.1) -> None:
        super().__init__()
        if n_layer < 1:
            raise ValueError("n_layer must be positive")
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
        for _ in range(n_layer - 1):
            layers.extend([nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        layers.extend([nn.Linear(hidden_dim, output_dim), nn.LayerNorm(output_dim)])
        self.mlp = nn.Sequential(*layers)
        self.input_dim, self.output_dim = input_dim, output_dim

    def forward(self, force: torch.Tensor) -> torch.Tensor:
        return self.mlp(force)


class ObservationEncoder(nn.Module):
    """[image tokens of every frame, wrench tokens, pose tokens] -> [B, L, D]."""

    def __init__(self, image_encoder: ImageEncoder, force_encoder: ForceEncoder, obs_steps: int,
                 force_steps: int, pose_dim: int, width: int) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.force_encoder = force_encoder
        self.pose_projection = nn.Linear(pose_dim, width)
        self.obs_steps, self.force_steps, self.pose_dim, self.width = obs_steps, force_steps, pose_dim, width
        with torch.no_grad():
            probe = torch.zeros(1, obs_steps, 3, image_encoder.image_size, image_encoder.image_size)
            self.image_tokens_per_step = image_encoder(probe).shape[1] // obs_steps

    @property
    def num_tokens(self) -> int:
        return self.obs_steps * (self.image_tokens_per_step + 1) + self.force_steps

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        image, pose, force = obs["image"], obs["pose"], obs["force"]
        batch, steps = image.shape[:2]
        if steps != self.obs_steps:
            raise ValueError(f"expected {self.obs_steps} image steps, received {steps}")
        if pose.shape != (batch, steps, self.pose_dim):
            raise ValueError(f"unexpected pose shape {tuple(pose.shape)}")
        if force.shape != (batch, self.force_steps, self.force_encoder.input_dim):
            raise ValueError(f"unexpected force shape {tuple(force.shape)}")
        image_tokens = self.image_encoder(image)
        if image_tokens.shape != (batch, steps * self.image_tokens_per_step, self.width):
            raise ValueError(f"unexpected image token shape {tuple(image_tokens.shape)}")
        return torch.cat([image_tokens, self.force_encoder(force), self.pose_projection(pose)], dim=1)
