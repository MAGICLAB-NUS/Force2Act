"""Mixture of Force Experts (MoFE): seven contact laws for the normal force and a router that selects one.

    pushing      F_n* = b_push
    sticking_1   F_n* = c_1 F_t + b_1          b_1 <= offset_bound
    sticking_2   F_n* = c_2 F_t + b_2          b_2 <= offset_bound
    sliding_k    F_n* = L_k,  k = 1, 2, 3
    insertion    F_n* = c_i |F_perp| + b_i

The constants are learned (no physical parameter is supplied). The router (ResNet-18 on the camera frame + MLP on
the force history, press-blind: only |F_t| and the tool speed per step) scores the four contact mechanisms; the
sliding level is chosen from the image alone. Experts whose gate is closed get -inf. One expert is used per
step (straight-through Gumbel-softmax in training, argmax with hysteresis at inference).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from force2act.mofe.gates import EXPERTS

MECHANISMS = ["pushing", "sticking_1", "sticking_2", "sliding", "insertion"]
SLIDING_LEVELS = ["sliding_1", "sliding_2", "sliding_3"]
CONSTANTS = ["pushing_level", "sticking_1_slope", "sticking_1_offset", "sticking_2_slope", "sticking_2_offset",
             "sliding_level_1", "sliding_level_2", "sliding_level_3", "insertion_slope", "insertion_offset"]
BOUNDED = ("sticking_1_offset", "sticking_2_offset")   # in [0, offset_bound]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def law_text(k: dict[str, float]) -> dict[str, str]:
    return {"pushing": f"Fn = {k['pushing_level']:.2f}",
            "sticking_1": f"Fn = {k['sticking_1_slope']:.2f} Ft + {k['sticking_1_offset']:.2f}",
            "sticking_2": f"Fn = {k['sticking_2_slope']:.2f} Ft + {k['sticking_2_offset']:.2f}",
            "sliding_1": f"Fn = {k['sliding_level_1']:.2f}", "sliding_2": f"Fn = {k['sliding_level_2']:.2f}",
            "sliding_3": f"Fn = {k['sliding_level_3']:.2f}",
            "insertion": f"Fn = {k['insertion_slope']:.2f} |Fperp| + {k['insertion_offset']:.2f}"}


class MoFE(nn.Module):
    def __init__(self, feature_dim: int, history: int, image_size: int = 224, hidden: int = 256,
                 dropout: float = 0.2, offset_bound: float = 2.0, max_force: float = 60.0, pretrained: bool = True,
                 router_sees_direction: bool = False, one_per_case: bool = False) -> None:
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        self.image_size, self.offset_bound, self.max_force = image_size, float(offset_bound), float(max_force)
        self.one_per_case = bool(one_per_case)
        # Press-blind router input: per step the features are (|Ft|, speed, [lever arm x3], direction x3);
        # only |Ft| and speed are kept (router_sees_direction: all of them, ablation).
        per_step = feature_dim // history
        mask = torch.zeros(feature_dim)
        for i in range(history):
            if router_sees_direction:
                mask[i * per_step:(i + 1) * per_step] = 1.0
            else:
                mask[i * per_step] = 1.0
                mask[i * per_step + 1] = 1.0
        self.register_buffer("router_feature_mask", mask)
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        width = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.force_mlp = nn.Sequential(nn.Linear(feature_dim, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU())
        self.trunk = nn.Sequential(nn.Linear(width + 128, hidden), nn.GELU(), nn.Dropout(dropout))
        self.router = nn.Linear(hidden, len(MECHANISMS))
        self.level_head = nn.Linear(width, len(SLIDING_LEVELS))   # sliding level from the image alone
        nn.init.zeros_(self.level_head.weight)
        nn.init.zeros_(self.level_head.bias)
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)
        self.raw = nn.ParameterDict({name: nn.Parameter(torch.zeros(())) for name in CONSTANTS})
        self.register_buffer("init", torch.ones(len(CONSTANTS)))
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.register_buffer("feature_mean", torch.zeros(feature_dim))
        self.register_buffer("feature_std", torch.ones(feature_dim))

    def constant(self, name: str) -> torch.Tensor:
        """raw = 0 gives the initial value; every constant stays in its physical range."""
        raw, init = self.raw[name], self.init[CONSTANTS.index(name)]
        if name in BOUNDED:
            frac = (init / self.offset_bound).clamp(1e-3, 1 - 1e-3)
            return torch.sigmoid(raw + torch.log(frac / (1 - frac))) * self.offset_bound
        return F.softplus(raw) / np.log(2.0) * init  # >= 0

    def constants(self) -> dict[str, float]:
        return {name: float(self.constant(name).detach()) for name in CONSTANTS}

    def expert_alpha_beta(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x [B, 9] = (T, |Fperp|, admissible[7]) -> every expert as F_n* = alpha * T + beta, [B, 7] each."""
        k = self.constant
        ones = torch.ones_like(x[:, 0])
        if self.one_per_case:  # ablation: one sticking law and one sliding level
            c2, b2, l2, l3 = k("sticking_1_slope"), k("sticking_1_offset"), k("sliding_level_1"), k("sliding_level_1")
        else:
            c2, b2, l2, l3 = k("sticking_2_slope"), k("sticking_2_offset"), k("sliding_level_2"), k("sliding_level_3")
        alpha = torch.stack([ones * 0, k("sticking_1_slope") * ones, c2 * ones, ones * 0, ones * 0, ones * 0,
                             ones * 0], dim=-1)
        beta = torch.stack([k("pushing_level") * ones, k("sticking_1_offset") * ones, b2 * ones,
                            k("sliding_level_1") * ones, l2 * ones, l3 * ones,
                            k("insertion_slope") * x[:, 1] + k("insertion_offset")], dim=-1)
        return alpha, beta

    def expert_outputs(self, x: torch.Tensor) -> torch.Tensor:
        alpha, beta = self.expert_alpha_beta(x)
        return (alpha * x[:, 0:1] + beta).clamp(0.0, self.max_force)

    def logits(self, image: torch.Tensor, features: torch.Tensor, x: torch.Tensor, jitter: float = 0.0) -> torch.Tensor:
        """image [B, 3, H, W] in [0, 1], features [B, feature_dim], x [B, 9] -> [B, 7]; closed gates at -inf."""
        image = F.interpolate(image.float(), size=(self.image_size, self.image_size), mode="bilinear",
                              align_corners=False, antialias=True)
        visual = self.backbone((image - self.mean) / self.std)
        f = (features.float() - self.feature_mean) / self.feature_std * self.router_feature_mask
        trunk = self.trunk(torch.cat([visual, self.force_mlp(f)], -1))
        if jitter > 0:
            trunk = trunk * torch.empty_like(trunk).uniform_(1.0 - jitter, 1.0 + jitter)
        mechanism = self.router(trunk).float()                                   # [B, 5]
        level = torch.log_softmax(self.level_head(visual).float(), dim=-1)       # [B, 3]
        sliding = mechanism[:, MECHANISMS.index("sliding"), None] + level
        logits = torch.cat([mechanism[:, :3], sliding, mechanism[:, 4:5]], dim=-1)
        return logits.masked_fill(x[:, 2:2 + len(EXPERTS)] <= 0.5, float("-inf"))

    def forward(self, image, features, x, tau: float | None = None, jitter: float = 0.0):
        """Normal force of the selected expert [B, 1], its index [B], every expert's output [B, 7], probabilities."""
        logits = self.logits(image, features, x, jitter)
        choice = (F.one_hot(logits.argmax(-1), len(EXPERTS)).float() if tau is None
                  else F.gumbel_softmax(logits, tau=tau, hard=True))
        outputs = self.expert_outputs(x.float())
        return (choice * outputs).sum(-1, keepdim=True), choice.argmax(-1), outputs, torch.softmax(logits, -1)
