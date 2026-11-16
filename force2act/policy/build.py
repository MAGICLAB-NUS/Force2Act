"""Build the global force policy from a config, and save / load trained policies."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml

from force2act.policy.encoders import ForceEncoder, ImageEncoder, ObservationEncoder
from force2act.policy.policy import WRENCH_DIM, Force2ActPolicy

WEIGHTS = "model.safetensors"
CONFIG = "config.yaml"


def build_policy(config: dict[str, Any], pretrained: bool = True) -> Force2ActPolicy:
    """pretrained=True loads CLIP and T5 weights (training); False builds the architecture only (loading)."""
    data, policy = config["data"], config["policy"]
    width = int(policy["width"])
    image_encoder = ImageEncoder(policy["image_encoder"], int(policy["image_size"]), width,
                                 pretrained=pretrained, frozen=bool(policy.get("frozen_image_encoder", False)))
    force = policy["force_encoder"]
    force_encoder = ForceEncoder(WRENCH_DIM, width, int(force["hidden_dim"]), int(force["layers"]),
                                 float(force["dropout"]))
    obs_encoder = ObservationEncoder(image_encoder, force_encoder, int(data["obs_steps"]), int(data["wrench_steps"]),
                                     pose_dim=6, width=width)
    return Force2ActPolicy(
        obs_encoder, obs_steps=int(data["obs_steps"]), horizon=int(data["horizon"]),
        action_steps=int(policy["action_steps"]), sampling_steps=int(policy["sampling_steps"]),
        n_layer=int(policy["depth"]), n_head=int(policy["heads"]), n_emb=width,
        mlp_ratio=float(policy["mlp_ratio"]), dropout=float(policy["dropout"]), qk_norm=bool(policy["qk_norm"]),
        language_model=policy.get("language_model"), pretrained_language=pretrained)


def save_policy(policy: Force2ActPolicy, config: dict[str, Any], directory: str | Path) -> Path:
    from safetensors.torch import save_model

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    save_model(policy, str(directory / (WEIGHTS + ".partial")))
    (directory / (WEIGHTS + ".partial")).replace(directory / WEIGHTS)
    (directory / CONFIG).write_text(yaml.safe_dump(config, sort_keys=False))
    return directory


def resolve(path: str | Path) -> Path:
    """A local directory, or "hf://<org>/<repo>/<subfolder>" downloaded from the Hugging Face Hub."""
    text = str(path)
    if not text.startswith("hf://"):
        return Path(text).expanduser()
    from huggingface_hub import snapshot_download

    parts = text[len("hf://"):].strip("/").split("/")
    repo, subfolder = "/".join(parts[:2]), "/".join(parts[2:])
    root = Path(snapshot_download(repo, allow_patterns=[f"{subfolder}/*"] if subfolder else None))
    return root / subfolder if subfolder else root


def load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def load_policy(path: str | Path, device: str | torch.device = "cuda") -> tuple[Force2ActPolicy, dict[str, Any]]:
    """A trained policy (directory with model.safetensors and config.yaml) in eval mode."""
    from safetensors.torch import load_model

    directory = resolve(path)
    config = load_config(directory / CONFIG)
    policy = build_policy(config, pretrained=False)
    load_model(policy, str(directory / WEIGHTS), strict=True)
    return policy.to(device).eval(), config
