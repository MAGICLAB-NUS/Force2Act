"""Train the global force policy.

    python -m force2act.train_policy --config configs/policy/pivot_block.yaml --output runs/pivot_block

Flow matching on the demonstrations; one AdamW (betas 0.9, 0.95) with linear warm-up and cosine decay, mixed
precision, gradient clipping, and an EMA of the weights, which is what is saved and deployed.
"""

from __future__ import annotations

import argparse
import copy
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from force2act.data.dataset import DemonstrationDataset
from force2act.data.normalizers import fit_normalizers
from force2act.policy.build import build_policy, save_policy
from force2act.utils.ema import EMAModel
from force2act.utils.rotation import euler_xyz_to_matrix


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def episode_split(episodes: np.ndarray, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    if not 0 <= val_ratio < 1:
        raise ValueError("data.val_ratio must be in [0, 1)")
    unique = np.unique(episodes)
    np.random.default_rng(seed).shuffle(unique)
    held = set(unique[:max(1, round(len(unique) * val_ratio)) if val_ratio > 0 else 0].tolist())
    val = [i for i, e in enumerate(episodes) if e in held]
    train = [i for i, e in enumerate(episodes) if e not in held]
    return train, val


def move(batch, device: torch.device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {key: move(value, device) for key, value in batch.items()}
    return batch


def mean_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    return {key: sum(float(m[key]) for m in metrics) / len(metrics) for key in metrics[0]} if metrics else {}


@torch.no_grad()
def sampled_pose_error(policy, batch) -> dict[str, float]:
    """Sampled chunks of the EMA policy against the demonstration: position (mm) and rotation (deg) error."""
    was_training = policy.training
    policy.eval()
    start = policy.obs_steps - 1
    predicted = policy.predict_action(batch["obs"])["action"].float()
    truth = batch["action"][:, start:start + predicted.shape[1]].float().to(predicted.device)
    position = (predicted[..., :3] - truth[..., :3]).norm(dim=-1) * 1000.0
    relative = euler_xyz_to_matrix(predicted[..., 3:6]).transpose(-1, -2) @ euler_xyz_to_matrix(truth[..., 3:6])
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0).clamp(-1.0, 1.0)
    rotation = torch.rad2deg(torch.arccos(cosine))
    policy.train(was_training)
    return {"pos_error_mm": float(position.mean()), "rot_error_deg": float(rotation.mean())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", help="override data.root")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wandb", metavar="PROJECT", help="log to Weights & Biases")
    parser.add_argument("--max-steps", type=int, help="stop after this many steps (smoke test)")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if args.data_root:
        config["data"]["root"] = args.data_root
    train = config["train"]
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    device = torch.device(args.device)

    dataset = DemonstrationDataset.from_config(config)
    train_indices, val_indices = episode_split(dataset.episode_index(), float(config["data"].get("val_ratio", 0.0)),
                                               seed)
    loader_kwargs = {"batch_size": int(train["batch_size"]), "num_workers": int(train["num_workers"]),
                     "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(Subset(dataset, train_indices), shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(Subset(dataset, val_indices), shuffle=False, **loader_kwargs) if val_indices else None
    pose_samples = int(train.get("pose_eval_samples", 64))

    def fixed_batch(indices):
        if pose_samples <= 0 or not indices:
            return None
        spread = np.linspace(0, len(indices) - 1, min(pose_samples, len(indices))).astype(int)
        subset = Subset(dataset, [indices[i] for i in spread])
        return move(next(iter(DataLoader(subset, batch_size=len(spread), num_workers=0))), device)

    pose_batches = {"train": fixed_batch(train_indices), "val": fixed_batch(val_indices)}
    policy = build_policy(config).to(device)
    fit_normalizers(policy, dataset, train_indices, config)
    policy.to(device)

    lr, weight_decay = float(train["lr"]), float(train["weight_decay"])
    encoder = policy.obs_encoder
    groups = [dict(group, lr=lr) for group in policy.action_head.optim_groups(weight_decay)]
    for module, module_lr in ((encoder.image_encoder, train.get("image_encoder_lr", lr)),
                              (encoder.force_encoder, train.get("force_encoder_lr", lr)),
                              (encoder.pose_projection, train.get("pose_projection_lr", lr))):
        groups.append({"params": [p for p in module.parameters() if p.requires_grad], "lr": float(module_lr),
                       "weight_decay": weight_decay})
    in_optimizer = {id(p) for group in groups for p in group["params"]}
    left_out = [name for name, p in policy.named_parameters()
                if p.requires_grad and id(p) not in in_optimizer and not name.startswith("normalizer.")]
    if left_out:
        raise RuntimeError(f"trainable parameters outside the optimizer: {left_out[:8]}")
    optimizer = torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))
    base_lrs = [group["lr"] for group in optimizer.param_groups]
    parameters = [p for group in optimizer.param_groups for p in group["params"]]

    total_steps = max(1, int(train["epochs"]) * len(train_loader))
    warmup_steps, min_lr_ratio = int(train["warmup_steps"]), float(train["min_lr_ratio"])

    def lr_factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + np.cos(np.pi * progress))

    ema = EMAModel(copy.deepcopy(policy))
    use_amp = bool(train.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    clip_norm = float(train.get("grad_clip_norm", 1.0))
    run = None
    if args.wandb and args.max_steps is None:
        import wandb

        run = wandb.init(project=args.wandb, name=args.output.name, config=config)
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"{len(train_indices)} training frames, {len(val_indices)} validation frames, "
          f"{len(train_loader)} steps per epoch", flush=True)

    policy.train()
    step, skipped, stop, started = 0, 0, False, time.monotonic()
    epochs = int(train["epochs"])
    for epoch in range(epochs):
        epoch_metrics = []
        for batch in train_loader:
            factor = lr_factor(step)
            for group, base_lr in zip(optimizer.param_groups, base_lrs):
                group["lr"] = base_lr * factor
            batch = move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss, metrics = policy.compute_loss(batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, clip_norm)
            if not (torch.isfinite(loss.detach()) and torch.isfinite(grad_norm)):
                skipped += 1
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                step += 1
                continue
            scaler.step(optimizer)
            scaler.update()
            ema.step(policy)
            step += 1
            epoch_metrics.append(metrics)
            if step % 100 == 0:
                print(f"epoch {epoch} step {step} " + " ".join(f"{k} {v:.5f}" for k, v in metrics.items())
                      + f" grad {float(grad_norm):.3f} lr {optimizer.param_groups[0]['lr']:.2e}", flush=True)
                if run is not None:
                    run.log({**{f"train/{k}": v for k, v in metrics.items()}, "train/grad_norm": float(grad_norm),
                             "train/lr": optimizer.param_groups[0]["lr"], "epoch": epoch}, step=step)
            if args.max_steps is not None and step >= args.max_steps:
                stop = True
                break

        log = {f"epoch/{k}": v for k, v in mean_metrics(epoch_metrics).items()}
        if val_loader is not None:
            policy.eval()
            with torch.no_grad():
                log.update({f"val/{k}": v for k, v in mean_metrics(
                    [policy.compute_loss(move(b, device))[1] for b in val_loader]).items()})
            policy.train()
        for name, pose_batch in pose_batches.items():
            if pose_batch is not None:
                log.update({f"{name}/{k}": v for k, v in sampled_pose_error(ema.averaged_model, pose_batch).items()})
        print(f"epoch {epoch} ({time.monotonic() - started:.0f} s, {skipped} skipped steps) "
              + " ".join(f"{k} {v:.4f}" for k, v in log.items()), flush=True)
        if run is not None and log:
            run.log(log, step=step)
        if (epoch + 1) % int(train.get("checkpoint_every", 10)) == 0 or stop or epoch + 1 == epochs:
            save_policy(ema.averaged_model, config, args.output)
            print(f"saved {args.output} (epoch {epoch + 1})", flush=True)
        if stop:
            break
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
