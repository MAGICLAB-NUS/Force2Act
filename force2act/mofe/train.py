"""Train MoFE: one model for all tasks, without task labels.

    python -m force2act.mofe.train --configs configs/policy/*.yaml --output runs/mofe

Every contact frame of every demonstration is a sample; its target is the demonstrated normal force over the next
chunk (the maximum over 8 steps of the 5-step smoothed normal force). The expert is sampled with a
straight-through Gumbel-softmax (temperature 1 -> 0.1); a level penalty lam (N^2) is added when a constant-level
expert (pushing or sliding) is chosen where a sticking expert is admissible. The router's |Ft| inputs are scaled
by a random factor in [e^-0.7, e^0.7] per sample, so that routing does not depend on the object's weight.

A held-out report (20% of each task's episodes) is computed first; the saved model is then trained on all
contact frames. The tasks are concatenated in a fixed order (TASK_ORDER), so the result does not depend on the
order of --configs.

Ablations: --no-gates, --router-sees-direction, --one-per-case, --offset-bound 60, --lam 0.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from force2act.control.contact import contact_split
from force2act.mofe.gates import EXPERTS, physical_inputs, router_features, tool_z_axis
from force2act.mofe.law import ExpertSelector, save_mofe
from force2act.mofe.model import CONSTANTS, MoFE, law_text

# Order in which the tasks' frames are concatenated (it sets the minibatches); other tasks follow in --configs order.
TASK_ORDER = ("pivot_block", "push_box", "bottle_wiping", "plug_insertion_round", "plug_insertion_usb")


# ---------------------------------------------------------------------------------------------------- data
def step_targets(wrench: np.ndarray, velocity: np.ndarray, episodes: np.ndarray, min_speed: float,
                 push_angle: float, contact_force: float, history: int, chunk: int, window: int = 5):
    """Per frame: router features of the wrench history, the target normal force (max over the next `chunk`
    frames of the `window`-smoothed normal force) and whether the next frame is in contact."""
    tangential, normal = contact_split(wrench[:, :3], velocity, min_speed, push_angle)
    in_contact = np.linalg.norm(wrench[:, :3], axis=1) >= contact_force
    n = len(wrench)
    smooth, past, following = np.zeros(n), np.zeros((n, history), dtype=int), np.zeros((n, chunk), dtype=int)
    for episode in np.unique(episodes):
        rows = np.flatnonzero(episodes == episode)
        half = window // 2
        padded = np.pad(normal[rows], (half, window - 1 - half), mode="edge")
        smooth[rows] = np.convolve(padded, np.ones(window) / window, mode="valid")
        local = np.arange(len(rows))
        past[rows] = rows[np.clip(local[:, None] + np.arange(1 - history, 1)[None], 0, len(rows) - 1)]
        following[rows] = rows[np.minimum(local[:, None] + np.arange(1, chunk + 1)[None], len(rows) - 1)]
    features = router_features(wrench[past], velocity[past], min_speed, push_angle, include_torque=True)
    return features, smooth[following].max(axis=1)[:, None].astype(np.float32), in_contact[following[:, 0]]


def frame_cache(dataset, image_key: str, size: int, path: Path, workers: int) -> np.ndarray:
    """Every camera frame resized to size x size (bilinear, antialiased), uint8 [N, 3, size, size], cached."""
    if path.exists():
        return np.load(path, mmap_mode="r")

    class Decode(torch.utils.data.Dataset):
        def __len__(self):
            return len(dataset)

        def __getitem__(self, row):
            image = torch.as_tensor(dataset[row][image_key], dtype=torch.float32)[None]
            image = F.interpolate(image, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
            return (image[0].clamp(0, 1) * 255).round().to(torch.uint8)

    path.parent.mkdir(parents=True, exist_ok=True)
    loader = torch.utils.data.DataLoader(Decode(), batch_size=64, num_workers=workers)
    frames = torch.cat(list(loader)).numpy()
    if len(frames) != len(dataset):
        raise RuntimeError("frame cache incomplete")
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, frames)
    tmp.rename(path)
    return frames


def load_task(config: dict, cache_dir: Path, workers: int, seed: int, gates: bool) -> dict:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    data, control = config["data"], config["inference"]["control"]
    min_speed, push_angle = float(control["min_speed"]), float(control["push_angle"])
    history = int(data["wrench_steps"])
    dataset = LeRobotDataset(repo_id=data.get("repo_id", "local/force2act"), root=data["root"], video_backend="pyav")
    hf = dataset.hf_dataset
    wrench = np.stack(hf[data["wrench_key"]]).astype(np.float64)[:, :6]
    velocity = np.stack(hf[data["velocity_key"]]).astype(np.float64)[:, :3]
    pose = np.stack(hf[data["tool_pose_key"]]).astype(np.float64)
    episodes = np.asarray(hf["episode_index"])
    features, target, contact = step_targets(wrench, velocity, episodes, min_speed, push_angle,
                                             float(control["contact_force"]), history,
                                             int(config["policy"]["action_steps"]))
    x = np.zeros((len(wrench), 2 + len(EXPERTS)), dtype=np.float32)
    tool_z = tool_z_axis(pose[:, -4:])
    for e in np.unique(episodes):
        rows = np.flatnonzero(episodes == e)
        x[rows] = physical_inputs(wrench[rows, :3], velocity[rows], tool_z[rows], min_speed, push_angle, gates)
    _, normal_now = contact_split(wrench[:, :3], velocity, min_speed, push_angle)
    images = frame_cache(dataset, data["image_key"], 224, cache_dir / f"{config['task']}_224.npy", workers)
    unique = np.random.default_rng(seed).permutation(np.unique(episodes))
    held = np.isin(episodes, unique[:max(1, int(round(0.2 * len(unique))))])
    return dict(name=config["task"], images=images, features=features, x=x, target=target, normal_now=normal_now,
                contact=contact, held=held, episodes=episodes, max_force=float(control["f_max"]),
                min_speed=min_speed, push_angle=push_angle, history=history)


def augment(image: torch.Tensor) -> torch.Tensor:
    """Random resized crop (area 0.8-1, aspect 0.9-1.1) and colour jitter (+-20%), per sample."""
    b = len(image)
    area = torch.empty(b, device=image.device).uniform_(0.8, 1.0)
    aspect = torch.empty(b, device=image.device).uniform_(0.9, 1.1)
    sx, sy = (area * aspect).sqrt().clamp(max=1), (area / aspect).sqrt().clamp(max=1)
    tx = (1 - sx) * torch.empty(b, device=image.device).uniform_(-1, 1)
    ty = (1 - sy) * torch.empty(b, device=image.device).uniform_(-1, 1)
    theta = torch.zeros(b, 2, 3, device=image.device)
    theta[:, 0, 0], theta[:, 1, 1], theta[:, 0, 2], theta[:, 1, 2] = sx, sy, tx, ty
    grid = F.affine_grid(theta, image.shape, align_corners=False)
    image = F.grid_sample(image, grid, mode="bilinear", padding_mode="border", align_corners=False)
    jitter = lambda: torch.empty(b, 1, 1, 1, device=image.device).uniform_(0.8, 1.2)  # noqa: E731
    image = image * jitter()
    mean = image.mean(dim=(1, 2, 3), keepdim=True)
    image = (image - mean) * jitter() + mean
    gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
    image = (image - gray) * jitter() + gray
    return image.clamp(0, 1)


# ------------------------------------------------------------------------------------------------- report
def hysteresis(probs: np.ndarray, margin: float, hold: int) -> np.ndarray:
    selector = ExpertSelector(margin, hold)
    return np.array([selector(p) for p in probs], dtype=int)


def evaluate(model, tensors, rows, device, task_of_row, names, episode, margin, hold) -> dict:
    images, features, x, target = tensors
    model.eval()
    preds, outs, probs = [], [], []
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=device.type == "cuda"):
        for s in range(0, len(rows), 1024):
            idx = torch.as_tensor(rows[s:s + 1024], device=device)
            p, _, o, q = model(images[idx].float() / 255.0, features[idx], x[idx])
            preds.append(p.float().cpu().numpy()[:, 0])
            outs.append(o.float().cpu().numpy())
            probs.append(q.float().cpu().numpy())
    pred, out, prob = (np.concatenate(v) for v in (preds, outs, probs))
    ep = episode[rows]
    held_pick = np.empty(len(rows), dtype=int)
    for e in np.unique(ep):
        held_pick[ep == e] = hysteresis(prob[ep == e], margin, hold)
    held_pred = out[np.arange(len(out)), held_pick]
    truth = target[rows].cpu().numpy()[:, 0]
    report = {}
    for t, name in enumerate(names):
        mine = task_of_row[rows] == t
        rmse = lambda p: float(np.sqrt(np.mean((p[mine] - truth[mine]) ** 2)))  # noqa: E731
        report[name] = {"rmse_contact_N": rmse(pred), "rmse_hysteresis_N": rmse(held_pred),
                        "constant_rmse_contact_N": float(np.std(truth[mine])),
                        "expert_share_pct": {EXPERTS[e]: round(100 * float(np.mean(held_pick[mine] == e)), 1)
                                             for e in range(len(EXPERTS))}}
    return report


# --------------------------------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--configs", type=Path, nargs="+", required=True, help="policy configs of the tasks")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--constant-lr", type=float, default=2e-2)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--offset-bound", type=float, default=2.0, help="bound on the sticking offsets (N)")
    parser.add_argument("--lam", type=float, default=20.0, help="level penalty (N^2)")
    parser.add_argument("--force-scale-aug", type=float, default=0.7, help="log-range of the router's |Ft| scaling")
    parser.add_argument("--tau-start", type=float, default=1.0)
    parser.add_argument("--tau-end", type=float, default=0.1)
    parser.add_argument("--jitter", type=float, default=0.01, help="router feature jitter")
    parser.add_argument("--hysteresis-margin", type=float, default=0.2)
    parser.add_argument("--hysteresis-hold", type=int, default=3)
    parser.add_argument("--no-gates", action="store_true", help="ablation: every expert admissible")
    parser.add_argument("--router-sees-direction", action="store_true", help="ablation: unmask the force direction")
    parser.add_argument("--one-per-case", action="store_true", help="ablation: one sticking law, one sliding level")
    parser.add_argument("--no-report", action="store_true", help="skip the held-out report")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--cache-dir", type=Path, default=Path("cache/frames"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    configs = [yaml.safe_load(path.read_text()) for path in args.configs]
    configs.sort(key=lambda c: TASK_ORDER.index(c["task"]) if c["task"] in TASK_ORDER else len(TASK_ORDER))
    tasks =[load_task(c, args.cache_dir, args.num_workers, args.seed, not args.no_gates) for c in configs]
    names = [t["name"] for t in tasks]
    feature_dim = tasks[0]["features"].shape[1]
    history = tasks[0]["history"]
    if any(t["features"].shape[1] != feature_dim or t["history"] != history for t in tasks):
        raise ValueError("all tasks need the same wrench history")
    cat = lambda key: np.concatenate([t[key] for t in tasks])  # noqa: E731
    task_of_row = np.concatenate([np.full(len(t["target"]), i) for i, t in enumerate(tasks)])
    held, contact, x, target = cat("held"), cat("contact"), cat("x"), cat("target")
    episode = np.concatenate([i * 1_000_000 + t["episodes"].astype(np.int64) for i, t in enumerate(tasks)])
    rows = np.arange(len(task_of_row))
    images = torch.from_numpy(np.concatenate([np.ascontiguousarray(t["images"]) for t in tasks])).to(device)
    features_np = cat("features")
    features, xs, targets = (torch.from_numpy(a).to(device) for a in (features_np, x, target))
    max_force = max(t["max_force"] for t in tasks)

    def train(train_mask: np.ndarray, label: str) -> MoFE:
        torch.manual_seed(args.seed)
        model = MoFE(feature_dim, history, offset_bound=args.offset_bound, max_force=max_force,
                     router_sees_direction=args.router_sees_direction, one_per_case=args.one_per_case)
        tr = np.flatnonzero(train_mask)
        level = float(np.median(target[tr, 0]))
        t_med = float(np.median(x[tr][x[tr, 0] > 0.5, 0])) if (x[tr, 0] > 0.5).any() else 1.0
        p_med = float(np.median(x[tr][x[tr, 1] > 0.5, 1])) if (x[tr, 1] > 0.5).any() else 1.0
        lo, mid, hi = (float(np.quantile(target[tr, 0], q)) for q in (0.2, 0.5, 0.8))
        init = {"pushing_level": level, "sticking_1_slope": level / max(t_med, 0.1), "sticking_1_offset": 1.0,
                "sticking_2_slope": level / max(t_med, 0.1), "sticking_2_offset": 1.0,
                "sliding_level_1": max(mid if args.one_per_case else lo, 0.5), "sliding_level_2": max(mid, 0.5),
                "sliding_level_3": max(hi, 0.5), "insertion_slope": 0.5 * level / max(p_med, 0.1),
                "insertion_offset": 0.5 * level}
        with torch.no_grad():
            model.init.copy_(torch.tensor([init[n] for n in CONSTANTS]))
            model.feature_mean.copy_(torch.from_numpy(features_np[tr].mean(0)))
            model.feature_std.copy_(torch.from_numpy(features_np[tr].std(0)).clamp_min(1e-3))
        model.to(device)
        optimizer = torch.optim.AdamW([
            {"params": [p for n, p in model.named_parameters() if not n.startswith("raw.")],
             "lr": args.lr, "weight_decay": args.weight_decay},
            {"params": list(model.raw.parameters()), "lr": args.constant_lr, "weight_decay": 0.0}])
        steps = len(tr) // args.batch_size
        total_steps, step = steps * args.epochs, 0
        generator = torch.Generator(device=device).manual_seed(args.seed)
        order_rows = torch.as_tensor(tr, device=device)
        level_index = [i for i, e in enumerate(EXPERTS) if e.startswith("sliding") or e == "pushing"]
        sticking = [EXPERTS.index("sticking_1"), EXPERTS.index("sticking_2")]
        per_step = feature_dim // history
        scale_mask = torch.zeros(feature_dim, device=device)
        for i in range(history):
            scale_mask[i * per_step] = 1.0     # |Ft| only
        scale_mask = scale_mask[None]
        for epoch in range(args.epochs):
            start, total = time.time(), 0.0
            model.train()
            order = order_rows[torch.randperm(len(order_rows), device=device, generator=generator)]
            for i in range(steps):
                tau = args.tau_start * (args.tau_end / args.tau_start) ** (step / max(total_steps - 1, 1))
                idx = order[i * args.batch_size:(i + 1) * args.batch_size]
                feat = features[idx]
                if args.force_scale_aug > 0:
                    scale = torch.exp(torch.empty(len(idx), 1, device=device).uniform_(-args.force_scale_aug,
                                                                                        args.force_scale_aug))
                    feat = feat * scale_mask * scale + feat * (1 - scale_mask)
                with torch.autocast("cuda", torch.bfloat16, enabled=device.type == "cuda"):
                    logits = model.logits(augment(images[idx].float() / 255.0), feat, xs[idx], args.jitter)
                    choice = F.gumbel_softmax(logits, tau=tau, hard=True)
                    pred = (choice * model.expert_outputs(xs[idx].float())).sum(-1, keepdim=True)
                loss = ((pred.float() - targets[idx]) ** 2).mean()
                mse = float(loss)
                if args.lam > 0:
                    sticking_admissible = xs[idx][:, 2 + sticking[0]] + xs[idx][:, 2 + sticking[1]] > 0.5
                    loss = loss + args.lam * (choice[:, level_index].sum(-1) * sticking_admissible.float()).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total, step = total + mse, step + 1
            print(f"[{label}] epoch {epoch + 1}/{args.epochs} ({time.time() - start:.0f} s) tau {tau:.2f} "
                  f"train RMSE {(total / max(steps, 1)) ** 0.5:.2f} N | "
                  + " | ".join(f"{e}: {t}" for e, t in law_text(model.constants()).items()), flush=True)
        return model.eval()

    report = None
    if not args.no_report:
        model = train(contact & ~held, "held-out split")
        report = evaluate(model, (images, features, xs, targets), rows[contact & held], device, task_of_row, names,
                          episode, args.hysteresis_margin, args.hysteresis_hold)
        for name, r in report.items():
            print(f"{name}: held-out contact RMSE {r['rmse_contact_N']:.2f} N (constant {r['constant_rmse_contact_N']:.2f}) "
                  f"| experts {r['expert_share_pct']}", flush=True)
    model = train(contact, "all frames")
    t0 = tasks[0]
    config = {"feature_dim": int(feature_dim), "history": int(history), "include_torque": True, "image_size": 224,
              "min_speed": t0["min_speed"], "push_angle": t0["push_angle"], "offset_bound": args.offset_bound,
              "max_force": max_force, "gates": not args.no_gates, "router_sees_direction": args.router_sees_direction,
              "one_per_case": args.one_per_case, "tasks": names, "laws": law_text(model.constants())}
    save_mofe(model.cpu(), config, args.output)
    if report is not None:
        (args.output / "heldout.json").write_text(json.dumps(report, indent=2))
    print(f"saved {args.output}: {config['laws']}")


if __name__ == "__main__":
    main()
