"""Evaluate a policy on a Force2Act-Bench task.

    force2act-bench --task pivot_block --checkpoint runs/pivot_block
    force2act-bench --config configs/bench/pivot_block.yaml
    force2act-bench --task push_box --control impedance --policy my_package.policy:make_policy

Trials run in parallel worker processes (one simulator and one policy each). Results go to
<output>/results.jsonl (one line per trial) and <output>/summary.md; --video also writes one MP4 per trial.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from force2act_bench.tasks import FREQUENCY_HZ, TASKS, trials

DEFAULTS = {"policy": "force2act", "control": "hybrid", "workers": 8, "device": "cuda", "first_core": 0,
            "video": False, "video_every": 2}


class VideoRecorder:
    """One MP4 per trial from the policy camera (every `every`-th step)."""

    def __init__(self, output: Path, camera: str, every: int) -> None:
        self.output, self.camera, self.every = output, camera, max(1, int(every))
        self.writer, self.partial = None, None

    def wants(self, step: int) -> bool:
        return step % self.every == 0

    def start(self, name: str) -> None:
        self.name, self.partial, self.writer = name, self.output / f".{name}.partial.mp4", None

    def add(self, step: int, observation: dict[str, Any], label: str) -> None:
        image = observation.get("rgb", {}).get(self.camera)
        if image is None or not self.wants(step):
            return
        import cv2

        frame = np.ascontiguousarray(np.array(image, dtype=np.uint8, copy=True)[..., :3])
        cv2.putText(frame, f"step {step}  {label}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                    cv2.LINE_AA)
        if self.writer is None:
            import imageio.v2 as imageio

            self.writer = imageio.get_writer(self.partial, format="FFMPEG", fps=FREQUENCY_HZ / self.every,
                                             codec="libx264", quality=8, macro_block_size=1,
                                             ffmpeg_log_level="error")
        self.writer.append_data(frame)

    def finish(self, outcome: str) -> None:
        if self.writer is not None:
            self.writer.close()
            self.partial.replace(self.output / f"{self.name}_{outcome}.mp4")
        self.writer = None


def run_trial(env, policy, condition, trial: int, seed: int, recorder: VideoRecorder | None) -> dict[str, Any]:
    from force2act_bench.env import TrialResult

    name = f"{condition.name}_trial{trial:02d}"
    if recorder is not None:
        recorder.start(name)
    try:
        policy.reset(seed)
        observation = env.reset(condition, seed, trial)
        if recorder is not None:
            recorder.add(0, observation, "start")
        while not env.done:
            action = policy.act(observation)
            need_image = policy.needs_image()
            record = recorder is not None and recorder.wants(env.index + 2)
            observation = env.step(action, images=need_image or record)
            if recorder is not None:
                recorder.add(env.index + 1, observation, env.task.success["type"])
            if not need_image:
                observation = {key: value for key, value in observation.items() if key != "rgb"}
        result = env.result()
    except Exception as error:  # a trial that crashes counts as a failure
        traceback.print_exc()
        env.error = f"{type(error).__name__}: {error}"
        try:
            result = env.result()
        except Exception:
            result = TrialResult(task=env.task.name, condition=condition.name, trial=trial, seed=seed,
                                   success=False, success_time_s=None, steps=0, error=env.error)
    outcome = "ERROR" if result.error else ("PASS" if result.success else "FAIL")
    if recorder is not None:
        recorder.finish(outcome)
    return {**result.to_dict(), "outcome": outcome}


def _gl_renderer() -> str | None:
    try:
        from OpenGL import GL

        value = GL.glGetString(GL.GL_RENDERER)
        return value.decode() if value else None
    except Exception:
        return None


def _worker(index: int, settings: dict[str, Any], jobs: mp.Queue, messages: mp.Queue) -> None:
    try:
        if settings.get("pin_cores"):
            cores = sorted(os.sched_getaffinity(0))
            os.sched_setaffinity(0, {cores[(settings["first_core"] + index) % len(cores)]})
        from force2act_bench.env import BenchEnv
        from force2act_bench.policies import load_policy

        task = TASKS[settings["task"]]
        env = BenchEnv(task, control=settings["control"], images=True, gpu=False, seed=task.base_seed + index)
        renderer = _gl_renderer()
        if renderer and any(name in renderer.lower() for name in ("llvmpipe", "softpipe", "swrast")):
            raise RuntimeError(f"camera rendering fell back to software ({renderer}); check the GPU's EGL setup")
        kwargs = dict(settings.get("policy_args") or {})
        if settings["policy"] == "force2act":
            kwargs.setdefault("checkpoint", settings["checkpoint"])
            kwargs.setdefault("camera", task.camera)
            kwargs.setdefault("device", settings["device"])
            if settings.get("mofe"):
                kwargs.setdefault("mofe", settings["mofe"])
        policy = load_policy(settings["policy"], **kwargs)
        recorder = VideoRecorder(Path(settings["output"]), task.camera, settings["video_every"]) \
            if settings["video"] else None
        messages.put(("ready", index, None))
        while True:
            job = jobs.get()
            if job is None:
                break
            condition_name, trial, seed = job
            condition = next(c for c in task.conditions if c.name == condition_name)
            messages.put(("result", index, run_trial(env, policy, condition, trial, seed, recorder)))
        env.close()
    except BaseException as error:
        messages.put(("fatal", index, f"{type(error).__name__}: {error}\n{traceback.format_exc()}"))


def wilson(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total ** 2)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def summarize(task, results: list[dict[str, Any]]) -> str:
    lines = [f"# {task.name}", "", "| Condition | Success | Rate | 95% CI | Errors |", "|---|---|---|---|---|"]
    total = passed = 0
    for condition in task.conditions:
        rows = [r for r in results if r["condition"] == condition.name]
        if not rows:
            continue
        wins = sum(r["success"] for r in rows)
        low, high = wilson(wins, len(rows))
        lines.append(f"| {condition.name} | {wins}/{len(rows)} | {100 * wins / len(rows):.1f}% | "
                     f"{100 * low:.0f}-{100 * high:.0f}% | {sum(r['error'] is not None for r in rows)} |")
        total, passed = total + len(rows), passed + wins
    lines.append(f"| **all** | **{passed}/{total}** | **{100 * passed / max(total, 1):.1f}%** | | |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, help="YAML with any of the options below")
    parser.add_argument("--task", choices=sorted(TASKS))
    parser.add_argument("--policy", help="force2act (default) or module:factory")
    parser.add_argument("--checkpoint", help="Force2Act checkpoint: directory or hf://<owner>/<repo>/<task> "
                        "(default: the released model of the task)")
    parser.add_argument("--mofe", help="Force2Act MoFE (default: the one named in the checkpoint config)")
    parser.add_argument("--control", help="position, impedance, hybrid (default) or a controller YAML")
    parser.add_argument("--trials", type=int, help="trials per condition (default: the paper protocol)")
    parser.add_argument("--conditions", nargs="+", help="run only these conditions")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--device", help="policy device (default cuda)")
    parser.add_argument("--pin-cores", action="store_true", help="one CPU core per worker, from --first-core")
    parser.add_argument("--first-core", type=int)
    parser.add_argument("--video", action="store_true", default=None, help="write one MP4 per trial")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    settings = dict(DEFAULTS)
    if args.config:
        settings.update(yaml.safe_load(args.config.read_text()) or {})
    settings.update({key: value for key, value in vars(args).items() if value is not None and key != "config"})
    if "task" not in settings:
        parser.error("--task (or task: in --config) is required")
    task = TASKS[settings["task"]]
    if settings["policy"] == "force2act":
        from force2act import HUB_MODELS

        settings.setdefault("checkpoint", f"hf://{HUB_MODELS}/{task.name}")
    output = Path(settings.get("output") or Path("results") / task.name)
    output.mkdir(parents=True, exist_ok=True)
    settings["output"] = str(output)
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    jobs_list = [(c.name, trial, seed) for c, trial, seed in trials(task, settings.get("trials"))
                 if not settings.get("conditions") or c.name in settings["conditions"]]
    workers = max(1, min(int(settings["workers"]), len(jobs_list)))
    print(f"{task.name}: {len(jobs_list)} trials, {workers} workers, control {settings['control']}, "
          f"policy {settings['policy']} {settings.get('checkpoint', '')}", flush=True)
    context = mp.get_context("spawn")
    jobs, messages = context.Queue(), context.Queue()
    processes = [context.Process(target=_worker, args=(i, settings, jobs, messages), daemon=True)
                 for i in range(workers)]
    for process in processes:
        process.start()
    results: list[dict[str, Any]] = []
    started = time.monotonic()
    try:
        ready = 0
        while ready < workers:
            kind, index, payload = messages.get()
            if kind == "fatal":
                raise RuntimeError(f"worker {index} failed:\n{payload}")
            ready += 1
        for job in jobs_list:
            jobs.put(job)
        for _ in processes:
            jobs.put(None)
        with open(output / "results.jsonl", "w") as log:
            while len(results) < len(jobs_list):
                kind, index, payload = messages.get()
                if kind == "fatal":
                    raise RuntimeError(f"worker {index} failed:\n{payload}")
                results.append(payload)
                log.write(json.dumps(payload) + "\n")
                log.flush()
                print(f"[{len(results)}/{len(jobs_list)}] {payload['condition']} trial {payload['trial']:02d}: "
                      f"{payload['outcome']} ({time.monotonic() - started:.0f} s)", flush=True)
    finally:
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
    summary = summarize(task, results)
    (output / "summary.md").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
