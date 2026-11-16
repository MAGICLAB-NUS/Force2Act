"""The five Force2Act-Bench tasks: scene, conditions, success test and step budget.

Values are those of the paper's evaluation. Each condition is run for `trials` trials; trial k of condition c
uses seed = base_seed + c * trials + k - 1, so every policy sees the same trials.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Condition:
    name: str
    mass_kg: float | None = None        # object mass (block pivoting)
    incline_deg: float | None = None    # table incline (press-push box)
    offset_m: float | None = None       # target displacement in a seeded direction (insertion)


@dataclass(frozen=True)
class Task:
    name: str
    scene: str
    instruction: str
    camera: str                          # the policy camera
    conditions: tuple[Condition, ...]
    trials: int
    base_seed: int
    max_steps: int                       # control steps at 15 Hz
    success: dict[str, Any]
    object: str | None = None
    wrench_samples: int = 1              # contact wrench averaged over this many physics samples per step
    incline: dict[str, Any] | None = None
    target_objects: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)


FREQUENCY_HZ = 15.0
IMAGE_SIZE = (640, 360)

TASKS: dict[str, Task] = {
    "pivot_block": Task(
        name="pivot_block", scene="pivot_block", camera="external", object="flip_box",
        instruction="Push the phone-shaped block along the wall in Genesis",
        conditions=tuple(Condition(f"{m:.2f}kg", mass_kg=m) for m in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)),
        trials=25, base_seed=1050, max_steps=400,
        success={"type": "object_tilt", "target_degrees": 90.0, "tolerance_degrees": 2.0}),
    "push_box": Task(
        name="push_box", scene="push_box", camera="external", object="push_box", wrench_samples=4,
        instruction="Push the box to the red line by pressing on its top in Genesis",
        conditions=tuple(Condition(f"incline_{d}deg", incline_deg=float(d)) for d in (0, 2, 4)),
        trials=10, base_seed=1000, max_steps=400,
        incline={"bodies": ("table", "push_box", "start_stop", "finish_line"), "pivot": (0.48, 0.0, 0.05),
                 "axis": (0.0, -1.0, 0.0), "cameras": ("external",)},
        success={"type": "push", "frame": "table", "push_axis": (1.0, 0.0, 0.0), "min_push_mm": 155.0,
                 "top_margin_mm": 3.0, "max_tilt_degrees": 20.0}),
    "bottle_wiping": Task(
        name="bottle_wiping", scene="bottle_wiping", camera="overview", object="bottle", wrench_samples=4,
        instruction="Wipe the ink line off the water bottle with the sponge tool in Genesis",
        conditions=(Condition("nominal"),), trials=50, base_seed=1000, max_steps=400,
        success={"type": "wiping", "min_erased_fraction": 0.9, "max_object_shift_mm": 15.0}),
    "plug_insertion_round": Task(
        name="plug_insertion_round", scene="plug_insertion_round", camera="external",
        instruction="Insert the round peg into the round hole in Genesis",
        conditions=(Condition("demo", offset_m=0.0), Condition("offset_2mm", offset_m=0.002)),
        trials=25, base_seed=1000, max_steps=600, target_objects=("insertion_fixture",),
        success={"type": "insertion", "min_depth_mm": 30.0, "max_lateral_mm": 2.0}),
    "plug_insertion_usb": Task(
        name="plug_insertion_usb", scene="plug_insertion_usb", camera="overview",
        instruction="Insert the USB plug into the USB socket in Genesis",
        conditions=(Condition("demo", offset_m=0.0),), trials=25, base_seed=1000, max_steps=600,
        target_objects=("socket_mount", "usb_socket"),
        success={"type": "insertion", "min_depth_mm": 9.4, "max_lateral_mm": 1.0}),
}


def trials(task: Task, count: int | None = None) -> list[tuple[Condition, int, int]]:
    """(condition, trial number, seed) of every trial of the task, in order."""
    n = count or task.trials
    return [(condition, trial, task.base_seed + index * n + trial - 1)
            for index, condition in enumerate(task.conditions) for trial in range(1, n + 1)]
