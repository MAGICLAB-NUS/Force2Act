"""Policies evaluated by Force2Act-Bench.

A policy is any object with

    reset(seed: int) -> None          start an episode
    act(observation: dict) -> Action  one control step (15 Hz)
    needs_image() -> bool             whether the next observation must include camera frames

`force2act` is the Force2Act policy; `load_policy("module:factory")` loads your own.
"""

from __future__ import annotations

import importlib
from typing import Any, Protocol

from force2act_bench.sim import Action


class Policy(Protocol):
    def reset(self, seed: int) -> None: ...

    def act(self, observation: dict[str, Any]) -> Action: ...

    def needs_image(self) -> bool: ...


def load_policy(name: str, **kwargs) -> Policy:
    """"force2act" or "package.module:factory" (called with the keyword arguments)."""
    if name == "force2act":
        from force2act_bench.policies.force2act import Force2ActPolicy

        return Force2ActPolicy(**kwargs)
    module, _, factory = name.partition(":")
    if not factory:
        raise ValueError(f"policy must be 'force2act' or 'module:factory', got {name!r}")
    return getattr(importlib.import_module(module), factory)(**kwargs)
