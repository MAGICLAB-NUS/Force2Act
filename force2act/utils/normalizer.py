# Adapted from ManiFlow (https://github.com/geyan21/ManiFlow_Policy), MIT License, Copyright (c) 2025 Ge Yan.
# See MANIFLOW_LICENSE in this folder. Changes: no zarr dependency; continuous roll/pitch/yaw fold for "pose".
"""Per-field linear normalizers, stored in the policy's state dict."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from force2act.utils.rotation import RPY_CENTER_KEY, rpy_to_continuous


def dict_apply(x: dict, func: Callable) -> dict:
    result = {}
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        elif isinstance(value, list):
            result[key] = [func(item) if hasattr(item, "to") else item for item in value]
        else:
            result[key] = func(value)
    return result


class DictOfTensorMixin(nn.Module):
    def __init__(self, params_dict: nn.ParameterDict | None = None) -> None:
        super().__init__()
        self.params_dict = nn.ParameterDict() if params_dict is None else params_dict

    @property
    def device(self) -> torch.device:
        return next(iter(self.parameters())).device

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        def dfs_add(dest, keys, value):
            if len(keys) == 1:
                dest[keys[0]] = value
                return
            if keys[0] not in dest:
                dest[keys[0]] = nn.ParameterDict()
            dfs_add(dest[keys[0]], keys[1:], value)

        out = nn.ParameterDict()
        start = prefix + "params_dict"
        for key, value in state_dict.items():
            if key.startswith(start):
                dfs_add(out, key[len(start):].split(".")[1:], value.clone())
        self.params_dict = out
        self.params_dict.requires_grad_(False)


def _fit(data, last_n_dims: int = 1, dtype=torch.float32, mode: str = "limits", output_max: float = 1.0,
         output_min: float = -1.0, range_eps: float = 1e-4, fit_offset: bool = True) -> nn.ParameterDict:
    assert mode in ("limits", "gaussian")
    assert last_n_dims >= 0
    assert output_max > output_min
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    if dtype is not None:
        data = data.type(dtype)
    dim = 1
    if last_n_dims > 0:
        dim = np.prod(data.shape[-last_n_dims:])
    data = data.reshape(-1, dim)

    input_min, _ = data.min(axis=0)
    input_max, _ = data.max(axis=0)
    input_mean = data.mean(axis=0)
    input_std = data.std(axis=0)

    if mode == "limits":
        if fit_offset:
            input_range = input_max - input_min
            ignore_dim = input_range < range_eps
            input_range[ignore_dim] = output_max - output_min
            scale = (output_max - output_min) / input_range
            offset = output_min - scale * input_min
            offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]
        else:
            assert output_max > 0
            assert output_min < 0
            output_abs = min(abs(output_min), abs(output_max))
            input_abs = torch.maximum(torch.abs(input_min), torch.abs(input_max))
            ignore_dim = input_abs < range_eps
            input_abs[ignore_dim] = output_abs
            scale = output_abs / input_abs
            offset = torch.zeros_like(input_mean)
    else:
        ignore_dim = input_std < range_eps
        scale = input_std.clone()
        scale[ignore_dim] = 1
        scale = 1 / scale
        offset = -input_mean * scale if fit_offset else torch.zeros_like(input_mean)

    params = nn.ParameterDict({
        "scale": scale,
        "offset": offset,
        "input_stats": nn.ParameterDict({"min": input_min, "max": input_max, "mean": input_mean, "std": input_std}),
    })
    for p in params.parameters():
        p.requires_grad_(False)
    return params


def _normalize(x, params, forward: bool = True) -> torch.Tensor:
    assert "scale" in params
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    scale, offset = params["scale"], params["offset"]
    x = x.to(device=scale.device, dtype=scale.dtype)
    shape = x.shape
    x = x.reshape(-1, scale.shape[0])
    x = x * scale + offset if forward else (x - offset) / scale
    return x.reshape(shape)


class SingleFieldLinearNormalizer(DictOfTensorMixin):
    @torch.no_grad()
    def fit(self, data, **kwargs) -> None:
        self.params_dict = _fit(data, **kwargs)

    @classmethod
    def create_fit(cls, data, **kwargs) -> "SingleFieldLinearNormalizer":
        obj = cls()
        obj.fit(data, **kwargs)
        return obj

    @classmethod
    def create_manual(cls, scale, offset, input_stats_dict: dict) -> "SingleFieldLinearNormalizer":
        def to_tensor(x):
            if not isinstance(x, torch.Tensor):
                x = torch.from_numpy(x)
            return x.flatten()

        for x in [offset] + list(input_stats_dict.values()):
            assert x.shape == scale.shape
            assert x.dtype == scale.dtype
        params = nn.ParameterDict({
            "scale": to_tensor(scale),
            "offset": to_tensor(offset),
            "input_stats": nn.ParameterDict(dict_apply(input_stats_dict, to_tensor)),
        })
        return cls(params)

    @classmethod
    def create_identity(cls, dtype=torch.float32) -> "SingleFieldLinearNormalizer":
        return cls.create_manual(
            torch.tensor([1], dtype=dtype), torch.tensor([0], dtype=dtype),
            {"min": torch.tensor([-1], dtype=dtype), "max": torch.tensor([1], dtype=dtype),
             "mean": torch.tensor([0], dtype=dtype), "std": torch.tensor([1], dtype=dtype)})

    def normalize(self, x) -> torch.Tensor:
        return _normalize(x, self.params_dict, forward=True)

    def unnormalize(self, x) -> torch.Tensor:
        return _normalize(x, self.params_dict, forward=False)

    def __call__(self, x) -> torch.Tensor:
        return self.normalize(x)


class LinearNormalizer(DictOfTensorMixin):
    """One normalizer per observation / action field. Before normalizing, raw [-pi, pi) roll/pitch/yaw of
    "pose" are folded into the continuous window fitted at training time (rotation.install_rpy_center)."""

    def __getitem__(self, key: str) -> SingleFieldLinearNormalizer:
        return SingleFieldLinearNormalizer(self.params_dict[key])

    def __setitem__(self, key: str, value: SingleFieldLinearNormalizer) -> None:
        self.params_dict[key] = value.params_dict

    def __call__(self, x):
        return self.normalize(x)

    def _normalize_impl(self, x, forward: bool = True):
        if (forward and isinstance(x, dict) and "pose" in x
                and "pose" in self.params_dict and RPY_CENTER_KEY in self.params_dict):
            x = {**x, "pose": rpy_to_continuous(torch.as_tensor(x["pose"]), self.params_dict[RPY_CENTER_KEY])}
        if isinstance(x, dict):
            return {key: (_normalize(value, self.params_dict[key], forward=forward) if key in self.params_dict
                          else value) for key, value in x.items()}
        if "_default" not in self.params_dict:
            raise RuntimeError("normalizer not initialized")
        return _normalize(x, self.params_dict["_default"], forward=forward)

    def normalize(self, x):
        return self._normalize_impl(x, forward=True)

    def unnormalize(self, x):
        return self._normalize_impl(x, forward=False)
