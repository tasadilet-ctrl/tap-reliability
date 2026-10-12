"""Builds the PyTorch TAPIR / BootsTAPIR models and loads their checkpoints.

The two differ in more than their weights, and the difference is not in the
PyTorch defaults -- it is recorded in the JAX configs upstream:

                  pyramid_level  extra_convs  softmax_temperature
    TAPIR                     0        False                 20.0
    BootsTAPIR                1         True                 10.0

(configs/tapir_config.py and configs/tapir_bootstrap_config.py). The PyTorch
class defaults to extra_convs=True and pyramid_level=1, i.e. to BootsTAPIR, so
building TAPIR with the defaults silently gives the wrong architecture. Rather
than trust the table, every checkpoint is loaded with strict=True and the
resulting key set is checked, so a mismatch is an error instead of a quietly
degraded score.
"""

from __future__ import annotations

import pathlib
from typing import Dict

import torch

# pyramid_level feeds the mixer's input width (dim + (pyramid_level + 2) * 49),
# so getting it wrong is a shape error, not a silent accuracy loss.
CONFIGS: Dict[str, dict] = {
    "tapir": dict(pyramid_level=0, extra_convs=False, softmax_temperature=20.0),
    "bootstapir": dict(pyramid_level=1, extra_convs=True, softmax_temperature=10.0),
}

CHECKPOINTS = {
    "tapir": "tapir_panning.pt",
    "bootstapir": "bootstapir_v2.pt",
}


def build(name: str, checkpoint_dir="checkpoints", device="cpu",
          tapnet_root: str | None = None):
    """Return an eval-mode TAPIR with its published weights loaded."""
    if name not in CONFIGS:
        raise ValueError(f"unknown model {name!r}; expected one of {list(CONFIGS)}")
    if tapnet_root:
        import sys
        if tapnet_root not in sys.path:
            sys.path.insert(0, tapnet_root)
    from tapnet.torch import tapir_model

    model = tapir_model.TAPIR(**CONFIGS[name])
    path = pathlib.Path(checkpoint_dir) / CHECKPOINTS[name]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Fetch it with scripts/download_assets.sh")
    state = torch.load(path, map_location="cpu", weights_only=True)
    # strict=True: a wrong pyramid_level or extra_convs shows up here rather
    # than as a mysteriously low AJ.
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def describe(name: str, checkpoint_dir="checkpoints") -> dict:
    """Checkpoint facts without building anything: key count, params, extras."""
    path = pathlib.Path(checkpoint_dir) / CHECKPOINTS[name]
    state = torch.load(path, map_location="cpu", weights_only=True)
    mixer_in = next((v.shape for k, v in state.items()
                     if "torch_pips_mixer" in k and v.ndim == 2), None)
    return {
        "keys": len(state),
        "params": sum(v.numel() for v in state.values()),
        "has_extra_convs": any(k.startswith("extra_convs") for k in state),
        "mixer_first_weight": tuple(mixer_in) if mixer_in else None,
    }
