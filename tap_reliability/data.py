"""TAP-Vid-DAVIS loader, mirroring upstream create_davis_dataset without TensorFlow.

Upstream's loader lives in tapnet/tapvid/evaluation_datasets.py and pulls in
tensorflow, tensorflow_datasets and mediapy. The only things it does that
matter for evaluation are reproduced here exactly:

  frames = frames.astype(np.float32) / 255.0 * 2.0 - 1.0
  target_points = points * [width, height]
  queries via sample_queries_first / sample_queries_strided  (vendored verbatim)

The resize matters and is kept identical. TAP-Vid-DAVIS ships at full
resolution (480x854, 480x910, 480x1152 across the 30 videos), not at 256x256,
so every video is resampled -- and upstream does it with mediapy, whose
resize_video uses a Lanczos filter. Swapping in PIL or OpenCV bilinear would
change the inputs, and therefore the metrics, without saying so. Order is
upstream's too: resize, then normalise, then scale the (0-1) ground-truth
points by the RESIZED width and height.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Iterator, Optional, Sequence

import mediapy as media
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vendor.tapvid_official_eval import (  # noqa: E402
    sample_queries_first, sample_queries_strided)

RESOLUTION = (256, 256)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def iter_davis(davis_pickle, query_mode: str = "first",
               videos: Optional[Sequence[str]] = None,
               resolution=RESOLUTION) -> Iterator[dict]:
    """Yield one dict per DAVIS video, in upstream's format.

    Keys: video [1,T,H,W,3] float32 in [-1,1], query_points [1,N,3] (t,y,x),
    target_points [1,N,T,2] (x,y), occluded [1,N,T] bool, plus 'video_name'.
    """
    if query_mode not in ("first", "strided"):
        raise ValueError(f"query_mode must be 'first' or 'strided', got {query_mode!r}")
    data = load_pickle(davis_pickle) if not isinstance(davis_pickle, dict) else davis_pickle
    names = list(data.keys()) if videos is None else list(videos)
    missing = [v for v in names if v not in data]
    if missing:
        raise KeyError(f"not in the pickle: {missing}")

    sampler = sample_queries_first if query_mode == "first" else sample_queries_strided
    for name in names:
        frames = data[name]["video"]
        if resolution is not None and tuple(frames.shape[1:3]) != tuple(resolution):
            frames = media.resize_video(frames, resolution)   # Lanczos, as upstream
        frames = frames.astype(np.float32) / 255.0 * 2.0 - 1.0
        target_points = data[name]["points"] * np.array(
            [frames.shape[2], frames.shape[1]])
        converted = sampler(data[name]["occluded"], target_points, frames)
        converted["video_name"] = name
        yield converted


def video_names(davis_pickle) -> list:
    return list(load_pickle(davis_pickle).keys())
