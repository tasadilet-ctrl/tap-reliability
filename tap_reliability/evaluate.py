#!/usr/bin/env python3
"""Replicates the published TAP-Vid-DAVIS numbers for TAPIR and BootsTAPIR.

Targets from the upstream README (Jax checkpoints; the PyTorch ports are stated
to "closely match"):

    model       DAVIS First (AJ)   DAVIS Strided (AJ)
    TAPIR                  58.5%               63.3%
    BootsTAPIR             62.4%               67.4%

Metrics come from the benchmark's own compute_tapvid_metrics, vendored verbatim
(see vendor/). Visibility is derived exactly as upstream's demo does:

    visible = (1 - sigmoid(occlusion)) * (1 - sigmoid(expected_dist)) > 0.5

Per-video metrics, per-track metrics and the raw predictions are all written
out, so the confidence analysis never has to re-run the model.

    python -m tap_reliability.evaluate --model tapir --query-mode first
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import data, models
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vendor.tapvid_official_eval import compute_tapvid_metrics  # noqa: E402

PUBLISHED = {("tapir", "first"): 58.5, ("tapir", "strided"): 63.3,
             ("bootstapir", "first"): 62.4, ("bootstapir", "strided"): 67.4}


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def visible_from_logits(occlusion, expected_dist):
    """Upstream's postprocess_occlusions, unchanged."""
    return (1 - torch.sigmoid(occlusion)) * (1 - torch.sigmoid(expected_dist)) > 0.5


@torch.no_grad()
def track_one(model, sample, device, query_chunk_size=64):
    """Run the model on one video. Returns numpy tracks/occlusion/uncertainty."""
    # Cast before the device move: the vendored query sampler returns float64,
    # and MPS cannot hold float64 at all.
    video = torch.from_numpy(sample["video"]).float().to(device)
    queries = torch.from_numpy(sample["query_points"]).float().to(device)
    out = model(video=video, query_points=queries,
                query_chunk_size=query_chunk_size)
    return {k: out[k].cpu().numpy() for k in ("tracks", "occlusion", "expected_dist")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(models.CONFIGS), default="tapir")
    ap.add_argument("--query-mode", choices=("first", "strided"), default="first")
    ap.add_argument("--davis", default="data/tapvid_davis/tapvid_davis.pkl")
    ap.add_argument("--checkpoint-dir", default="checkpoints")
    ap.add_argument("--tapnet-root", default=None,
                    help="path to a tapnet checkout (for tapnet.torch.tapir_model)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--query-chunk-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="first N videos only")
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--save-predictions", action="store_true",
                    help="write per-video raw outputs for the confidence analysis")
    args = ap.parse_args()

    device = pick_device(args.device)
    out_dir = Path(args.out_dir or f"results/{args.model}_{args.query_mode}")
    out_dir.mkdir(parents=True, exist_ok=True)

    model = models.build(args.model, args.checkpoint_dir, device, args.tapnet_root)
    print(f"[setup] {args.model} on {device}, query_mode={args.query_mode}")
    print(f"[setup] checkpoint: {models.describe(args.model, args.checkpoint_dir)}")

    names = args.videos or data.video_names(args.davis)
    if args.limit:
        names = names[: args.limit]

    per_video, per_track_rows, t0 = {}, [], time.time()
    for i, sample in enumerate(data.iter_davis(args.davis, args.query_mode, names)):
        name = sample["video_name"]
        t1 = time.time()
        pred = track_one(model, sample, device, args.query_chunk_size)
        visible = visible_from_logits(torch.from_numpy(pred["occlusion"]),
                                      torch.from_numpy(pred["expected_dist"])).numpy()
        common = dict(query_points=sample["query_points"],
                      gt_occluded=sample["occluded"],
                      gt_tracks=sample["target_points"],
                      pred_occluded=~visible,
                      pred_tracks=pred["tracks"],
                      query_mode=args.query_mode)
        m = compute_tapvid_metrics(**common)
        per_video[name] = {k: float(np.mean(v)) for k, v in m.items()}
        tw = compute_tapvid_metrics(**common, get_trackwise_metrics=True)
        for j in range(sample["query_points"].shape[1]):
            per_track_rows.append(dict(video=name, track=j,
                                       **{k: float(np.asarray(v).reshape(-1)[j])
                                          for k, v in tw.items()}))
        if args.save_predictions:
            np.savez_compressed(
                out_dir / f"{name}.npz",
                tracks=pred["tracks"].astype(np.float32),
                occlusion=pred["occlusion"].astype(np.float32),
                expected_dist=pred["expected_dist"].astype(np.float32),
                query_points=sample["query_points"].astype(np.float32),
                gt_tracks=sample["target_points"].astype(np.float32),
                gt_occluded=sample["occluded"])
        print(f"  [{i + 1}/{len(names)}] {name:22s} "
              f"AJ={per_video[name]['average_jaccard'] * 100:5.1f} "
              f"<d={per_video[name]['average_pts_within_thresh'] * 100:5.1f} "
              f"OA={per_video[name]['occlusion_accuracy'] * 100:5.1f} "
              f"({time.time() - t1:.1f}s)", flush=True)

    keys = next(iter(per_video.values())).keys()
    overall = {k: float(np.mean([v[k] for v in per_video.values()])) for k in keys}
    summary = dict(model=args.model, query_mode=args.query_mode, device=device,
                   n_videos=len(per_video), seconds=round(time.time() - t0, 1),
                   overall=overall, per_video=per_video)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with open(out_dir / "per_track.json", "w") as f:
        json.dump(per_track_rows, f)

    aj = overall["average_jaccard"] * 100
    print(f"\n{args.model} / DAVIS {args.query_mode}: "
          f"AJ {aj:.1f}  <delta {overall['average_pts_within_thresh'] * 100:.1f}  "
          f"OA {overall['occlusion_accuracy'] * 100:.1f}   ({len(per_video)} videos)")
    ref = PUBLISHED.get((args.model, args.query_mode))
    if ref and len(per_video) == 30:
        print(f"published AJ {ref}  ->  difference {aj - ref:+.1f}")
    elif ref:
        print(f"published AJ {ref} is over all 30 videos; this ran {len(per_video)}, "
              f"so the two are not comparable")
    print(f"wrote {out_dir}/summary.json")


if __name__ == "__main__":
    main()
