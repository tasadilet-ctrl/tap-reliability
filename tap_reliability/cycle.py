#!/usr/bin/env python3
"""Forward-backward cycle consistency as a training-free uncertainty baseline.

confidence.py showed TAPIR's `expected_dist` head is a strong error predictor
(AUROC 0.93-0.95). That head had to be trained. The obvious question the
README left open: how much of it does a tracker get for free?

Cycle consistency needs no head and no labels. Track forward, then re-track the
reversed video starting from where the forward pass ended, and measure how far
the two trajectories disagree at each frame. A point the tracker has lost
should not come back to where it started.

    cycle_error[q, t] = || forward[q, t] - backward[q, t] ||

One extra model pass per video, so it costs the same as the forward run.

    python -m tap_reliability.cycle --model tapir --tapnet-root third_party/tapnet
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import data, models
from .confidence import DELTA, auroc, calibration, eval_frame_mask, sigmoid
from .evaluate import pick_device, track_one

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@torch.no_grad()
def cycle_errors(model, sample, device, chunk=64):
    """Return (cycle_err [Q,T], forward tracks [Q,T,2])."""
    fwd = track_one(model, sample, device, chunk)
    tracks_f = fwd["tracks"][0]                      # [Q, T, 2] (x, y)
    video = sample["video"]
    n_frames = video.shape[1]

    # Re-query on the reversed video at its frame 0, i.e. the original last
    # frame, from the forward pass's own endpoint. query_points are [t, y, x].
    last_xy = tracks_f[:, -1, :]                     # [Q, 2] (x, y)
    q_back = np.zeros((1, last_xy.shape[0], 3), np.float32)
    q_back[0, :, 1] = last_xy[:, 1]                  # y
    q_back[0, :, 2] = last_xy[:, 0]                  # x
    back_sample = {"video": video[:, ::-1].copy(), "query_points": q_back}
    bwd = track_one(model, back_sample, device, chunk)
    tracks_b = bwd["tracks"][0][:, ::-1, :]          # flip time back
    return np.linalg.norm(tracks_f - tracks_b, axis=-1), tracks_f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(models.CONFIGS), default="tapir")
    ap.add_argument("--query-mode", default="first", choices=("first",))
    ap.add_argument("--davis", default="data/tapvid_davis/tapvid_davis.pkl")
    ap.add_argument("--checkpoint-dir", default="checkpoints")
    ap.add_argument("--tapnet-root", default="third_party/tapnet")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--delta", type=float, default=DELTA)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = pick_device(args.device)
    model = models.build(args.model, args.checkpoint_dir, device, args.tapnet_root)
    names = data.video_names(args.davis)
    if args.limit:
        names = names[: args.limit]
    print(f"[setup] {args.model} on {device}, {len(names)} videos")

    cyc, err, unc, t0 = [], [], [], time.time()
    pred_dir = Path(f"results/{args.model}_{args.query_mode}")
    for i, sample in enumerate(data.iter_davis(args.davis, args.query_mode, names)):
        name = sample["video_name"]
        ce, tracks_f = cycle_errors(model, sample, device)
        gt = sample["target_points"][0]
        occ = sample["occluded"][0]
        scored = eval_frame_mask(sample["query_points"], gt.shape[1],
                                 args.query_mode)[0]
        keep = scored & ~occ
        true_err = np.linalg.norm(tracks_f - gt, axis=-1)
        cyc.append(ce[keep])
        err.append(true_err[keep])
        # the trained head, from the predictions evaluate.py already saved
        z = np.load(pred_dir / f"{name}.npz")
        unc.append(sigmoid(z["expected_dist"][0])[keep])
        print(f"  [{i+1}/{len(names)}] {name:22s} "
              f"median cycle err {np.median(ce[keep]):5.2f} px", flush=True)

    cyc, err, unc = (np.concatenate(x) for x in (cyc, err, unc))
    bad = (err > args.delta).astype(np.float64)
    print(f"\n{len(err):,} scored visible predictions, "
          f"{bad.mean():.1%} with error > {args.delta:g} px "
          f"({time.time() - t0:.0f}s)\n")

    print(f"{'predictor':34}{'AUROC':>8}{'ECE':>8}")
    a_cyc, a_unc = auroc(cyc, bad), auroc(unc, bad)
    # cycle error is in pixels, so it has no calibration to report; the trained
    # head is a probability and does.
    print(f"{'cycle error (training-free)':34}{a_cyc:8.3f}{'  --':>8}")
    print(f"{'sigma(expected_dist) (trained)':34}{a_unc:8.3f}"
          f"{calibration(unc, bad)['ece']:8.3f}")
    comb = auroc(0.5 * (unc + np.minimum(cyc / 20.0, 1.0)), bad)
    print(f"{'mean of the two (rank-combined)':34}{comb:8.3f}{'  --':>8}")
    print(f"\ncorrelation between the two scores: "
          f"{np.corrcoef(cyc, unc)[0, 1]:.3f}")

    out = args.out or f"results/cycle_{args.model}_{args.query_mode}.json"
    Path(out).write_text(json.dumps(dict(
        model=args.model, query_mode=args.query_mode, n=len(err),
        delta=args.delta, frac_bad=float(bad.mean()),
        auroc_cycle=a_cyc, auroc_trained=a_unc, auroc_combined=comb,
        corr=float(np.corrcoef(cyc, unc)[0, 1]),
        median_cycle_err=float(np.median(cyc))), indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
