#!/usr/bin/env python3
"""Does the tracker know when it is wrong?

TAPIR predicts, per point per frame, a logit called `expected_dist`. Despite the
name it is not a distance: upstream's prob_loss trains it as a binary classifier
for "this prediction's L2 error exceeds delta", on VISIBLE frames only, with
delta = 6 px at 256x256 (the default in supervised_point_prediction.py, which no
config overrides; prob_loss's own default of 8.0 is unused on that path).

    invalid = (err > expected_dist_thresh**2)        # err is squared distance
    logprob = sigmoid_binary_cross_entropy(expd, invalid)
    logprob *= 1.0 - occluded

Upstream uses it only as a gate inside postprocess_occlusions:
sigma(expected_dist) never has to be a calibrated probability for that to work.
This asks whether it is one anyway, and whether it is useful for ranking:

  1. calibration  -- reliability bins, ECE and Brier score for P(err > delta)
  2. discrimination -- AUROC for the same event, against the occlusion logit
  3. selective prediction -- if the least confident predictions are discarded,
     how fast does the error on what is kept fall?

Everything runs from the saved .npz predictions, on CPU, in seconds.

    python -m tap_reliability.confidence --pred-dir results/tapir_first
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DELTA = 6.0


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def auroc(scores, labels):
    """AUROC via the rank statistic; ties get average ranks. No sklearn."""
    labels = labels.astype(bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return (ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def calibration(prob, label, n_bins=15):
    """Equal-width reliability bins, ECE and Brier."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(prob, edges[1:-1]), 0, n_bins - 1)
    rows, ece = [], 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        conf, freq, w = prob[m].mean(), label[m].mean(), m.mean()
        ece += w * abs(conf - freq)
        rows.append(dict(bin=b, lo=float(edges[b]), hi=float(edges[b + 1]),
                         n=int(m.sum()), mean_pred=float(conf),
                         observed=float(freq)))
    return dict(ece=float(ece), brier=float(np.mean((prob - label) ** 2)),
                bins=rows)


def eval_frame_mask(query_points, n_frames, query_mode):
    """Which (track, frame) pairs the benchmark scores. Mirrors the metric."""
    eye = np.eye(n_frames, dtype=np.int32)
    if query_mode == "first":
        table = np.cumsum(eye, axis=1) - eye
    elif query_mode == "strided":
        table = 1 - eye
    else:
        raise ValueError(query_mode)
    qf = np.round(query_points[..., 0]).astype(np.int32)
    return table[qf] > 0


def gather(pred_dir, query_mode, delta=DELTA):
    """Flatten every scored, ground-truth-visible prediction into 1-D arrays."""
    rows = {k: [] for k in ("err", "p_unc", "p_occ", "p_gate", "video")}
    files = sorted(Path(pred_dir).glob("*.npz"))
    if not files:
        raise SystemExit(f"[fatal] no .npz in {pred_dir}; run evaluate.py "
                         f"--save-predictions first")
    for f in files:
        z = np.load(f)
        gt, pr = z["gt_tracks"][0], z["tracks"][0]           # (N, T, 2)
        occ_gt = z["gt_occluded"][0]                          # (N, T)
        n_frames = gt.shape[1]
        scored = eval_frame_mask(z["query_points"], n_frames, query_mode)[0]
        # The uncertainty head is trained only on visible frames, so it is only
        # evaluated on visible frames here.
        keep = scored & ~occ_gt
        err = np.linalg.norm(pr - gt, axis=-1)
        rows["err"].append(err[keep])
        rows["p_unc"].append(sigmoid(z["expected_dist"][0])[keep])
        rows["p_occ"].append(sigmoid(z["occlusion"][0])[keep])
        rows["p_gate"].append(
            (1 - (1 - sigmoid(z["occlusion"][0])) *
             (1 - sigmoid(z["expected_dist"][0])))[keep])
        rows["video"].append(np.full(int(keep.sum()), f.stem))
    out = {k: np.concatenate(v) for k, v in rows.items()}
    out["bad"] = (out["err"] > delta).astype(np.float64)
    return out


def selective(err, score, fractions):
    """Mean error after discarding the highest-score (least confident) part."""
    order = np.argsort(score, kind="mergesort")          # most confident first
    e = err[order]
    out = []
    for keep in fractions:
        k = max(1, int(round(keep * len(e))))
        out.append(dict(coverage=keep, n=k, mean_err=float(e[:k].mean()),
                        frac_over_delta=float((e[:k] > DELTA).mean())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--query-mode", default=None,
                    help="defaults to the mode in the directory name")
    ap.add_argument("--delta", type=float, default=DELTA)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    mode = args.query_mode or ("strided" if "strided" in Path(args.pred_dir).name
                               else "first")
    d = gather(args.pred_dir, mode, args.delta)
    n = len(d["err"])
    print(f"{args.pred_dir}: {n:,} scored visible predictions, "
          f"{d['bad'].mean():.1%} with error > {args.delta:g} px\n")

    print(f"{'score':24}{'AUROC':>8}{'ECE':>8}{'Brier':>8}")
    report = {"n": n, "delta": args.delta, "query_mode": mode,
              "frac_bad": float(d["bad"].mean()), "scores": {}}
    for name, key in [("sigma(expected_dist)", "p_unc"),
                      ("sigma(occlusion)", "p_occ"),
                      ("upstream gate 1-(1-o)(1-u)", "p_gate")]:
        a = auroc(d[key], d["bad"])
        c = calibration(d[key], d["bad"])
        print(f"{name:24}{a:8.3f}{c['ece']:8.3f}{c['brier']:8.3f}")
        report["scores"][key] = dict(name=name, auroc=a, **c)

    fracs = [1.0, 0.9, 0.8, 0.7, 0.5, 0.3]
    print(f"\nselective prediction, ranking by sigma(expected_dist):")
    print(f"{'coverage':>9}{'mean err':>10}{'frac > delta':>14}")
    report["selective"] = {}
    for key in ("p_unc", "p_occ", "p_gate"):
        report["selective"][key] = selective(d["err"], d[key], fracs)
    for r in report["selective"]["p_unc"]:
        print(f"{r['coverage']:>8.0%}{r['mean_err']:>10.2f}"
              f"{r['frac_over_delta']:>13.1%}")

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
