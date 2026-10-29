# Does a point tracker know when it's wrong?

A replication of DeepMind's [TAPIR and BootsTAPIR](https://github.com/google-deepmind/tapnet)
on TAP-Vid-DAVIS, plus a measurement the upstream repo does not make: whether
the uncertainty these models already predict is **calibrated** and **useful for
discarding bad tracks on video with no ground truth**.

Short answer: yes, clearly. σ(`expected_dist`) reaches **AUROC 0.93–0.95** at
telling correct predictions from wrong ones, with **expected calibration error
0.010–0.023** — and it was never trained to be calibrated. Dropping the least
confident 10% of predictions roughly halves the mean error.

## Part 1 — replication

Four configurations, all 30 DAVIS videos, metrics from the benchmark's own
`compute_tapvid_metrics` (vendored verbatim; see below). Published numbers are
the Jax figures from the upstream README, which states the PyTorch ports
"closely match".

| model | query mode | published AJ | **this repo** | diff | < δ | OA |
|---|---|---|---|---|---|---|
| TAPIR | first | 58.5 | **58.5** | −0.0 | 70.6 | 87.3 |
| TAPIR | strided | 63.3 | **62.8** | −0.5 | 74.7 | 89.5 |
| BootsTAPIR | first | 62.4 | **62.3** | −0.1 | 74.6 | 89.7 |
| BootsTAPIR | strided | 67.4 | **67.5** | +0.1 | 79.1 | 91.4 |

Three of four land within 0.1 AJ and the fourth within 0.5, comparing a
PyTorch run against published Jax numbers.

Two details decide whether this reproduces at all, and neither is in the
PyTorch defaults:

- **The architectures differ and the class defaults to BootsTAPIR.** The Jax
  configs give TAPIR `pyramid_level=0, extra_convs=False, softmax_temperature=20`
  and BootsTAPIR `pyramid_level=1, extra_convs=True, softmax_temperature=10`.
  Building TAPIR with the PyTorch defaults silently gives the wrong model, so
  every checkpoint here is loaded with `strict=True`. The TAPIR checkpoint
  confirms it: no `extra_convs.*` keys, and a mixer input width of 486 =
  388 + (0+2)·49, which is `pyramid_level=0`.
- **TAP-Vid-DAVIS ships at full resolution** (480×854, 480×910, 480×1152), not
  256×256, so every video is resampled — and upstream uses `mediapy`, which
  resizes with a **Lanczos** filter. Substituting PIL or OpenCV bilinear
  changes the inputs and therefore the scores.

## Part 2 — is the uncertainty any good?

TAPIR emits a logit per point per frame called `expected_dist`. The name is
misleading: upstream's `prob_loss` trains it as a **binary classifier for
"this prediction's error exceeds δ"**, on visible frames only, with δ = 6 px at
256×256 (the default in `supervised_point_prediction.py`, which no config
overrides — `prob_loss`'s own default of 8.0 is unused on that path).

Upstream uses it only as a gate inside `postprocess_occlusions`:

```python
visible = (1 - sigmoid(occlusion)) * (1 - sigmoid(expected_dist)) > 0.5
```

A gate needs only a decent threshold, not a calibrated probability. So: is it
calibrated anyway, and how well does it rank?

Measured on every scored, ground-truth-visible prediction (28,174 for `first`;
331,234 for `strided`), against the event *error > 6 px*:

| | | AUROC ↑ | ECE ↓ | Brier ↓ |
|---|---|---|---|---|
| TAPIR first | σ(expected_dist) | **0.947** | **0.023** | **0.056** |
| | σ(occlusion) | 0.872 | 0.064 | 0.093 |
| | upstream gate | 0.935 | 0.022 | 0.061 |
| TAPIR strided | σ(expected_dist) | **0.940** | **0.011** | **0.042** |
| | σ(occlusion) | 0.863 | 0.029 | 0.065 |
| | upstream gate | 0.924 | 0.032 | 0.050 |
| BootsTAPIR first | σ(expected_dist) | **0.936** | **0.021** | **0.054** |
| | σ(occlusion) | 0.815 | 0.051 | 0.078 |
| | upstream gate | 0.917 | 0.020 | 0.062 |
| BootsTAPIR strided | σ(expected_dist) | **0.929** | **0.010** | **0.037** |
| | σ(occlusion) | 0.835 | 0.027 | 0.054 |
| | upstream gate | 0.906 | 0.030 | 0.047 |

**Three findings, each consistent across all four configurations.**

1. **The head is well calibrated without being asked to be.** ECE 0.010–0.023
   means that when it says "15% chance this point is off by more than 6 px",
   about 15% of such points are. It is trained with binary cross-entropy, a
   proper scoring rule, so this is plausible — but nothing in the loss or the
   way it is used requires it, and it is not checked upstream.
2. **Occlusion is the weaker signal.** σ(occlusion) trails σ(expected_dist) by
   0.07–0.12 AUROC everywhere. Localisation failure and occlusion are not the
   same event, and the model separates them.
3. **The upstream gate is a slightly worse ranker than the uncertainty alone**
   (0.906–0.935 vs 0.929–0.947, lower in all four). Multiplying in the
   occlusion term is the right thing for deciding *visibility*, which is what
   the gate is for, but if the question is "how likely is this position wrong",
   `expected_dist` on its own is the better score.

### Selective prediction: what this buys you

Rank predictions by σ(`expected_dist`), discard the least confident, and
measure what remains. TAPIR, `first`:

| coverage | mean error (px) | fraction over 6 px |
|---|---|---|
| 100% | 4.69 | 14.4% |
| 90% | 2.56 | 6.6% |
| 80% | 1.93 | 2.8% |
| 70% | 1.64 | 1.5% |
| 50% | 1.38 | 0.7% |

Giving up a tenth of the predictions cuts mean error by 45% and the rate of
gross errors by more than half. The same shape holds in all four configs
(`results/confidence_*.json`). This needs no ground truth, so it transfers to
unlabelled video — which is the point.

## Part 3 — how much of that is free? (cycle consistency)

The uncertainty head had to be trained. A tracker can estimate its own
reliability without one: track forward, re-track the reversed video from where
the forward pass ended, and measure how far the two trajectories disagree.

    cycle_error[q, t] = || forward[q, t] - backward[q, t] ||

No head, no labels, one extra model pass. Scored on the same 28,174 predictions
against the same event (error > 6 px):

| predictor | TAPIR | BootsTAPIR | needs training? |
|---|---|---|---|
| cycle error | 0.665 | 0.691 | no |
| **σ(expected_dist)** | **0.947** | **0.936** | yes |
| naive mean of the two | 0.850 | 0.836 | — |

**The trained head is worth a great deal.** Cycle consistency is clearly better
than chance, so it is a usable fallback where no uncertainty head exists — but
it recovers only a fraction of the trained head's discrimination, 0.67 against
0.95. On this benchmark, training the head is justified rather than redundant.

The two scores are close to independent (correlation 0.12 and 0.18), so they
are not measuring the same thing. The naive mean scoring *below* the head alone
says only that equal weighting is the wrong way to combine a strong signal with
a weak one; it is not evidence that the weak one adds nothing. A fitted
combiner is untested here, and would need a held-out split to be honest.

One methodological note worth recording: on a 2-video subset cycle consistency
looked far stronger (0.813 against the head's 0.888). Across all 30 videos the
gap is three times wider. Two videos were not enough to rank these predictors.

## Reproducing

```bash
bash scripts/setup.sh          # tapnet checkout (pinned), DAVIS, 2 checkpoints: ~2.7 GB
python -m tap_reliability.evaluate --model tapir --query-mode first \
    --tapnet-root third_party/tapnet --save-predictions
python -m tap_reliability.confidence --pred-dir results/tapir_first
python -m tap_reliability.cycle --model tapir        # the training-free baseline
```

About 2 minutes per configuration for `first` and ~25 for `strided` on an
Apple M-series GPU (MPS); `--device cuda` works too. The analysis runs on CPU
from the saved predictions in seconds.

## What is and isn't established here

- **DAVIS only.** TAP-Vid also has Kinetics (1000 videos) and RGB-Stacking, and
  RoboTAP. Whether the calibration holds off DAVIS is untested, and DAVIS is 30
  videos — the per-config AUROC spread here is small, but it is one dataset.
- **δ = 6 px is the trained threshold**, so the head is being scored on exactly
  the event it was fit to. Calibration at other thresholds is a different
  question and is not claimed.
- **δ = 6 px is the trained threshold** for the head, so cycle consistency is
  being scored on the head's home ground. A different threshold is a different
  question and is not claimed.
- These are the published checkpoints, evaluated as released. Nothing is
  retrained.

## Attribution

Upstream is [google-deepmind/tapnet](https://github.com/google-deepmind/tapnet)
(Apache 2.0), pinned at commit `730cda1c73`. The PyTorch TAPIR model code is
imported from that checkout, not copied.

`vendor/tapvid_official_eval.py` contains three functions copied **verbatim**
from `tapnet/tapvid/evaluation_datasets.py` — `compute_tapvid_metrics`,
`sample_queries_first`, `sample_queries_strided` — so the benchmark's own metric
and query sampling are used rather than a reimplementation. They are copied
rather than imported because the upstream module imports TensorFlow at module
level, which these pure-numpy functions do not use.
`tests/test_vendored_is_verbatim.py` re-fetches the pinned commit and fails if
they have drifted:

```bash
python tests/test_vendored_is_verbatim.py
```

TAP-Vid-DAVIS and the checkpoints are downloaded from Google's public
storage by `scripts/setup.sh` and are not redistributed here.
