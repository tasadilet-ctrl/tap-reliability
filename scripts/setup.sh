#!/usr/bin/env bash
# Fetches everything needed to reproduce the numbers in the README:
# the pinned tapnet checkout (for its PyTorch TAPIR code), the TAP-Vid-DAVIS
# benchmark, and the two published checkpoints. ~2.7 GB total.
set -euo pipefail
cd "$(dirname "$0")/.."

TAPNET_COMMIT=730cda1c73   # the commit the vendored eval code was taken from

mkdir -p third_party data checkpoints
if [ ! -d third_party/tapnet ]; then
  git clone -q https://github.com/google-deepmind/tapnet.git third_party/tapnet
fi
git -C third_party/tapnet fetch -q --depth 1 origin "$TAPNET_COMMIT" 2>/dev/null \
  || git -C third_party/tapnet fetch -q origin
git -C third_party/tapnet checkout -q "$TAPNET_COMMIT"
echo "tapnet at $(git -C third_party/tapnet rev-parse --short HEAD)"

if [ ! -f data/tapvid_davis/tapvid_davis.pkl ]; then
  curl -L -o /tmp/tapvid_davis.zip https://storage.googleapis.com/dm-tapnet/tapvid_davis.zip
  unzip -q -o /tmp/tapvid_davis.zip -d data/ && rm /tmp/tapvid_davis.zip
fi
[ -f checkpoints/tapir_panning.pt ] || curl -L -o checkpoints/tapir_panning.pt \
  https://storage.googleapis.com/dm-tapnet/tapir_checkpoint_panning.pt
[ -f checkpoints/bootstapir_v2.pt ] || curl -L -o checkpoints/bootstapir_v2.pt \
  https://storage.googleapis.com/dm-tapnet/bootstap/bootstapir_checkpoint_v2.pt

python3 tests/test_vendored_is_verbatim.py
echo "ready: python -m tap_reliability.evaluate --model tapir --query-mode first --tapnet-root third_party/tapnet"
