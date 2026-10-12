#!/usr/bin/env python3
"""Checks the vendored TAP-Vid functions still match upstream, byte for byte.

The replication's credibility rests on using the benchmark's own metric. This
clones the pinned upstream commit and compares the copied function bodies, so
an accidental local edit cannot quietly change what "AJ" means here.

    python tests/test_vendored_is_verbatim.py            # uses the pinned commit
    python tests/test_vendored_is_verbatim.py --ref main # check against upstream HEAD
"""
import argparse
import ast
import pathlib
import re
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
VENDORED = ROOT / "vendor" / "tapvid_official_eval.py"
UPSTREAM = "https://github.com/google-deepmind/tapnet.git"
REL = "tapnet/tapvid/evaluation_datasets.py"
FUNCS = ["compute_tapvid_metrics", "sample_queries_strided", "sample_queries_first"]


def bodies(text):
    lines = text.splitlines(True)
    tree = ast.parse(text)
    out = {}
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name in FUNCS:
            out[n.name] = "".join(lines[n.lineno - 1: n.end_lineno]).rstrip() + "\n"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=None, help="git ref to compare against "
                                               "(default: the pinned commit)")
    args = ap.parse_args()

    local = VENDORED.read_text()
    pinned = re.search(r"at commit ([0-9a-f]{7,40})", local)
    if not pinned:
        sys.exit("[fatal] vendored file does not record its source commit")
    ref = args.ref or pinned.group(1)

    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["git", "init", "-q", tmp], check=True)
        subprocess.run(["git", "-C", tmp, "remote", "add", "origin", UPSTREAM], check=True)
        r = subprocess.run(["git", "-C", tmp, "fetch", "-q", "--depth", "1", "origin", ref],
                           capture_output=True, text=True)
        if r.returncode:
            sys.exit(f"[fatal] could not fetch {ref} from upstream: {r.stderr.strip()}")
        up = subprocess.run(["git", "-C", tmp, "show", f"FETCH_HEAD:{REL}"],
                            capture_output=True, text=True, check=True).stdout

    mine, theirs = bodies(local), bodies(up)
    bad = []
    for name in FUNCS:
        if name not in theirs:
            bad.append(f"{name}: not found upstream at {ref}")
        elif mine.get(name) != theirs[name]:
            bad.append(f"{name}: DIFFERS from upstream")
        else:
            print(f"  [ok] {name} matches upstream {ref[:10]}")
    if bad:
        print("\n".join("  [FAIL] " + b for b in bad))
        sys.exit(1)
    print(f"\nall {len(FUNCS)} vendored functions verbatim at {ref[:10]}")


if __name__ == "__main__":
    main()
