"""Pooled scale analysis of the P3 channel across all baseline A40 measurements.

Pools the baseline (unperturbed) runs from every A40 session and identifies
chips by GPU UUID, so chips that recur across sessions are matched across them.
Uses P3 only (per-block arrival rank from the kernel's numerical output, no
%smid).

  12 chips / 72 measurements : 8-chip set + stability T1 + reboot T1
  19 chips / 112 measurements: the above + the scale-up session

Reports:
  * leave-one-out nearest-centroid accuracy
  * separation = mean distance between chip centroids / mean within-chip
    pairwise distance
  * worst-case margin = min centroid gap / max within-chip pairwise distance
  * fraction of chip pairs whose centroid gap exceeds 2x the mean within-chip
    distance

Usage: python analysis/pooled_analyze.py
"""
import glob
import itertools
import json
import os

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

POOLS = [
    ("12 chips (8-chip + stability T1 + reboot T1)",
     ["a40_8chip/*.json", "a40_stability/*_t1_*.json", "a40_reboot/*_t1_*.json"]),
    ("19 chips (+ scale-up session)",
     ["a40_8chip/*.json", "a40_stability/*_t1_*.json", "a40_reboot/*_t1_*.json",
      "a40_scale/*.json"]),
]


def p3_vec(run):
    mats = [np.array(e["p2"]["block_mean_rank"], dtype=float) for e in run["epochs"]]
    L = min(len(m) for m in mats)
    return np.mean([m[:L] for m in mats], axis=0)


def analyze(label, patterns):
    files = []
    for pat in patterns:
        files += sorted(glob.glob(os.path.join(DATA, pat)))
    runs = [json.load(open(p)) for p in files]
    X = [p3_vec(r) for r in runs]
    L = min(len(v) for v in X)
    X = np.array([v[:L] for v in X])
    lab = np.array([r["identity"]["uuid"] for r in runs])
    chips = sorted(set(lab))
    R = X - X.mean(0)                                   # remove architecture
    dist = lambda a, b: float(np.linalg.norm(a - b))
    idx = np.arange(len(R))

    within = [dist(R[i], R[j]) for i, j in itertools.combinations(idx, 2) if lab[i] == lab[j]]
    cent = {u: R[lab == u].mean(0) for u in chips}
    gaps = [dist(cent[a], cent[b]) for a, b in itertools.combinations(chips, 2)]

    ok = 0
    for i in idx:
        c = {u: R[(lab == u) & (idx != i)].mean(0) for u in chips
             if ((lab == u) & (idx != i)).any()}
        ok += min(c, key=lambda u: dist(R[i], c[u])) == lab[i]

    w = np.mean(within)
    print("\n%s" % label)
    print("  measurements %d, distinct chips %d, drivers %s"
          % (len(R), len(chips), sorted(set(r["identity"]["driver"] for r in runs))))
    print("  LOO nearest-centroid       : %d/%d = %.1f%%  (chance %.1f%%)"
          % (ok, len(R), 100.0 * ok / len(R), 100.0 / len(chips)))
    print("  separation (centroid gap / within) : %.2fx" % (np.mean(gaps) / w))
    print("  worst-case margin (min gap / max within) : %.2fx" % (min(gaps) / max(within)))
    print("  chip pairs with centroid gap > 2x within : %.0f%%"
          % (100.0 * np.mean(np.array(gaps) > 2 * w)))


def main():
    print("=" * 70)
    print("POOLED A40 SCALE ANALYSIS  (P3, numerical output only)")
    print("=" * 70)
    for label, pats in POOLS:
        analyze(label, pats)


if __name__ == "__main__":
    main()
