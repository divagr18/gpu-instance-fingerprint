"""Temporal stability: does a chip re-identify after a perturbation?

Enroll each chip on T1 (baseline measurements), then classify measurements
taken after a perturbation against the T1 enrollment centroids:

  a40_stability : T2 = after 60 s of full GPU stress (thermal/load shift)
  a40_reboot    : T3 = after a container restart (fresh CUDA context, fresh
                  install), same physical GPU verified by UUID

Chips are matched by the GPU UUID stored in each JSON. Runs both channels:
  P3  per-block arrival rank (numerical output only, no %smid)
  P2  per-SM arrival rank (%smid)

Usage: python analysis/temporal_analyze.py [a40_stability|a40_reboot ...]
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
LABELS = {"t2": "T2 after 60s STRESS", "t3": "T3 after REBOOT"}


def load(dataset):
    out = []
    for p in sorted(glob.glob(os.path.join(DATA, dataset, "*.json"))):
        if os.path.getsize(p) == 0:
            continue
        d = json.load(open(p))
        d["_path"] = os.path.basename(p)
        out.append(d)
    return out


def p3_vec(run):
    mats = [np.array(e["p2"]["block_mean_rank"], dtype=float) for e in run["epochs"]]
    L = min(len(m) for m in mats)
    return np.mean(np.vstack([m[:L] for m in mats]), axis=0)


def p2_vec(run):
    mats = []
    for e in run["epochs"]:
        n = e["p2"]["n_sm"]
        mats.append(np.array([np.nan if x is None else x
                              for x in e["p2"]["sm_mean_rank"][:n]], dtype=float))
    L = min(len(m) for m in mats)
    return np.nanmean(np.vstack([m[:L] for m in mats]), axis=0)


def phase(path):
    for t in ("t1", "t2", "t3"):
        if "_%s_" % t in path:
            return t
    return "?"


def dist(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    return float(np.linalg.norm(a[m] - b[m]))


def run_channel(runs, vec_fn, name):
    for r in runs:
        r["_v"] = vec_fn(r)
    L = min(len(r["_v"]) for r in runs)
    T1 = [r for r in runs if r["_phase"] == "t1"]
    grand = np.nanmean([r["_v"][:L] for r in T1], axis=0)      # architecture, from T1 only
    for r in runs:
        r["_res"] = r["_v"][:L] - grand
    chips = sorted(set(r["_uuid"] for r in T1))
    cent = {u: np.nanmean([r["_res"] for r in T1 if r["_uuid"] == u], axis=0) for u in chips}
    chance = 100.0 / len(chips)
    print("\n  --- %s ---" % name)

    ok = 0
    for r in T1:                                                # LOO within T1
        others = defaultdict(list)
        for q in T1:
            if q is not r:
                others[q["_uuid"]].append(q["_res"])
        c = {u: np.nanmean(v, axis=0) for u, v in others.items()}
        ok += min(c, key=lambda u: dist(r["_res"], c[u])) == r["_uuid"]
    print("  [T1 baseline] leave-one-out: %d/%d = %.0f%% (chance %.0f%%)"
          % (ok, len(T1), 100.0 * ok / len(T1), chance))

    for ph in ("t2", "t3"):
        meas = [r for r in runs if r["_phase"] == ph]
        if not meas:
            continue
        ok = 0
        d_true, d_other = [], []
        for r in meas:
            d = {u: dist(r["_res"], cent[u]) for u in cent}
            ok += min(d, key=d.get) == r["_uuid"]
            d_true.append(d[r["_uuid"]])
            d_other.append(min(v for u, v in d.items() if u != r["_uuid"]))
        print("  [%s] re-identify to T1 enrollment: %d/%d = %.0f%%  (chance %.0f%%)"
              % (LABELS[ph], ok, len(meas), 100.0 * ok / len(meas), chance))
        print("      mean dist to true chip %.5f, to nearest other %.5f  (margin %.2fx)"
              % (np.mean(d_true), np.mean(d_other), np.mean(d_other) / max(np.mean(d_true), 1e-12)))


def main():
    datasets = sys.argv[1:] or ["a40_stability", "a40_reboot"]
    for ds in datasets:
        runs = load(ds)
        for r in runs:
            r["_uuid"] = r["identity"]["uuid"]
            r["_phase"] = phase(r["_path"])
        chips = sorted(set(r["_uuid"] for r in runs))
        print("=" * 70)
        print("TEMPORAL STABILITY [%s]: %d chips, %d measurements" % (ds, len(chips), len(runs)))
        print("=" * 70)
        for u in chips:
            ph = sorted(set(r["_phase"] for r in runs if r["_uuid"] == u))
            n = sum(1 for r in runs if r["_uuid"] == u)
            print("  chip %s..  measurements=%d  phases=%s" % (u[:16], n, ph))
        run_channel(runs, p3_vec, "P3 per-block rank (numerical, no %smid)")
        run_channel(runs, p2_vec, "P2 per-SM rank (%smid)")
        print()


if __name__ == "__main__":
    main()
