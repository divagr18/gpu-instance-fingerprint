"""Passive fingerprint analysis: is the chip identifiable from plain op output?

Datasets (all A40):
  passive_k65536  sparse index_add, K=65536 sign signature, 6 chips x 3 reps
  passive_k4096   sparse index_add, K=4096 sign signature, 8 chips x 3 reps
  passive_dense   dense index_add (2^20 contributions into 256 bins), 6 chips x 4
                  reps -> negative control: summation averages the order away
  passive_scatter scatter-overwrite (last writer wins), 5 chips x 4 reps

Method: subtract the cross-chip grand mean (architecture), then leave-one-out
nearest-centroid classification on Euclidean distance. For the sparse
datasets the analysis is repeated within each driver version, since the raw
op output also carries a driver-dependent component. The binomial p-value is
P(at least k of n correct) at chance 1/n_chips.

Usage: python analysis/passive_analyze.py
"""
import glob
import json
import os
from math import comb

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def load(pat):
    return [json.load(open(p)) for p in sorted(glob.glob(os.path.join(DATA, pat)))]


def loo(vecs, labs):
    L = min(len(v) for v in vecs)
    X = np.vstack([np.asarray(v, float)[:L] for v in vecs])
    R = X - X.mean(0)
    lab = np.array(labs)
    chips = sorted(set(labs))
    idx = np.arange(len(R))
    ok = 0
    for i in idx:
        c = {u: R[(lab == u) & (idx != i)].mean(0) for u in chips
             if ((lab == u) & (idx != i)).any()}
        ok += min(c, key=lambda u: np.linalg.norm(R[i] - c[u])) == lab[i]
    return int(ok), len(R), len(chips)


def binom_p(k, n, p):
    return sum(comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(k, n + 1))


def show(name, ok, n, nchips):
    print("    %-34s LOO %2d/%2d = %3.0f%%  (chance %.0f%%, binomial p=%.1e)"
          % (name, ok, n, 100.0 * ok / n, 100.0 / nchips, binom_p(ok, n, 1.0 / nchips)))


def sparse(ds, title):
    runs = load(ds + "/*.json")
    labs = [r["identity"]["uuid"] for r in runs]
    print("\n%s  [%s]  K=%d, %d chips, %d measurements" % (title, ds, runs[0]["K"],
                                                         len(set(labs)), len(runs)))
    print("    fraction of bins deterministic run-to-run: %.3f"
          % np.mean([r["det_frac"] for r in runs]))
    show("all drivers, sign signature", *loo([r["fp_sign"] for r in runs], labs))
    by = {}
    for r in runs:
        by.setdefault(r["identity"]["driver"], []).append(r)
    for drv, sub in sorted(by.items()):
        sl = [r["identity"]["uuid"] for r in sub]
        if len(set(sl)) >= 2:
            show("driver %s only" % drv, *loo([r["fp_sign"] for r in sub], sl))


def main():
    print("=" * 74)
    print("PASSIVE FINGERPRINT FROM PLAIN OP OUTPUT")
    print("=" * 74)
    sparse("passive_k65536", "Sparse index_add")
    sparse("passive_k4096", "Sparse index_add")

    runs = load("passive_dense/*.json")
    labs = [r["identity"]["uuid"] for r in runs]
    print("\nDense index_add (negative control)  [passive_dense]  %d chips, %d measurements"
          % (len(set(labs)), len(runs)))
    print("    distinct float32 results per bin: median %.0f, max %d"
          % (np.median([r["distinct_median"] for r in runs]), max(r["distinct_max"] for r in runs)))
    for f, name in [("fp_mean", "per-bin mean residual"), ("fp_std", "per-bin residual std"),
                    ("fp_p1", "fraction-positive residual")]:
        show(name, *loo([r[f] for r in runs], labs))
    show("combined", *loo([np.concatenate([r["fp_mean"], r["fp_std"], r["fp_p1"]])
                            for r in runs], labs))

    runs = load("passive_scatter/*.json")
    labs = [r["identity"]["uuid"] for r in runs]
    print("\nScatter-overwrite, last writer wins  [passive_scatter]  %d chips, %d measurements"
          % (len(set(labs)), len(runs)))
    show("per-bin mean winning id", *loo([r["v1"]["fp_mean"] for r in runs], labs))


if __name__ == "__main__":
    main()
