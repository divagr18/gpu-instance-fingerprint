"""ROC and bit-capacity: the fingerprint scored as a biometric.

Every pair of measurements is scored. Genuine = same GPU UUID, impostor =
different UUID. Scores are single-shot (one raw measurement against another,
no enrollment averaging), so these are pessimistic operating points; the
LOO numbers elsewhere use multi-measurement enrollment centroids.

The architecture is removed first (subtract the grand mean across all
measurements in the set). Two scores per set:
  * residual correlation (Pearson)            - higher = more similar
  * fractional Hamming distance of sign(residual) - lower = more similar
Reported: AUC, EER (at the threshold where FAR and FRR cross, the larger of
the two), d', and for the Hamming score the iris-style degrees of freedom
DoF = p(1-p)/var(HD) over impostor pairs (number of independent bits).

Sets:
  active P3, 19 A40s : every A40 probe measurement (baseline, stress, reboot,
                       scale-up), per-block arrival rank
  passive K=4096     : data/passive_k4096 (8 A40s)
  passive K=65536    : data/passive_k65536 (6 A40s)
  passive RTX 4090   : data/rtx4090 passive (3 chips, K=65536)

Usage: python analysis/roc_capacity.py
"""
import glob
import itertools
import json
import os

import numpy as np
from scipy.stats import rankdata

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def load(patterns):
    files = []
    for pat in patterns:
        files += sorted(glob.glob(os.path.join(DATA, pat)))
    return [json.load(open(p)) for p in files if os.path.getsize(p) > 0]


def auc(gen, imp):
    """P(genuine score > impostor score), ties counted half (Mann-Whitney)."""
    r = rankdata(np.concatenate([gen, imp]))
    return (r[:len(gen)].sum() - len(gen) * (len(gen) + 1) / 2) / (len(gen) * len(imp))


def eer(gen, imp):
    """Equal error rate; scores where higher = genuine."""
    best = None
    for t in np.unique(np.concatenate([gen, imp])):
        frr = np.mean(gen < t)
        far = np.mean(imp >= t)
        if best is None or abs(far - frr) < best[0]:
            best = (abs(far - frr), max(far, frr))
    return best[1]


def dprime(gen, imp):
    return abs(gen.mean() - imp.mean()) / np.sqrt(0.5 * (gen.var() + imp.var()))


def report(name, X, lab):
    X = np.asarray(X, float)
    R = X - X.mean(0)                                   # remove architecture
    C = np.corrcoef(R)
    S = np.sign(R)
    g_c, i_c, g_h, i_h = [], [], [], []
    for i, j in itertools.combinations(range(len(R)), 2):
        hd = float(np.mean(S[i] != S[j]))
        if lab[i] == lab[j]:
            g_c.append(C[i, j]); g_h.append(hd)
        else:
            i_c.append(C[i, j]); i_h.append(hd)
    g_c, i_c, g_h, i_h = map(np.array, (g_c, i_c, g_h, i_h))
    p = i_h.mean()
    print("\n%s: %d measurements, %d chips, %d genuine / %d impostor pairs"
          % (name, len(R), len(set(lab)), len(g_c), len(i_c)))
    print("    residual correlation : AUC %.3f  EER %4.1f%%  d'=%.2f"
          % (auc(g_c, i_c), 100 * eer(g_c, i_c), dprime(g_c, i_c)))
    print("    sign Hamming         : AUC %.3f  EER %4.1f%%  d'=%.2f  DoF ~%.0f bits"
          % (auc(-g_h, -i_h), 100 * eer(-g_h, -i_h), dprime(g_h, i_h), p * (1 - p) / i_h.var()))


def p3_vec(run):
    mats = [np.array(e["p2"]["block_mean_rank"], dtype=float) for e in run["epochs"]]
    L = min(len(m) for m in mats)
    return np.mean([m[:L] for m in mats], axis=0)


def main():
    print("=" * 74)
    print("ROC / BIT-CAPACITY (single-shot, architecture removed)")
    print("=" * 74)
    runs = load(["a40_8chip/*.json", "a40_stability/*.json", "a40_reboot/*.json",
                 "a40_scale/*.json"])
    v = [p3_vec(r) for r in runs]
    L = min(len(x) for x in v)
    report("ACTIVE P3, A40", [x[:L] for x in v], [r["identity"]["uuid"] for r in runs])
    for name, pat in [("PASSIVE K=4096, A40", "passive_k4096/*.json"),
                      ("PASSIVE K=65536, A40", "passive_k65536/*.json"),
                      ("PASSIVE K=65536, RTX 4090", "rtx4090/g90_pass_*.json")]:
        rs = load([pat])
        report(name, [r["fp_sign"] for r in rs], [r["identity"]["uuid"] for r in rs])


if __name__ == "__main__":
    main()
