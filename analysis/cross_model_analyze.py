"""Cross-model generality: does the fingerprint separate identical chips beyond A40?

  RTX 3090 (Ampere consumer), 7 chips : data/rtx3090 (5 chips) + the 2 RTX 3090s
                                        in data/cross_model. Epochs act as
                                        replicates; LOO + 2000-shuffle permutation.
  RTX 4090 (Ada consumer)             : data/rtx4090. Active probe on 2 chips
                                        (3 replicates each); passive sparse
                                        index_add on 3 chips (2 reps each).
  RTX 3090 pair, L40S pair            : data/cross_model. Active probe (epochs as
                                        replicates) + passive sparse index_add.

Channels: P3 per-block numerical rank, P2 per-SM %smid rank, passive sign
signature. Architecture removed by subtracting the per-model grand mean.
Reports within/between-chip mean distance, their ratio, and LOO accuracy.

Usage: python analysis/cross_model_analyze.py
"""
import glob
import json
import os

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def loadj(p):
    if os.path.getsize(p) == 0:
        return None
    try:
        return json.load(open(p))
    except Exception:
        return None


def load(pat):
    return [d for d in (loadj(p) for p in sorted(glob.glob(os.path.join(DATA, pat)))) if d]


def p3(ep_list):
    m = [np.array(e["p2"]["block_mean_rank"], float) for e in ep_list]
    L = min(len(x) for x in m)
    return [x[:L] for x in m]


def p2(ep_list):
    out = []
    for e in ep_list:
        n = e["p2"]["n_sm"]
        out.append(np.array([np.nan if x is None else x for x in e["p2"]["sm_mean_rank"][:n]], float))
    L = min(len(x) for x in out)
    return [x[:L] for x in out]


def _d(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    return np.linalg.norm(a[m] - b[m])


def _loo(R, lab, chips):
    ok = 0
    for i in range(len(R)):
        cen = {u: np.nanmean([R[k] for k in range(len(R)) if lab[k] == u and k != i], 0)
               for u in chips if any(lab[k] == u and k != i for k in range(len(R)))}
        if cen and min(cen, key=lambda u: _d(R[i], cen[u])) == lab[i]:
            ok += 1
    return ok


def sep_and_loo(vecs, labels, name, perm=0):
    L = min(len(v) for v in vecs)
    X = np.vstack([v[:L] for v in vecs])
    R = X - np.nanmean(X, 0)                       # architecture removal
    lab = np.array(labels)
    chips = sorted(set(labels))
    win = [_d(R[i], R[j]) for i in range(len(R)) for j in range(i + 1, len(R)) if lab[i] == lab[j]]
    bet = [_d(R[i], R[j]) for i in range(len(R)) for j in range(i + 1, len(R)) if lab[i] != lab[j]]
    ok = _loo(R, lab, chips)
    w = np.mean(win) if win else float('nan')
    b = np.mean(bet) if bet else float('nan')
    msg = ("    [%s] within %.4f  between %.4f  sep %.2fx  LOO %d/%d (chance %.0f%%)"
           % (name, w, b, b / w if w else float('nan'), ok, len(R), 100 / len(chips)))
    if perm:
        rng = np.random.default_rng(0)
        ge = sum(_loo(R, rng.permutation(lab), chips) >= ok for _ in range(perm))
        msg += "  perm p=%.4f (%d shuffles)" % ((ge + 1) / (perm + 1), perm)
    print(msg)


def active(runs, per_epoch=True, perm=0):
    """per_epoch: each epoch is a replicate; else each run (epochs averaged) is one."""
    v3, v2, lab = [], [], []
    for d in runs:
        a, b = p3(d["epochs"]), p2(d["epochs"])
        if per_epoch:
            v3 += a
            v2 += b
            lab += [d["identity"]["uuid"]] * len(a)
        else:
            v3.append(np.mean(a, 0))
            v2.append(np.nanmean(b, 0))
            lab.append(d["identity"]["uuid"])
    sep_and_loo(v3, lab, "P3 per-block numerical", perm)
    sep_and_loo(v2, lab, "P2 per-SM %smid", perm)


def passive(runs):
    sep_and_loo([np.array(d["fp_sign"], float) for d in runs],
                [d["identity"]["uuid"] for d in runs], "passive sparse sign")
    print("    passive det_frac=%.3f" % np.mean([d["det_frac"] for d in runs]))


def header(title, runs):
    chips = sorted(set(d["identity"]["uuid"] for d in runs))
    drivers = sorted(set(d["identity"]["driver"] for d in runs))
    print("\n%s  (SMs=%s, distinct chips=%d, drivers=%s)"
          % (title, runs[0]["identity"].get("sm_count", "?"), len(chips), drivers))


def main():
    print("=" * 74)
    print("CROSS-MODEL GENERALITY")
    print("=" * 74)

    xm_probe = load("cross_model/xm_probe_*.json")
    xm_pass = load("cross_model/xm_pass_*.json")

    r3090 = load("rtx3090/*.json") + [d for d in xm_probe if "3090" in d["identity"]["name"]]
    header("NVIDIA GeForce RTX 3090, within-model scale-up", r3090)
    active(r3090, per_epoch=True, perm=2000)

    g90 = load("rtx4090/g90_probe_*.json")
    header("NVIDIA GeForce RTX 4090, active probe", g90)
    active(g90, per_epoch=False)
    g90p = load("rtx4090/g90_pass_*.json")
    header("NVIDIA GeForce RTX 4090, passive", g90p)
    passive(g90p)

    for nm in sorted(set(d["identity"]["name"] for d in xm_probe + xm_pass)):
        ps = [d for d in xm_probe if d["identity"]["name"] == nm]
        pa = [d for d in xm_pass if d["identity"]["name"] == nm]
        header("%s pair" % nm, ps + pa)
        active(ps, per_epoch=True)
        passive(pa)


if __name__ == "__main__":
    main()
