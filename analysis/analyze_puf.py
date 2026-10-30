"""Within-chip vs between-chip separation of the active-probe fingerprint.

Takes a set of probes/probe_puf.py JSON outputs (multiple chips, multiple
replicates each) and asks whether the between-chip variation in the signature
is larger than the within-chip (replicate + thermal) variation.

The architecture is removed so only the instance remains: for each channel,
subtract the cross-chip grand mean from every measurement's vector and keep the
per-chip residual.

  P1  per-input symbol-probability vector of the float32 reduction result
  P2  per-SM mean arrival rank (%smid)
  P3  per-block mean arrival rank (numerical output only, no %smid)

Reports, per GPU model:
  * within-chip vs between-chip mean pairwise distance and their ratio
  * leave-one-out nearest-centroid classification accuracy (chance = 1/n_chips)
  * a 2000-shuffle label-permutation p-value for the LOO accuracy

Usage (globs are resolved against the repo's data/ directory if they match
nothing relative to the current directory):
  python analysis/analyze_puf.py "a40_8chip/*.json"               # 8 A40s
  python analysis/analyze_puf.py "a40_8chip/puf8_[4678]_*.json"   # same-driver 4
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def load(paths):
    runs = []
    for p in paths:
        try:
            d = json.load(open(p))
        except Exception as e:
            sys.stderr.write("skip %s: %s\n" % (p, e))
            continue
        d["_path"] = p
        runs.append(d)
    return runs


def p2_vector(run):
    """Per-SM mean arrival rank, averaged over this run's epochs."""
    mats = []
    for e in run["epochs"]:
        v = e["p2"]["sm_mean_rank"]
        n = e["p2"]["n_sm"]
        row = np.array([np.nan if x is None else x for x in v[:n]], dtype=float)
        mats.append(row)
    L = min(len(m) for m in mats)
    return np.nanmean(np.vstack([m[:L] for m in mats]), axis=0)


def p1_vectors(run):
    """One probability vector per fixed input, over that input's own support."""
    out = []
    n_inputs = len(run["epochs"][0]["p1"])
    for k in range(n_inputs):
        acc = defaultdict(float)
        for e in run["epochs"]:
            for sym, c in e["p1"][k].items():
                acc[int(sym)] += c
        out.append(acc)
    return out


def build_p1_common(runs):
    """Align every run's P1 dicts onto a shared support per input."""
    n_inputs = len(runs[0]["epochs"][0]["p1"])
    supports = []
    for k in range(n_inputs):
        syms = set()
        for r in runs:
            for e in r["epochs"]:
                syms |= set(int(s) for s in e["p1"][k])
        supports.append(sorted(syms))
    vecs = {}
    for r in runs:
        pv = p1_vectors(r)
        parts = []
        for k in range(n_inputs):
            sup = supports[k]
            v = np.array([pv[k].get(s, 0.0) for s in sup], dtype=float)
            v = v / v.sum() if v.sum() else v
            parts.append(v)
        vecs[r["_path"]] = np.concatenate(parts)
    return vecs


def separation(chip_of, vec_of, label):
    """within vs between pairwise distance + leave-one-out classification."""
    paths = list(vec_of.keys())
    chips = sorted(set(chip_of[p] for p in paths))
    if len(chips) < 2:
        print("  [%s] only %d chip(s) present - within-chip only" % (label, len(chips)))
        wd = _within(chip_of, vec_of)
        print("      within-chip mean distance : %.6f" % wd)
        return
    win, bet = [], []
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            d = np.linalg.norm(vec_of[paths[i]] - vec_of[paths[j]])
            (win if chip_of[paths[i]] == chip_of[paths[j]] else bet).append(d)
    win, bet = np.array(win), np.array(bet)

    # leave-one-out nearest centroid
    correct = 0
    for p in paths:
        cents = {}
        for c in chips:
            others = [q for q in paths if chip_of[q] == c and q != p]
            if others:
                cents[c] = np.mean([vec_of[q] for q in others], axis=0)
        if not cents:
            continue
        pred = min(cents, key=lambda c: np.linalg.norm(vec_of[p] - cents[c]))
        correct += (pred == chip_of[p])
    acc = correct / len(paths)

    # permutation test: how often does shuffled-label LOO beat the observed acc?
    rng = np.random.default_rng(0)
    truth = [chip_of[p] for p in paths]
    ge = 0
    TRIALS = 2000
    for _ in range(TRIALS):
        perm = list(rng.permutation(truth))
        cmap = {p: perm[i] for i, p in enumerate(paths)}
        c2 = 0
        for p in paths:
            cents = {}
            for c in chips:
                oth = [q for q in paths if cmap[q] == c and q != p]
                if oth:
                    cents[c] = np.mean([vec_of[q] for q in oth], axis=0)
            if cents:
                pr = min(cents, key=lambda c: np.linalg.norm(vec_of[p] - cents[c]))
                c2 += (pr == cmap[p])
        if c2 / len(paths) >= acc:
            ge += 1
    pval = (ge + 1) / (TRIALS + 1)

    print("  [%s]  %d chips, %d measurements" % (label, len(chips), len(paths)))
    print("      within-chip  mean dist : %.6f  (n=%d)" % (win.mean() if len(win) else float('nan'), len(win)))
    print("      between-chip mean dist : %.6f  (n=%d)" % (bet.mean() if len(bet) else float('nan'), len(bet)))
    if len(win) and win.mean() > 0:
        print("      separation ratio       : %.2fx  %s"
              % (bet.mean() / win.mean(),
                 "<-- FINGERPRINT" if bet.mean() / win.mean() > 2 else "(weak/none)"))
    print("      LOO nearest-centroid   : %d/%d = %.1f%% correct  (chance %.1f%%)  %s"
          % (correct, len(paths), acc * 100, 100.0 / len(chips),
             "<-- SEPARABLE" if acc > 1.5 / len(chips) else "(near chance)"))
    print("      permutation p-value    : %.4f  (%d shuffles)  %s"
          % (pval, TRIALS, "significant" if pval < 0.05 else "NOT significant"))


def _within(chip_of, vec_of):
    paths = list(vec_of.keys())
    ds = [np.linalg.norm(vec_of[paths[i]] - vec_of[paths[j]])
          for i in range(len(paths)) for j in range(i + 1, len(paths))
          if chip_of[paths[i]] == chip_of[paths[j]]]
    return float(np.mean(ds)) if ds else float("nan")


def main():
    paths = []
    for a in sys.argv[1:] or ["a40_8chip/*.json"]:
        hits = sorted(glob.glob(a)) or sorted(glob.glob(os.path.join(DATA, a)))
        paths += hits
    runs = load(paths)
    if not runs:
        print("no JSON inputs")
        return

    by_model = defaultdict(list)
    for r in runs:
        by_model[r["identity"].get("name", "?")].append(r)

    print("=" * 74)
    print("GPU INSTANCE FINGERPRINT ANALYSIS   %d runs, %d model(s)"
          % (len(runs), len(by_model)))
    print("=" * 74)
    for model, rs in by_model.items():
        chips = sorted(set(r["identity"].get("uuid") for r in rs))
        drivers = sorted(set(r["identity"].get("driver") for r in rs))
        print("\n%s   %d measurements, %d distinct chip(s), driver(s)=%s"
              % (model, len(rs), len(chips), drivers))
        for r in rs:
            e0 = r["epochs"][0]
            print("    %-14s uuid=%s repl=%s temp~%s clk~%s"
                  % (os.path.basename(r["_path"])[:14],
                     (r["identity"].get("uuid") or "?")[:20],
                     r.get("replicate"), e0.get("temp_post"), e0.get("clock_post")))

        chip_of = {r["_path"]: r["identity"].get("uuid") for r in rs}
        p2 = {r["_path"]: p2_vector(r) for r in rs}
        L = min(len(v) for v in p2.values())
        grand = np.nanmean(np.vstack([v[:L] for v in p2.values()]), axis=0)
        p2res = {k: (v[:L] - grand) for k, v in p2.items()}  # remove architecture
        separation(chip_of, p2res, "P2 ordering / per-SM residual")

        p1 = build_p1_common(rs)
        separation(chip_of, p1, "P1 numerical / symbol-prob")

        # P3: per-block arrival rank (no %smid - numerical output only)
        if "block_mean_rank" in rs[0]["epochs"][0]["p2"]:
            b = {}
            for r in rs:
                mats = [np.array(e["p2"]["block_mean_rank"], dtype=float)
                        for e in r["epochs"]]
                b[r["_path"]] = np.mean(np.vstack(mats), axis=0)
            Lb = min(len(v) for v in b.values())
            gb = np.mean(np.vstack([v[:Lb] for v in b.values()]), axis=0)
            b = {k: v[:Lb] - gb for k, v in b.items()}
            separation(chip_of, b, "P3 per-block rank / no smid, numerical")

    print("\nInterpretation: separation ratio >2x AND LOO accuracy well above")
    print("chance in BOTH or either channel => the noise carries a per-instance")
    print("fingerprint. Near chance => it does not (architecture only).")


if __name__ == "__main__":
    main()
