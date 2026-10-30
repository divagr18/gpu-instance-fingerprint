"""Defense: deterministic mode vs the passive index_add channel.

Input: experiments/defense_probe.py outputs in data/defense (2 A40 chips on
the same host, 2 runs each in normal mode and 2 in deterministic mode).

For each mode, reports:
  * det_frac  - fraction of output bins that are bit-identical run-to-run
  * genuine HD  - fractional Hamming distance between sign signatures of the
                  same chip (different runs)
  * impostor HD - same, between different chips
If deterministic mode works, det_frac -> 1.0 and genuine HD = impostor HD = 0:
every chip emits the same signature, so there is nothing to identify.

Usage: python analysis/defense_analyze.py
"""
import glob
import itertools
import json
import os

import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def main():
    runs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(DATA, "defense", "*.json")))]
    print("=" * 70)
    print("DEFENSE: torch.use_deterministic_algorithms vs passive index_add")
    print("=" * 70)
    for det, name in [(False, "NORMAL"), (True, "DETERMINISTIC")]:
        rs = [r for r in runs if r["deterministic"] == det]
        errs = sorted(set(r["err"] for r in rs if r["err"]))
        S = [np.array(r["fp_sign"]) for r in rs]
        lab = [r["identity"]["uuid"] for r in rs]
        gen, imp = [], []
        for i, j in itertools.combinations(range(len(rs)), 2):
            (gen if lab[i] == lab[j] else imp).append(float(np.mean(S[i] != S[j])))
        print("\n  %s mode: %d runs, %d chips%s" % (name, len(rs), len(set(lab)),
                                                  "  errors: %s" % errs if errs else ""))
        print("    det_frac (bins bit-identical run-to-run): %.4f"
              % np.mean([r["det_frac"] for r in rs]))
        print("    genuine HD  (same chip)      : %.5f  (n=%d)" % (np.mean(gen), len(gen)))
        print("    impostor HD (different chips): %.5f  (n=%d)" % (np.mean(imp), len(imp)))
        print("    signatures identical across all runs and chips: %s"
              % all(np.array_equal(S[0], s) for s in S))


if __name__ == "__main__":
    main()
