"""Forgery analysis: can attacker chip A reproduce target chip B's per-SM
signature by sweeping the launch-config knobs it controls?

Input: experiments/forge_probe.py outputs in data/forge (3 A40 chips x 2 runs,
10 launch configs each). Signature = per-physical-SM mean arrival rank
(indexed by smid, comparable across configs), z-scored per vector.
Similarity = Pearson correlation.

  genuine (run-to-run)     : corr(chip X canonical run 0, chip X canonical run 1)
                             -> the band an enrolled chip reproduces itself in
  genuine (cross-config)   : corr(chip X canonical, chip X under another config)
  impostor (natural)       : corr(chip A canonical, chip B canonical)
  FORGE (adversarial best) : max over ALL of attacker A's swept configs of
                             corr(A_config, B canonical)

The attack fails if the best forge stays below the genuine run-to-run floor.

Usage: python analysis/forge_analyze.py
"""
import glob, json, os
import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
CANON = (4096, 128, 15000, 0)          # (grid, block, burn, delay) used for enrollment


def load():
    runs = []
    for p in sorted(glob.glob(os.path.join(DATA, "forge", "*.json"))):
        if os.path.getsize(p) == 0:
            continue
        d = json.load(open(p))
        d["_uuid"] = d["identity"]["uuid"]
        runs.append(d)
    return runs


def z(v):
    v = np.array([np.nan if x is None else x for x in v], float)
    return (v - np.nanmean(v)) / (np.nanstd(v) + 1e-12)


def corr(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 4:
        return np.nan
    return np.corrcoef(a[m], b[m])[0, 1]


def cfgkey(c):
    return (c["gord"], c["tord"], c["burn"], c["delay"])


def main():
    runs = load()
    if not runs:
        print("no forge data")
        return
    chips = sorted(set(r["_uuid"] for r in runs))
    print("=" * 70)
    print("FORGERY TEST: %d A40 chips, %d runs" % (len(chips), len(runs)))
    print("=" * 70)

    # per chip: list of (cfg, z-scored signature) over all runs
    per = {u: [] for u in chips}
    for r in runs:
        for c in r["configs"]:
            per[r["_uuid"]].append((cfgkey(c["cfg"]), z(c["sm_rank"])))

    # enrolled vector per chip = mean of its canonical-config signatures
    cvec = {}
    for u in chips:
        cs = [s for k, s in per[u] if k == CANON]
        L = min(len(s) for s in cs)
        cvec[u] = np.nanmean([s[:L] for s in cs], 0)
    L = min(len(v) for v in cvec.values())
    cvec = {u: v[:L] for u, v in cvec.items()}

    # genuine run-to-run: same chip, canonical config, different runs
    rep = []
    for u in chips:
        cs = [s for k, s in per[u] if k == CANON]
        for a in range(len(cs)):
            for b in range(a + 1, len(cs)):
                Lc = min(len(cs[a]), len(cs[b]))
                rep.append(corr(cs[a][:Lc], cs[b][:Lc]))
    # genuine cross-config: same chip, non-canonical config vs its enrolled vector
    gen = [corr(s[:L], cvec[u]) for u in chips for k, s in per[u] if k != CANON and len(s) >= L]
    # impostor: enrolled A vs enrolled B
    imp = [corr(cvec[a], cvec[b]) for a in chips for b in chips if a != b]
    # forge: attacker A's best config against target B's enrolled vector
    forge, detail = [], []
    for A in chips:
        for B in chips:
            if A == B:
                continue
            best = max(corr(s[:L], cvec[B]) for k, s in per[A] if len(s) >= L)
            forge.append(best)
            detail.append((A[:12], B[:12], best))

    rep, gen, imp, forge = map(np.array, (rep, gen, imp, forge))
    print("\n  GENUINE run-to-run corr        : %.3f  (floor %.3f, n=%d)" % (rep.mean(), rep.min(), len(rep)))
    print("  GENUINE cross-config corr      : %.3f +/- %.3f  (n=%d)" % (gen.mean(), gen.std(), len(gen)))
    print("  IMPOSTOR natural corr          : %.3f +/- %.3f  (n=%d)" % (imp.mean(), imp.std(), len(imp)))
    print("  FORGE best-over-configs corr   : %.3f +/- %.3f  (n=%d)" % (forge.mean(), forge.std(), len(forge)))
    print("     (best single forge attempt : %.3f)" % forge.max())
    print("\n  gap [genuine floor - best forge] = %.3f" % (rep.min() - forge.max()))
    if forge.max() < rep.min():
        print("  => the attacker's best knob-tuned attempt never reaches the genuine floor.")
    print("\n  forge attempts (A tries to look like B):")
    for a, b, c in sorted(detail, key=lambda x: -x[2]):
        print("     %s -> %s : best corr %.3f" % (a, b, c))


if __name__ == "__main__":
    main()
