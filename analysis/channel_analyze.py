"""Leakage boundary: tensor-core (TF32 matmul) and DVFS power/clock channels.

Input: experiments/channel_probe.py outputs in data/channels (3 A40 chips).
  tc   : run-to-run nondeterminism of a tall-skinny TF32 GEMM; if any, LOO on
         the per-element ULP residual
  dvfs : steady-state clock and power under a fixed 30 s matmul load, spread
         across chips vs within-chip jitter

Usage: python analysis/channel_analyze.py
"""
import glob, json, os
import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

def load(pat):
    out=[]
    for p in sorted(glob.glob(os.path.join(DATA, "channels", pat))):
        if os.path.getsize(p)==0: continue
        try: d=json.load(open(p))
        except: continue
        d["_u"]=d["identity"]["uuid"]; out.append(d)
    return out

def loo(vecs,labs,name):
    L=min(len(v) for v in vecs); X=np.vstack([v[:L] for v in vecs]); R=X-X.mean(0)
    lab=np.array(labs); chips=sorted(set(labs))
    dfn=lambda a,b:np.linalg.norm(a-b)
    win=[dfn(R[i],R[j]) for i in range(len(R)) for j in range(i+1,len(R)) if lab[i]==lab[j]]
    bet=[dfn(R[i],R[j]) for i in range(len(R)) for j in range(i+1,len(R)) if lab[i]!=lab[j]]
    ok=0
    for i in range(len(R)):
        cen={u:np.mean([R[k] for k in range(len(R)) if lab[k]==u and k!=i],0) for u in chips
             if sum(1 for k in range(len(R)) if lab[k]==u and k!=i)>0}
        if cen and min(cen,key=lambda u:dfn(R[i],cen[u]))==lab[i]: ok+=1
    w=np.mean(win) if win else float('nan'); b=np.mean(bet) if bet else float('nan')
    print("    [%s] within %.4g between %.4g sep %.2fx  LOO %d/%d (chance %.0f%%)"
          %(name,w,b,b/w if w else float('nan'),ok,len(R),100/len(chips)))

def main():
    print("="*70); print("NON-ATOMIC CHANNELS: tensor-core matmul and DVFS"); print("="*70)
    tc=load("ch_tc_*.json")
    if tc:
        print("\nB1 Tensor-Core / TF32 split-K accumulation (%d runs, %d chips):"%(len(tc),len(set(d['_u'] for d in tc))))
        for d in tc[:6]:
            print("   %s..  nondet_frac=%.3f distinct_med=%.0f  (%.2fs)"%(d["_u"][:16],d["nondet_frac"],d["distinct_med"],d["secs"]))
        if any(d["nondet_frac"]>0 for d in tc):
            loo([np.array(d["fp_resid_mean"]) for d in tc],[d["_u"] for d in tc],"TC residual")
        else:
            print("   -> tensor-core matmul was DETERMINISTIC (no split-K atomics) -> no leak this shape")
    dv=load("ch_dvfs_*.json")
    if dv:
        print("\nB2 DVFS power/clock silicon-quality (%d chips):"%len(set(d['_u'] for d in dv)))
        print("   %-18s %8s %8s %8s %8s"%("chip","clock","clk_std","power","pwr_std"))
        for d in dv:
            print("   %-18s %8.1f %8.2f %8.1f %8.2f"%(d["_u"][:16],d["steady_clock"],d["clock_std"],d["steady_power"],d["power_std"]))
        clks=np.array([d["steady_clock"] for d in dv]); pwrs=np.array([d["steady_power"] for d in dv])
        cstd=np.mean([d["clock_std"] for d in dv]); pstd=np.mean([d["power_std"] for d in dv])
        print("   clock spread across chips: %.1f MHz (within-chip jitter %.2f) -> %s"
              %(clks.max()-clks.min(),cstd,"SEPARABLE" if (clks.max()-clks.min())>3*cstd else "not separable"))
        print("   power spread across chips: %.1f W  (within-chip jitter %.2f) -> %s"
              %(pwrs.max()-pwrs.min(),pstd,"SEPARABLE" if (pwrs.max()-pwrs.min())>3*pstd else "not separable"))

if __name__=="__main__":
    main()
