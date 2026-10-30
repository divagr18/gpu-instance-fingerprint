"""Leakage boundary: does a vanilla transformer carry the instance fingerprint?

Input: experiments/transformer_leak.py outputs in data/transformer_leak
(3 A40 chips, 2 normal-mode runs each + 1 deterministic-mode run each).
Reports run-to-run nondeterminism of forward logits and the embedding
gradient, cross-chip LOO separability, and the wall-time cost of
torch.use_deterministic_algorithms(True).

Usage: python analysis/leak_analyze.py
"""
import glob, json, os
import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

def load(pat):
    out=[]
    for p in sorted(glob.glob(os.path.join(DATA, "transformer_leak", pat))):
        if os.path.getsize(p)==0: continue
        try: d=json.load(open(p))
        except: continue
        d["_u"]=d["identity"]["uuid"]; d["_p"]=p; out.append(d)
    return out

def loo(vecs, labs, name, chance_note=""):
    L=min(len(v) for v in vecs); X=np.vstack([v[:L] for v in vecs])
    g=X.mean(0); R=X-g; lab=np.array(labs); chips=sorted(set(labs))
    def d(a,b): return np.linalg.norm(a-b)
    win=[d(R[i],R[j]) for i in range(len(R)) for j in range(i+1,len(R)) if lab[i]==lab[j]]
    bet=[d(R[i],R[j]) for i in range(len(R)) for j in range(i+1,len(R)) if lab[i]!=lab[j]]
    ok=0
    for i in range(len(R)):
        cen={u:np.mean([R[k] for k in range(len(R)) if lab[k]==u and k!=i],0) for u in chips
             if sum(1 for k in range(len(R)) if lab[k]==u and k!=i)>0}
        if cen and min(cen,key=lambda u:d(R[i],cen[u]))==lab[i]: ok+=1
    w=np.mean(win) if win else float('nan'); b=np.mean(bet) if bet else float('nan')
    print("    [%s] within %.4g between %.4g sep %.2fx  LOO %d/%d (chance %.0f%%) %s"
          %(name,w,b,b/w if w else float('nan'),ok,len(R),100/len(chips),chance_note))

def main():
    norm=load("tl_*_r*.json"); det=load("tl_*_det.json")
    chips=sorted(set(d["_u"] for d in norm))
    print("="*70); print("TRANSFORMER LEAKAGE  (%d chips, %s)"%(len(chips), norm[0]["identity"]["name"] if norm else "?")); print("="*70)
    print("\nNondeterminism (fraction of output elements varying run-to-run):")
    for d in norm:
        print("  %s..  FWD nondet=%.3f (distinct med %.0f)   BWD nondet=%.3f (distinct med %.0f)"
              %(d["_u"][:16], d["fwd_nondet_frac"], d["fwd_distinct_med"], d["bwd_nondet_frac"], d["bwd_distinct_med"]))
    print("\nCross-chip separability (arch-subtracted, LOO):")
    loo([np.array(d["fwd_resid_mean"]) for d in norm],[d["_u"] for d in norm],"FORWARD logits")
    loo([np.array(d["bwd_grad_mean"]) for d in norm],[d["_u"] for d in norm],"BACKWARD gradient")

    if det:
        print("\n--- deterministic mode ---")
        for d in det:
            print("  %s..  FWD nondet=%.3f  BWD nondet=%.3f"%(d["_u"][:16],d["fwd_nondet_frac"],d["bwd_nondet_frac"]))
        # throughput cost
        nn={d["_u"]:d for d in norm if d["replicate"]==0}
        print("\n  throughput cost (deterministic vs normal):")
        for d in det:
            n=nn.get(d["_u"])
            if n:
                print("    %s..  fwd %.3fs->%.3fs (%.2fx)  bwd %.3fs->%.3fs (%.2fx)"
                      %(d["_u"][:12], n["fwd_secs"], d["fwd_secs"], d["fwd_secs"]/max(n["fwd_secs"],1e-9),
                        n["bwd_secs"], d["bwd_secs"], d["bwd_secs"]/max(n["bwd_secs"],1e-9)))

if __name__=="__main__":
    main()
