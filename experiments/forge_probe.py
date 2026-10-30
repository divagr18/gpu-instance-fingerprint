"""Forgery test: can chip A be made to emit chip B's fingerprint?

Threat model. A verifier enrolls chip B's per-physical-SM arrival-rank signature
(the %smid channel: each SM's mean normalised slot at the global atomicAdd, biased
by that SM's manufacturing speed skew). An attacker running on a different chip A
wants its output to pass as B. The attacker cannot change silicon and cannot set
the per-bin atomic order deterministically, but does control the coarse launch
knobs that reshape the arrival-order distribution:
    - grid size  (number of blocks competing for the atomic)
    - block size (threads per block -> occupancy)
    - compute burn per thread before the atomic (shifts which SMs arrive first)
    - an injected per-thread spin delay (a deliberate timing perturbation)
This program sweeps that knob battery on each chip and records the resulting
%smid signature per config. analysis/forge_analyze.py then asks whether, over ALL
of attacker A's configs, the best match to B's enrolled signature ever reaches
B's own genuine reproducibility band.

The %smid signature is indexed by PHYSICAL SM id, so it is directly comparable
across different grid/block sizes (unlike per-block rank). Emits JSON on stdout.

Usage: python forge_probe.py [replicate=0] [runs=1500] > out.json
"""
import json
import subprocess
import sys
import time

import numpy as np
import cupy as cp

SRC = r"""
extern "C" __global__
void order_probe(int burn, int delay, int* counter, int* out_sm, int* out_slot,
                 float* sink){
    unsigned smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    float acc = 1.0f + (float)(threadIdx.x & 7);
    for(int k = 0; k < burn; k++) acc = fmaf(acc, 1.0000001f, 1e-6f);
    // optional injected spin delay (attacker's timing perturbation)
    for(int k = 0; k < delay; k++) acc = fmaf(acc, 0.9999999f, 1e-7f);
    if(threadIdx.x == 0){
        int slot = atomicAdd(counter, 1);   // fast SMs finish first
        out_sm[blockIdx.x] = (int)smid;
        out_slot[blockIdx.x] = slot;
        sink[blockIdx.x] = acc;
    }
}
"""
MOD = cp.RawModule(code=SRC, options=("-std=c++11",))
K_ORD = MOD.get_function("order_probe")
MAXSM = 512


def smi():
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid,name,driver_version,temperature.gpu,clocks.sm",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=8)
        f = [x.strip() for x in q.stdout.strip().splitlines()[0].split(",")]
        return dict(uuid=f[0], name=f[1], driver=f[2], temp=int(f[3]), clock=int(f[4]))
    except Exception as e:
        return dict(error=str(e))


def sm_signature(gord, tord, burn, delay, runs):
    """Per-physical-SM mean arrival rank under a given launch config."""
    counter = cp.zeros(1, dtype=cp.int32)
    out_sm = cp.zeros(gord, dtype=cp.int32)
    out_slot = cp.zeros(gord, dtype=cp.int32)
    sink = cp.zeros(gord, dtype=cp.float32)
    ssum = np.zeros(MAXSM); scnt = np.zeros(MAXSM)
    for _ in range(runs):
        counter[:] = 0
        K_ORD((gord,), (tord,),
              (np.int32(burn), np.int32(delay), counter, out_sm, out_slot, sink))
        cp.cuda.Stream.null.synchronize()
        sm = cp.asnumpy(out_sm)
        slot = cp.asnumpy(out_slot).astype(np.float64) / (gord - 1)
        np.add.at(ssum, sm, slot); np.add.at(scnt, sm, 1)
    active = scnt > 0
    rank = np.full(MAXSM, np.nan); rank[active] = ssum[active] / scnt[active]
    n = int(active.sum())
    return n, [None if np.isnan(x) else round(float(x), 6) for x in rank[:n]]


def main():
    rep = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
    p = cp.cuda.runtime.getDeviceProperties(0)
    ident = smi(); ident.update(sm_count=p["multiProcessorCount"], cupy=cp.__version__)

    # canonical enrollment config + the attacker's knob battery
    CANON = dict(gord=4096, tord=128, burn=15000, delay=0)
    battery = [CANON,
               dict(gord=1024, tord=128, burn=15000, delay=0),
               dict(gord=2048, tord=128, burn=15000, delay=0),
               dict(gord=8192, tord=128, burn=15000, delay=0),
               dict(gord=4096, tord=64,  burn=15000, delay=0),
               dict(gord=4096, tord=256, burn=15000, delay=0),
               dict(gord=4096, tord=128, burn=5000,  delay=0),
               dict(gord=4096, tord=128, burn=40000, delay=0),
               dict(gord=4096, tord=128, burn=15000, delay=8000),
               dict(gord=4096, tord=128, burn=15000, delay=30000)]

    configs = []
    for c in battery:
        t0 = time.perf_counter()
        n, sig = sm_signature(c["gord"], c["tord"], c["burn"], c["delay"], runs)
        configs.append(dict(cfg=c, n_sm=n, sm_rank=sig,
                            secs=round(time.perf_counter() - t0, 1)))
        sys.stderr.write("  cfg %s  n_sm=%d  %.1fs\n" % (c, n, configs[-1]["secs"]))
        sys.stderr.flush()

    print(json.dumps(dict(identity=ident, replicate=rep, canon=CANON, configs=configs)))


if __name__ == "__main__":
    main()
