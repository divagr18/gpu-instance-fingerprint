"""Active GPU instance fingerprint probe.

Measures the per-instance bias in a GPU's atomic-contention order. Thread
blocks race to a global atomicAdd; the order in which they arrive is decided by
the hardware, and per-SM manufacturing speed skew biases that order in a way
that is stable for a given physical chip. Subtracting the cross-chip mean
removes the architecture and leaves a per-chip residual that separates
physically identical same-model GPUs.

Three channels from one probe:

  P1  marginal sum  - histogram over the float32 result of a fixed-input
                      parallel reduction (weak channel).
  P2  per-SM rank   - per-physical-SM mean arrival rank at the global atomicAdd,
                      binned by the %smid register.
  P3  per-block rank- mean arrival slot per block index. Uses only the
                      numerical output of the kernel; no %smid, no timing.

The JSON also records GPU UUID, driver, CUDA, cupy version, and per-epoch clock
and temperature, so analyses can label chips and check thermal sensitivity.

Usage: python probe_puf.py [epochs=4] [replicate=0] > out.json
"""
import json
import subprocess
import sys
import time

import numpy as np
import cupy as cp

SRC = r"""
extern "C" __global__
void reduce_atomic(const float* __restrict__ x, int n, float* out){
    extern __shared__ float s[];
    int tid = threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    float v = 0.0f;
    for(int i = blockIdx.x*blockDim.x + tid; i < n; i += stride) v += x[i];
    s[tid] = v; __syncthreads();
    for(int st = blockDim.x/2; st > 0; st >>= 1){
        if(tid < st) s[tid] += s[tid+st];
        __syncthreads();
    }
    if(tid == 0) atomicAdd(out, s[0]);   // arrival order = hardware-decided
}

extern "C" __global__
void order_probe(int G, int burn, int* counter, int* out_sm, int* out_slot,
                 float* sink){
    unsigned smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    float acc = 1.0f + (float)(threadIdx.x & 7);
    for(int k = 0; k < burn; k++) acc = fmaf(acc, 1.0000001f, 1e-6f);
    if(threadIdx.x == 0){
        int slot = atomicAdd(counter, 1);   // fast SMs finish burn first
        out_sm[blockIdx.x] = (int)smid;
        out_slot[blockIdx.x] = slot;
        sink[blockIdx.x] = acc;             // keep the burn from being optimised out
    }
}
"""

MOD = cp.RawModule(code=SRC, options=("-std=c++11",))
K_RED = MOD.get_function("reduce_atomic")
K_ORD = MOD.get_function("order_probe")

N = 1 << 16
BLOCKS, THREADS = 256, 256
GORD, TORD = 4096, 128
P1_RUNS = 6000
P2_RUNS = 3000
BURN = 15000


def make_summands(kind, n, seed):
    rng = cp.random.RandomState(seed)
    u = rng.rand(n, dtype=cp.float32)
    if kind == "heavy":
        e = rng.rand(n, dtype=cp.float32) * 40.0 - 20.0
        s = cp.where(rng.rand(n, dtype=cp.float32) > 0.5, 1.0, -1.0).astype(cp.float32)
        return (s * u * cp.exp2(e)).astype(cp.float32)
    return u


def smi():
    try:
        q = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=uuid,name,driver_version,temperature.gpu,clocks.sm,"
             "clocks.max.sm,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8)
        f = [x.strip() for x in q.stdout.strip().splitlines()[0].split(",")]
        return dict(uuid=f[0], name=f[1], driver=f[2], temp=int(f[3]),
                    clock=int(f[4]), clock_max=int(f[5]), power=float(f[6]))
    except Exception as e:
        return dict(error=str(e))


def p1_numerical(inputs, runs):
    """Histogram over the float32 sum result, per fixed input. Returns, for each
    input, a {uint32_bitpattern: count} dict - the numerical-bias signature."""
    out = cp.zeros(runs, dtype=cp.float32)
    shmem = THREADS * 4
    per_input = []
    for x in inputs:
        for i in range(runs):
            K_RED((BLOCKS,), (THREADS,), (x, np.int32(N), out[i:i + 1]),
                  shared_mem=shmem)
        cp.cuda.Stream.null.synchronize()
        v = cp.asnumpy(out).view(np.uint32)
        vals, cnts = np.unique(v, return_counts=True)
        per_input.append({int(a): int(b) for a, b in zip(vals, cnts)})
    return per_input


def p2_ordering(runs, burn=4000):
    """Per-SM arrival-rank signature. For each physical SM, the mean normalised
    slot at which its blocks reach the global atomicAdd. A systematically faster
    SM (manufacturing skew) gets systematically lower ranks."""
    counter = cp.zeros(1, dtype=cp.int32)
    out_sm = cp.zeros(GORD, dtype=cp.int32)
    out_slot = cp.zeros(GORD, dtype=cp.int32)
    sink = cp.zeros(GORD, dtype=cp.float32)
    MAXSM = 256
    ssum = np.zeros(MAXSM, dtype=np.float64)
    scnt = np.zeros(MAXSM, dtype=np.int64)
    bsum = np.zeros(GORD, dtype=np.float64)   # per-block-index arrival slot
    bcnt = np.zeros(GORD, dtype=np.int64)
    for _ in range(runs):
        counter[:] = 0
        K_ORD((GORD,), (TORD,),
              (np.int32(GORD), np.int32(burn), counter, out_sm, out_slot, sink))
        cp.cuda.Stream.null.synchronize()
        sm = cp.asnumpy(out_sm)
        slot = cp.asnumpy(out_slot).astype(np.float64) / (GORD - 1)
        np.add.at(ssum, sm, slot)
        np.add.at(scnt, sm, 1)
        bsum += slot
        bcnt += 1
    active = scnt > 0
    mean_rank = np.full(MAXSM, np.nan)
    mean_rank[active] = ssum[active] / scnt[active]
    n_sm = int(active.sum())
    # P3: mean arrival slot per BLOCK INDEX (no %smid). CUDA's work distributor
    # maps blockIdx->SM in a near-fixed pattern, so this recovers the same
    # per-SM speed skew purely from which block lands where, readable from
    # the kernel's numerical output alone.
    block_mean = (bsum / np.maximum(bcnt, 1))
    return dict(n_sm=n_sm,
                sm_mean_rank=[None if np.isnan(x) else round(float(x), 6)
                              for x in mean_rank[:n_sm + 4]],
                sm_block_count=[int(x) for x in scnt[:n_sm + 4]],
                block_mean_rank=[round(float(x), 6) for x in block_mean])


def main():
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    replicate = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    p = cp.cuda.runtime.getDeviceProperties(0)
    ident = smi()
    ident.update(cc="%d.%d" % (p["major"], p["minor"]),
                 sm_count=p["multiProcessorCount"],
                 cupy=cp.__version__,
                 driver_api=cp.cuda.runtime.driverGetVersion())

    # fixed inputs: identical bit-for-bit on every chip
    inputs = [make_summands("heavy", N, seed=s) for s in (1234, 99, 7, 2024)]

    result = dict(identity=ident, replicate=replicate, epochs=[])
    for e in range(epochs):
        t0 = time.perf_counter()
        pre = smi()
        p1 = p1_numerical(inputs, runs=P1_RUNS)
        p2 = p2_ordering(runs=P2_RUNS, burn=BURN)
        post = smi()
        result["epochs"].append(dict(
            epoch=e, secs=round(time.perf_counter() - t0, 1),
            temp_pre=pre.get("temp"), temp_post=post.get("temp"),
            clock_pre=pre.get("clock"), clock_post=post.get("clock"),
            power=post.get("power"), p1=p1, p2=p2))
        sys.stderr.write("  epoch %d done  %.1fs  temp %s->%s  clock %s->%s\n"
                         % (e, result["epochs"][-1]["secs"], pre.get("temp"),
                            post.get("temp"), pre.get("clock"), post.get("clock")))
        sys.stderr.flush()

    print(json.dumps(result))


if __name__ == "__main__":
    main()
