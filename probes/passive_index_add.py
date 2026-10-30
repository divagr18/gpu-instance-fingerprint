"""Passive GPU instance fingerprint from plain torch.index_add_ output.

torch.index_add_ on CUDA accumulates with atomics, so the order in which
contributions land in each output bin is decided by the hardware. With FEW
contributions per bin at EXTREME magnitudes (2^-30..2^30), the float32 sum is
dominated by that add order (catastrophic cancellation), so each chip's stable
atomic ordering produces a stable, chip-specific output. Only the ordinary
output tensor is read: no %smid, no timing, no instrumentation.

Signature: per-bin sign of the mean ULP residual (float32 result minus a
float64 reference) over R repeats -> a K-bit per-chip signature. The
real-valued residual vector is saved as well.

Usage: python passive_index_add.py [replicate=0] [K=65536] > out.json
  K=4096  -> R=300 repeats (data/passive_k4096)
  K=65536 -> R=400 repeats (data/passive_k65536, data/rtx4090, data/cross_model)
"""
import json
import subprocess
import sys

import numpy as np
import torch

PER = 4            # contributions per bin (few -> order-dominated sum)


def identity():
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid,name,driver_version,temperature.gpu,clocks.sm",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=8)
        f = [x.strip() for x in q.stdout.strip().splitlines()[0].split(",")]
        return dict(uuid=f[0], name=f[1], driver=f[2], temp=int(f[3]), clock=int(f[4]))
    except Exception as e:
        return dict(error=str(e))


def main():
    rep = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 65536      # output bins = signature bits
    R = 300 if K <= 4096 else 400                               # repeats
    dev = "cuda"
    N = K * PER
    g = torch.Generator(device=dev).manual_seed(20260831)          # fixed on every chip
    idx = torch.arange(K, device=dev).repeat_interleave(PER)
    idx = idx[torch.randperm(N, generator=g, device=dev)]          # shuffled processing order
    e = torch.rand(N, generator=g, device=dev) * 60 - 30           # magnitudes 2^-30..2^30
    s = torch.where(torch.rand(N, generator=g, device=dev) > 0.5, 1.0, -1.0)
    src = (s * torch.exp2(e)).float()

    base = torch.zeros(K, dtype=torch.float64, device=dev)
    base.index_add_(0, idx, src.double())
    base_np = base.cpu().numpy()
    ulp = np.array([np.spacing(np.float32(abs(v) or 1.0)) for v in base_np])

    acc = np.zeros((R, K), dtype=np.float64)
    outs_i = np.zeros((R, K), dtype=np.int32)
    for r in range(R):
        out = torch.zeros(K, device=dev)
        out.index_add_(0, idx, src)
        acc[r] = out.double().cpu().numpy()
        outs_i[r] = out.view(torch.int32).cpu().numpy()

    resid = (acc - base_np[None, :]) / ulp[None, :]                # ULP residual per bin
    distinct = np.array([len(np.unique(outs_i[:, k])) for k in range(K)])
    fp_resid = resid.mean(0)                                       # per-bin mean residual
    fp_sign = np.sign(fp_resid)                                    # K-bit signature

    print(json.dumps(dict(
        identity=identity(), replicate=rep, K=K, PER=PER, R=R,
        det_frac=float((distinct == 1).mean()),                   # fraction of bins deterministic
        distinct_median=float(np.median(distinct)), distinct_max=int(distinct.max()),
        fp_resid=[round(float(x), 5) for x in fp_resid],
        fp_sign=[int(x) for x in fp_sign],
    )))


if __name__ == "__main__":
    main()
