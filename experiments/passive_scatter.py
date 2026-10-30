"""Passive fingerprint from order-preserving real ops.

  V1 scatter-overwrite: out.scatter_(0, idx, contrib_id). Duplicate indices race;
     the last writer wins, so out[k] is the id of whichever contribution wrote
     bin k last, i.e. arrival order lands directly in the output tensor.
     Feature: per-bin mean/std/quantiles of the winning id over many repeats.

  V2 index_add with FEW contributions per bin + extreme magnitudes: each bin
     sums a handful of wildly different values, so the sum is dominated by add
     order (an earlier, smaller version of probes/passive_index_add.py).

Fixed bit-identical input on every chip. Only the normal output tensor is read.

Usage: python passive_scatter.py [replicate=0] > out.json
"""
import json
import subprocess
import sys

import numpy as np
import torch

R = 4000


def identity():
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid,name,driver_version,temperature.gpu,clocks.sm",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=8)
        f = [x.strip() for x in q.stdout.strip().splitlines()[0].split(",")]
        return dict(uuid=f[0], name=f[1], driver=f[2], temp=int(f[3]), clock=int(f[4]))
    except Exception as e:
        return dict(error=str(e))


def v1_scatter(dev):
    """scatter-overwrite winner-id per bin. Big N per bin -> many racers."""
    N = 1 << 21
    K = 1024
    g = torch.Generator(device=dev).manual_seed(1234)
    idx = torch.randint(0, K, (N,), generator=g, device=dev)
    contrib = torch.arange(N, device=dev, dtype=torch.float32)   # id of each contribution
    acc = np.zeros((R, K), dtype=np.float64)
    distinct = None
    raws = []
    for r in range(R):
        out = torch.full((K,), -1.0, device=dev)
        out.scatter_(0, idx, contrib)                            # race: last writer wins
        acc[r] = out.cpu().numpy()
    winner = acc  # [R,K] winning contribution id per bin
    distinct = [int(len(np.unique(winner[:, k]))) for k in range(0, K, 8)]
    # normalize each bin's winner to a rank in [0,1] within that bin's contribution set
    # (bin k's contributions are the fixed set {i: idx[i]==k}; use global-id/N as proxy)
    norm = winner / N
    return dict(
        op="scatter_overwrite", N=N, K=K,
        distinct_median=float(np.median(distinct)), distinct_max=int(np.max(distinct)),
        fp_mean=[round(float(x), 6) for x in norm.mean(0)],
        fp_std=[round(float(x), 6) for x in norm.std(0)],
        fp_q25=[round(float(x), 6) for x in np.quantile(norm, 0.25, axis=0)],
        fp_q75=[round(float(x), 6) for x in np.quantile(norm, 0.75, axis=0)],
    )


def v2_index_add_sparse(dev):
    """index_add with FEW contributions per bin + extreme magnitudes."""
    K = 2048
    per = 6
    N = K * per
    g = torch.Generator(device=dev).manual_seed(4321)
    idx = torch.arange(K, device=dev).repeat_interleave(per)      # exactly `per` per bin
    idx = idx[torch.randperm(N, generator=g, device=dev)]         # shuffle processing order
    e = torch.rand(N, generator=g, device=dev) * 50 - 25
    s = torch.where(torch.rand(N, generator=g, device=dev) > 0.5, 1.0, -1.0)
    src = (s * torch.exp2(e)).float()
    base = torch.zeros(K, dtype=torch.float64, device=dev)
    base.index_add_(0, idx, src.double())
    acc = np.zeros((R, K), dtype=np.float64)
    outs_i = np.zeros((R, K), dtype=np.int32)
    for r in range(R):
        out = torch.zeros(K, device=dev)
        out.index_add_(0, idx, src)
        acc[r] = out.double().cpu().numpy()
        outs_i[r] = out.view(torch.int32).cpu().numpy()
    ulp = np.array([np.spacing(np.float32(abs(float(base[k].cpu())) or 1.0)) for k in range(K)])
    resid = (acc - base.cpu().numpy()) / ulp[None, :]
    distinct = [int(len(np.unique(outs_i[:, k]))) for k in range(0, K, 16)]
    return dict(
        op="index_add_sparse", N=N, K=K, per=per,
        distinct_median=float(np.median(distinct)), distinct_max=int(np.max(distinct)),
        fp_mean=[round(float(x), 5) for x in resid.mean(0)],
        fp_p1=[round(float(x), 5) for x in (resid > 0).mean(0)],
    )


def main():
    rep = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    dev = "cuda"
    print(json.dumps(dict(identity=identity(), replicate=rep,
                          v1=v1_scatter(dev), v2=v2_index_add_sparse(dev))))


if __name__ == "__main__":
    main()
