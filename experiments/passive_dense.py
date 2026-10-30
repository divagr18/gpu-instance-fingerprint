"""Passive fingerprint attempt in the DENSE regime (negative control).

torch.index_add_ with many contributions per bin (2^20 contributions into 256
bins) and heavy-tailed magnitudes. The op is strongly nondeterministic, but
with this many contributions per bin the summation averages the arrival order
out, so the per-bin residual statistics carry no recoverable chip identity.
Compare probes/passive_index_add.py (sparse regime), which does.

Feature per output bin: statistics of the ULP-level residual (float32 sum minus
float64 reference) across R repeats. Fixed bit-identical input on every chip.

Usage: python passive_dense.py [replicate=0] > out.json
"""
import json
import subprocess
import sys

import numpy as np
import torch

N = 1 << 20        # contributions
K = 256            # output bins
R = 4000           # repeats


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
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(1234)          # FIXED across chips
    u = torch.rand(N, generator=g, device=dev)
    e = torch.rand(N, generator=g, device=dev) * 40.0 - 20.0
    s = torch.where(torch.rand(N, generator=g, device=dev) > 0.5, 1.0, -1.0)
    src = (s * u * torch.exp2(e)).float()                      # heavy-tailed -> max non-assoc
    idx = torch.randint(0, K, (N,), generator=g, device=dev)

    base = torch.zeros(K, dtype=torch.float64, device=dev)
    base.index_add_(0, idx, src.double())                      # float64 reference sum

    # R repeats of the real float32 op; collect ULP-level residual per bin
    resid = torch.empty(R, K, dtype=torch.float64, device=dev)
    outs_i = torch.empty(R, K, dtype=torch.int32, device=dev)  # raw bits, for distinctness
    for r in range(R):
        out = torch.zeros(K, device=dev)
        out.index_add_(0, idx, src)
        resid[r] = out.double() - base
        outs_i[r] = out.view(torch.int32)
    resid = resid.cpu().numpy()
    outs_i = outs_i.cpu().numpy()

    ulp = np.array([np.spacing(np.float32(abs(float(base[k].cpu())) or 1.0)) for k in range(K)])
    resid_ulp = resid / ulp[None, :]                           # residual in ULPs

    # nondeterminism check: distinct float32 results per bin across repeats
    distinct = [int(len(np.unique(outs_i[:, k]))) for k in range(K)]

    # fingerprint features from the plain output only:
    fp_mean = resid_ulp.mean(0)                                # systematic per-bin bias
    fp_std = resid_ulp.std(0)
    fp_p1 = (resid_ulp > 0).mean(0)                            # fraction of positive residual

    result = dict(
        identity=identity(), replicate=rep, N=N, K=K, R=R,
        distinct_median=float(np.median(distinct)),
        distinct_max=int(np.max(distinct)),
        fp_mean=[round(float(x), 5) for x in fp_mean],
        fp_std=[round(float(x), 5) for x in fp_std],
        fp_p1=[round(float(x), 5) for x in fp_p1],
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
