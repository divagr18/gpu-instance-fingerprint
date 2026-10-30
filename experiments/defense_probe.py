"""Defense: does torch deterministic mode kill the passive channel?

Runs the sparse, order-dominated index_add signature (probes/passive_index_add.py)
in normal vs torch-deterministic mode. If deterministic mode forces a fixed
reduction order, run-to-run nondeterminism vanishes (det_frac -> 1.0) and the
signature becomes identical across chips. Cross-chip comparison is done in
analysis/defense_analyze.py.

Usage: python defense_probe.py <replicate> <deterministic 0|1> > out.json
"""
import json, os, subprocess, sys
import numpy as np
import torch

K, PER, R = 65536, 4, 200


def identity():
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=uuid,name,driver_version",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=8)
        f = [x.strip() for x in q.stdout.strip().splitlines()[0].split(",")]
        return dict(uuid=f[0], name=f[1], driver=f[2])
    except Exception as e:
        return dict(error=str(e))


def main():
    rep = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    determ = len(sys.argv) > 2 and sys.argv[2] == "1"
    if determ:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
    dev = "cuda"
    N = K * PER
    g = torch.Generator(device=dev).manual_seed(20260831)
    idx = torch.arange(K, device=dev).repeat_interleave(PER)
    idx = idx[torch.randperm(N, generator=g, device=dev)]
    e = torch.rand(N, generator=g, device=dev) * 60 - 30
    s = torch.where(torch.rand(N, generator=g, device=dev) > 0.5, 1.0, -1.0)
    src = (s * torch.exp2(e)).float()
    base = torch.zeros(K, dtype=torch.float64, device=dev)
    base.index_add_(0, idx, src.double())
    base_np = base.cpu().numpy()
    ulp = np.array([np.spacing(np.float32(abs(v) or 1.0)) for v in base_np])
    acc = np.zeros((R, K)); outs_i = np.zeros((R, K), dtype=np.int32)
    err = ""
    for r in range(R):
        out = torch.zeros(K, device=dev)
        try:
            out.index_add_(0, idx, src)
        except RuntimeError as ex:
            err = str(ex)[:120]; break
        acc[r] = out.double().cpu().numpy()
        outs_i[r] = out.view(torch.int32).cpu().numpy()
    resid = (acc - base_np[None]) / ulp[None]
    distinct = np.array([len(np.unique(outs_i[:, k])) for k in range(K)])
    fp_sign = np.sign(resid.mean(0))
    print(json.dumps(dict(identity=identity(), replicate=rep, deterministic=determ, err=err,
                          det_frac=float((distinct == 1).mean()),
                          fp_sign=[int(x) for x in fp_sign])))


if __name__ == "__main__":
    main()
