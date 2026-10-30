"""Leakage boundary: two non-atomic candidate channels.

  tc    Tensor-Core / TF32 matmul. A tall-skinny GEMM (large K, small M,N) is
        the shape most likely to use split-K accumulation. Feature: per-element
        ULP residual vs float64 over R runs, plus run-to-run nondeterminism.

  dvfs  Power/clock under a fixed sustained matmul load. Each chip's steady
        boost clock and power draw could reflect its voltage/leakage binning.
        Feature: steady-state clock and power, plus their within-run jitter.

Fixed bit-identical inputs on every chip. Emits JSON.

Usage: python channel_probe.py tc   <replicate> > out.json
       python channel_probe.py dvfs <replicate> > out.json
"""
import json, subprocess, sys, time
import numpy as np
import torch


def identity():
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid,name,driver_version,temperature.gpu,clocks.sm",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=8)
        f = [x.strip() for x in q.stdout.strip().splitlines()[0].split(",")]
        return dict(uuid=f[0], name=f[1], driver=f[2], temp=int(f[3]), clock=int(f[4]))
    except Exception as e:
        return dict(error=str(e))


def smi_sample():
    q = subprocess.run(
        ["nvidia-smi", "--query-gpu=clocks.sm,power.draw,temperature.gpu",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=8)
    f = [x.strip() for x in q.stdout.strip().splitlines()[0].split(",")]
    return int(f[0]), float(f[1]), int(f[2])


def channel_tc(rep):
    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    M, N, K = 8, 8, 1 << 20        # tall-skinny -> split-K -> atomic accumulation
    R = 300
    g = torch.Generator(device=dev).manual_seed(2026)
    A = (torch.rand(M, K, generator=g, device=dev) * 2 - 1)
    B = (torch.rand(K, N, generator=g, device=dev) * 2 - 1)
    ref = (A.double() @ B.double()).cpu().numpy().ravel()       # float64 reference
    ulp = np.spacing(np.abs(ref).astype(np.float32)).astype(np.float64); ulp[ulp == 0] = 1e-7
    acc = np.zeros((R, M * N)); bits = np.zeros((R, M * N), dtype=np.int32)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for r in range(R):
        C = (A @ B).float().contiguous()
        acc[r] = C.double().cpu().numpy().ravel()
        bits[r] = C.view(torch.int32).cpu().numpy().ravel()
    torch.cuda.synchronize(); secs = time.perf_counter() - t0
    resid = (acc - ref[None]) / ulp[None]
    distinct = np.array([len(np.unique(bits[:, k])) for k in range(M * N)])
    return dict(channel="tc", M=M, N=N, K=K, R=R, secs=round(secs, 3),
                nondet_frac=float((distinct > 1).mean()), distinct_med=float(np.median(distinct)),
                fp_resid_mean=[round(float(x), 5) for x in resid.mean(0)],
                fp_resid_p=[round(float(x), 4) for x in (resid > 0).mean(0)])


def channel_dvfs(rep):
    dev = "cuda"
    n = 4096
    g = torch.Generator(device=dev).manual_seed(11)
    A = torch.rand(n, n, generator=g, device=dev)
    B = torch.rand(n, n, generator=g, device=dev)
    # warm up to boost, then sample under sustained fixed load
    for _ in range(20):
        C = A @ B
    torch.cuda.synchronize()
    traj = []
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 30.0:                      # 30s fixed load
        for _ in range(40):
            C = A @ B
        torch.cuda.synchronize()
        clk, pwr, tmp = smi_sample()
        traj.append((round(time.perf_counter() - t0, 2), clk, pwr, tmp))
    clks = np.array([x[1] for x in traj], float); pwrs = np.array([x[2] for x in traj], float)
    tail = max(1, len(traj) // 2)                               # steady-state = 2nd half
    return dict(channel="dvfs", n=n, secs=round(time.perf_counter() - t0, 1),
                steady_clock=round(float(clks[-tail:].mean()), 1),
                steady_power=round(float(pwrs[-tail:].mean()), 1),
                clock_std=round(float(clks[-tail:].std()), 2),
                power_std=round(float(pwrs[-tail:].std()), 2),
                traj=traj)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "tc"
    rep = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    r = channel_tc(rep) if mode == "tc" else channel_dvfs(rep)
    print(json.dumps(dict(identity=identity(), replicate=rep, **r)))


if __name__ == "__main__":
    main()
