"""Leakage boundary: does a vanilla transformer leak the GPU instance fingerprint?

A 6-layer d512 GPT-style model with fixed weights and input is run R times:
  - forward: last-token logits, ULP residual vs a float64 reference, and
    run-to-run nondeterminism
  - backward: gradient of the embedding table (index_add-based), and
    run-to-run nondeterminism
If both are bit-identical run-to-run and across chips, ordinary inference and
training of this model carry no per-chip signal. Also records forward/backward
wall time so the cost of deterministic mode can be measured.

Usage: python transformer_leak.py <replicate> <deterministic 0|1> > out.json
"""
import json
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

D_MODEL = 512
N_LAYER = 6
N_HEAD = 8
SEQ = 128
VOCAB = 8192
R = 200            # repeats to expose nondeterminism / estimate the systematic bias


def identity():
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid,name,driver_version,temperature.gpu,clocks.sm",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=8)
        f = [x.strip() for x in q.stdout.strip().splitlines()[0].split(",")]
        return dict(uuid=f[0], name=f[1], driver=f[2], temp=int(f[3]), clock=int(f[4]))
    except Exception as e:
        return dict(error=str(e))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.attn = nn.MultiheadAttention(D_MODEL, N_HEAD, batch_first=True)
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.mlp = nn.Sequential(nn.Linear(D_MODEL, 4 * D_MODEL), nn.GELU(),
                                 nn.Linear(4 * D_MODEL, D_MODEL))

    def forward(self, x, mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, D_MODEL)
        self.pos = nn.Embedding(SEQ, D_MODEL)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYER)])
        self.lnf = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, VOCAB, bias=False)

    def forward(self, idx):
        b, t = idx.shape
        pos = torch.arange(t, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None]
        mask = torch.triu(torch.full((t, t), float("-inf"), device=idx.device), 1)
        for blk in self.blocks:
            x = blk(x, mask)
        return self.head(self.lnf(x))


def ulp_resid(obs, ref):
    """per-element residual in ULPs of the float64 reference."""
    ref = ref.astype(np.float64)
    ulp = np.spacing(np.abs(ref).astype(np.float32)).astype(np.float64)
    ulp[ulp == 0] = np.spacing(np.float32(1.0))
    return (obs.astype(np.float64) - ref) / ulp


def main():
    rep = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    determ = len(sys.argv) > 2 and sys.argv[2] == "1"
    if determ:
        torch.use_deterministic_algorithms(True, warn_only=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(20260901)
    model = TinyGPT().to(dev).eval()
    for p in model.parameters():          # fixed weights, bit-identical everywhere
        p.requires_grad_(True)
    g = torch.Generator(device="cpu").manual_seed(777)
    idx = torch.randint(0, VOCAB, (2, SEQ), generator=g).to(dev)
    tgt = torch.randint(0, VOCAB, (2, SEQ), generator=g).to(dev)

    # float64 reference (CPU, deterministic) for the LAST-token logits
    with torch.no_grad():
        m64 = model.double()
        ref_logits = m64(idx)[:, -1, :].detach().cpu().numpy()   # (2, VOCAB)
        model.float()

    # ---- FORWARD: collect last-token logits over R repeats ----
    fwd = np.zeros((R, 2 * VOCAB), dtype=np.float64)
    fwd_bits = np.zeros((R, 2 * VOCAB), dtype=np.int32)
    if dev == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for r in range(R):
            out = model(idx)[:, -1, :].contiguous()
            fwd[r] = out.double().cpu().numpy().ravel()
            fwd_bits[r] = out.view(torch.int32).cpu().numpy().ravel()
    if dev == "cuda":
        torch.cuda.synchronize()
    fwd_secs = time.perf_counter() - t0
    fwd_resid = ulp_resid(fwd, np.tile(ref_logits.ravel(), (R, 1)))
    fwd_distinct = np.array([len(np.unique(fwd_bits[:, k])) for k in range(fwd_bits.shape[1])])

    # ---- BACKWARD: gradient of the embedding table (index_add -> atomics) ----
    # collect the grad rows that actually receive contributions (the used token ids)
    used = torch.unique(idx).cpu().numpy()
    bwd = np.zeros((R, len(used) * D_MODEL), dtype=np.float64)
    bwd_bits = np.zeros((R, len(used) * D_MODEL), dtype=np.int32)
    if dev == "cuda":
        torch.cuda.synchronize()
    tb = time.perf_counter()
    for r in range(R):
        model.zero_grad(set_to_none=True)
        logits = model(idx)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), tgt.reshape(-1))
        loss.backward()
        gr = model.tok.weight.grad[torch.tensor(used, device=dev)].contiguous()
        bwd[r] = gr.double().cpu().numpy().ravel()
        bwd_bits[r] = gr.view(torch.int32).cpu().numpy().ravel()
    if dev == "cuda":
        torch.cuda.synchronize()
    bwd_secs = time.perf_counter() - tb
    bwd_mean = bwd.mean(0)
    bwd_distinct = np.array([len(np.unique(bwd_bits[:, k])) for k in range(bwd_bits.shape[1])])

    print(json.dumps(dict(
        identity=identity(), replicate=rep, deterministic=determ,
        cfg=dict(d_model=D_MODEL, n_layer=N_LAYER, seq=SEQ, vocab=VOCAB, R=R, dev=dev),
        fwd_secs=round(fwd_secs, 3), bwd_secs=round(bwd_secs, 3),
        fwd_nondet_frac=float((fwd_distinct > 1).mean()),
        fwd_distinct_med=float(np.median(fwd_distinct)),
        fwd_resid_mean=[round(float(x), 5) for x in fwd_resid.mean(0)[:4096]],
        bwd_nondet_frac=float((bwd_distinct > 1).mean()),
        bwd_distinct_med=float(np.median(bwd_distinct)),
        bwd_grad_mean=[round(float(x), 8) for x in bwd_mean[:4096]],
        bwd_used_rows=int(len(used)),
    )))


if __name__ == "__main__":
    main()
