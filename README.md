# GPU instance fingerprinting from atomic-contention order

This repo identifies individual NVIDIA GPUs that are physically identical and the same model, using only the numerical output of a parallel reduction. When thread blocks race to a global `atomicAdd`, the hardware decides the arrival order, per-SM manufacturing skew biases that order, and the bias is stable enough to tell one die from another.

It contains the active probe, a passive variant that works on plain `torch.index_add_` output, the forgery, leakage-boundary and defense experiments, every measurement JSON behind the results below, and analysis scripts that reproduce each number on a CPU.

## Results

| Result | Number | Reproduce |
|---|---|---|
| Active probe, **19 identical A40s**, P3 channel (numerical output only, no `%smid`) | **99.1% LOO (111/112)**, chance 5.3%, separation 4.28x | `analysis/pooled_analyze.py` |
| Active probe, 8 A40s | P3 **100% LOO**, separation 4.09x, permutation p = 0.0005 | `analysis/analyze_puf.py` |
| Driver confound: 4 A40s on the identical driver (580.159.04) | P3 **100%** (6.10x), P2 100% (5.43x) | `analysis/analyze_puf.py` |
| Reboot: container restart, same die verified by UUID (4 A40s) | **100% re-identification (12/12)**, both channels | `analysis/temporal_analyze.py` |
| RTX 3090, 7 chips across 4 driver versions | P3 **100% LOO** (4.55x), P2 100% (8.12x), p = 0.0005 | `analysis/cross_model_analyze.py` |
| RTX 4090, 2 chips on the same host and driver | P3 6/6 (12.5x), P2 6/6 (17.8x) | `analysis/cross_model_analyze.py` |
| L40S, 2 chips | P3 4/4 (17.2x), P2 4/4 (13.8x) | `analysis/cross_model_analyze.py` |
| **Passive** `torch.index_add_`, K=65536, 6 A40s | **100% same-driver (9/9)**, 78% cross-driver | `analysis/passive_analyze.py` |
| Passive signature capacity, K=65536 | **~217 independent bits** (A40), ~239 (RTX 4090) | `analysis/roc_capacity.py` |
| Forgery: attacker sweeps every launch knob it controls | best forge **0.616** correlation vs **0.92** genuine floor | `analysis/forge_analyze.py` |
| Boundary: vanilla transformer fwd/bwd, dense GEMM, TF32 matmul | bit-identical across chips, **no leak** | `analysis/leak_analyze.py`, `analysis/channel_analyze.py` |
| Defense: `torch.use_deterministic_algorithms(True)` | passive channel **eliminated**; 2.5-3.6x backward cost; active probe unaffected | `analysis/defense_analyze.py` |

LOO = leave-one-out nearest-centroid classification of single measurements against per-chip centroids. Separation = between-chip distance / within-chip distance.

## How it works

The probe (`probes/probe_puf.py`) is a small cupy kernel. It launches 4096 blocks of 128 threads. Every thread does the same fixed chain of FMAs, then thread 0 of each block takes a ticket from a global counter with `atomicAdd(counter, 1)`. The ticket is that block's arrival slot. Nothing about the input differs between chips; the only thing that varies is which blocks get there first.

That order is not random. The work distributor maps block indices onto SMs in a near-fixed pattern, and each SM on a given die runs slightly faster or slower than its siblings because of manufacturing variation. Averaged over 3000 launches, each block index settles to a mean arrival rank that reflects the speed of the SM it lands on.

```
fixed kernel, fixed input
        |
        v
blocks race to atomicAdd  -->  arrival slot per block (hardware-decided)
        |
        v  average over 3000 launches
mean arrival rank per block index (P3)  /  per physical SM via %smid (P2)
        |
        v  subtract the cross-chip mean (removes the architecture)
per-chip residual  -->  nearest-centroid match against enrolled chips
```

Three channels come out of one run:

- **P3, per-block rank.** Mean arrival slot per block index. It is read from the kernel's numerical output alone: no `%smid`, no timers, no privileged counters. This is the headline channel.
- **P2, per-SM rank.** The same slots binned by physical SM via the `%smid` register. Binning by physical SM averages out thermal noise, so P2 is the more stable channel under load (100% re-ID after 60 s of full-load stress, vs 67% for P3).
- **P1, marginal sum.** A histogram of the float32 result of a fixed reduction. It is weak (1.20x separation on 8 A40s), because summing destroys most of the order information.

Subtracting the mean across chips removes everything the chips share (architecture, driver, compiler) and leaves the per-die residual. Two measurements of the same chip land close together; measurements of different chips do not.

## Quickstart

```bash
git clone https://github.com/divagr18/gpu-instance-fingerprint.git
cd gpu-instance-fingerprint
pip install -r requirements.txt        # numpy + scipy, enough for every analysis
```

**Run the analysis on the included data (CPU, no GPU needed):**

```bash
python analysis/pooled_analyze.py      # 19 A40s: 111/112 = 99.1%
python analysis/analyze_puf.py "a40_8chip/*.json"   # 8 A40s: P3 100%
```

**Run the probe on a GPU:**

```bash
pip install numpy cupy-cuda12x
python probes/probe_puf.py 3 0 > gpu_a_r0.json   # 3 epochs, replicate 0 (about 2 s per epoch)
python probes/probe_puf.py 3 1 > gpu_a_r1.json
```

Collect a few replicates from each of two or more GPUs of the same model, then:

```bash
python analysis/analyze_puf.py "path/to/*.json"
```

Each JSON records the GPU UUID, driver, cupy version and per-epoch clock and temperature next to the measurements. The analysis groups by model name and labels chips by UUID.

To run on remote machines over SSH, `scripts/run_remote.sh` pushes a probe, runs it on several hosts in parallel and collects the JSON:

```bash
OUT_DIR=data/my_run REPS=4 scripts/run_remote.sh probe gpu1=root@HOST1:PORT gpu2=root@HOST2:PORT
```

`docker/Dockerfile` builds an image with CUDA 12.8, cupy and the probe preinstalled (`docker build -f docker/Dockerfile -t gpu-instance-fingerprint .`).

## Reproducing each result

All commands run from the repo root on a CPU. Scripts find their data in `data/` on their own.

| Result | Command | Data |
|---|---|---|
| 8 A40s: P3 100% (4.09x, p=0.0005), P2 94% (4.63x), P1 28% (1.20x) | `python analysis/analyze_puf.py "a40_8chip/*.json"` | `data/a40_8chip` (8 chips x 4 replicates) |
| Same-driver 4 A40s: P3 100% (6.10x), P2 100% (5.43x) | `python analysis/analyze_puf.py "a40_8chip/puf8_[4678]_*.json"` | `data/a40_8chip` |
| Pooled: 12 chips 98.6% (71/72); 19 chips 99.1% (111/112), 4.28x, 97% of chip pairs separated by more than 2x the within-chip noise | `python analysis/pooled_analyze.py` | `a40_8chip`, `a40_stability` (T1), `a40_reboot` (T1), `a40_scale` |
| Reboot: 100% re-ID (12/12) for P3 (margin 5.07x) and P2 (margin 5.03x) | `python analysis/temporal_analyze.py a40_reboot` | `data/a40_reboot` (4 chips) |
| 60 s stress: baseline LOO 96% (23/24); re-ID P2 100% (18/18), P3 67% (12/18) | `python analysis/temporal_analyze.py a40_stability` | `data/a40_stability` (6 chips) |
| RTX 3090 x7, RTX 4090, 3090 pair, L40S pair | `python analysis/cross_model_analyze.py` | `data/rtx3090`, `data/rtx4090`, `data/cross_model` |
| Passive index_add (K=65536, K=4096), dense control, scatter-overwrite | `python analysis/passive_analyze.py` | `data/passive_*` |
| ROC and bit capacity | `python analysis/roc_capacity.py` | all A40 probe data, `passive_k4096`, `passive_k65536`, `rtx4090` |
| Forgery: 0.616 best forge vs 0.946 genuine (floor 0.916), impostor 0.174 | `python analysis/forge_analyze.py` | `data/forge` (3 A40s) |
| Transformer boundary and deterministic-mode cost | `python analysis/leak_analyze.py` | `data/transformer_leak` (3 A40s) |
| TF32 matmul and DVFS boundary | `python analysis/channel_analyze.py` | `data/channels` (3 A40s) |
| Defense against the passive channel | `python analysis/defense_analyze.py` | `data/defense` (2 A40s) |

`analysis/analyze_puf.py` runs a 2000-shuffle permutation test per channel, so the larger sets take a few minutes.

### Bit capacity and ROC

Every pair of single measurements is scored (genuine = same UUID, impostor = different UUID), with no enrollment averaging, so these are the pessimistic single-shot operating points. Degrees of freedom are iris-style, `DoF = p(1-p)/var(HD)` over impostor Hamming distances.

| Signature | Pairs | Score | AUC | EER | Capacity |
|---|---|---|---|---|---|
| Active P3, 19 A40s | 644 genuine / 9367 impostor | residual correlation | 0.974 | 9.3% (d' = 2.95) | |
| Active P3, 19 A40s | | sign Hamming | 0.847 | 26.9% | ~5 bits |
| Passive K=4096, 8 A40s | | sign Hamming | 0.878 | 21.0% | ~97 bits |
| Passive K=65536, 6 A40s | | sign Hamming | 0.952 | 11.1% | ~217 bits |
| Passive K=65536, 3 RTX 4090s | | sign Hamming | | 0.0% | ~239 bits |

The active channel is best read through residual correlation: its sign pattern is spatially correlated, so the identity lives in the magnitudes. The passive sign signature behaves like a long bit string, and its capacity grows with K: about 200+ independent bits at K=65536, enough to index a very large fleet.

## The passive attack: plain `torch.index_add_`

The active probe needs to run a custom kernel. The passive variant reads nothing but the ordinary output of a stock PyTorch op.

`torch.index_add_` on CUDA accumulates with atomics, so the order in which contributions land in each output bin is hardware-decided. In the usual dense regime (many contributions per bin) the float32 sum averages that order out: on 2^20 contributions into 256 bins, every feature classifies at chance (combined 17% vs 17% chance on 6 A40s). The fix is a sparse, high-dynamic-range regime:

- 4 contributions per bin, K bins (K=65536 by default)
- magnitudes spread from 2^-30 to 2^30 with random signs, so each bin's float32 sum is dominated by add order (catastrophic cancellation)
- fixed seed, identical input on every chip

Each chip's stable atomic ordering then yields a stable output. The signature is the per-bin sign of the mean ULP residual against a float64 reference, a K-bit string.

```bash
pip install numpy torch
python probes/passive_index_add.py 0 65536 > passive_r0.json   # replicate 0, K=65536
```

Results on A40 (K=65536, 6 chips): **100% (9/9) within a driver version** and **78% across mixed drivers** (chance 17%). The raw op output also carries a driver-dependent component, which the active probe's order residual does not, so the passive channel is cleanest within one driver. At K=4096 it reaches 67% across 8 chips. A scatter-overwrite (`scatter_` with duplicate indices, last writer wins) reaches 60% across 5 chips. On RTX 4090 the passive signature separates all 3 chips (6/6). This regime shows up in real workloads such as MoE dispatch, GNN aggregation and embedding-bag backward with many collisions.

## Where the leak stops

Instance leakage is carried by hardware-scheduled atomic-contention order and nothing else. Every deterministic path tested is bit-identical across identical chips:

- **Vanilla transformer** (6 layers, d=512, fixed weights and input, 3 A40s, 200 forward + 200 backward passes): forward logits and the embedding gradient have nondeterministic fraction 0.000 on every chip and are bit-identical across chips. LOO is at chance (2/6).
- **Dense GEMM / TF32 tensor cores** (tall-skinny matmul, 3 A40s, 300 runs): nondeterministic fraction 0.000, residual bit-identical across chips.
- **DVFS** (steady-state clock and power under a 30 s matmul load, 3 A40s): clock spread across chips 13.4 MHz vs 12.5 MHz within-chip jitter; power spread 2.3 W vs 6.5 W jitter. Not separable.

So running ordinary ML on a GPU does not by itself expose the chip. Exposure needs the order-dominated regime: the active probe, or ops with many atomic contributions at high dynamic range.

## Forgery

Can chip A be made to emit chip B's fingerprint? The attacker cannot change silicon or set the atomic order, but it does control the launch: grid size, block size, compute burn before the atomic, and an injected spin delay. `experiments/forge_probe.py` sweeps a 10-config battery on each of 3 A40s (2 runs each) and records the per-SM signature for every config.

| Comparison | Correlation |
|---|---|
| Genuine: same chip, run to run | 0.946 (floor 0.916) |
| Impostor: chip A vs chip B | 0.174 |
| Forge: A's best config over the whole battery vs B | **0.616** max |

The best knob-tuned forgery never reaches the genuine floor, so a verifier thresholding around 0.8 admits every genuine chip and rejects every forgery.

## Defense

Turn on deterministic algorithms:

```python
import os, torch
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)
```

On 2 A40s on the same host, the passive `index_add_` signature goes from 91.2% of bins deterministic run-to-run (det_frac 0.912) to 100% (1.0000), and the signatures become bit-identical across chips (genuine and impostor Hamming distance both 0). The passive channel is gone. The cost, measured on the transformer above, is 2.5-3.6x on the backward pass.

The active probe is unaffected: it is the tenant's own CUDA kernel, so framework flags do not reach it. Stopping it means not allowing custom kernels, or virtualizing SM scheduling at the hypervisor level.

## Repo layout

```
probes/
  probe_puf.py            active probe (cupy): P1, P2, P3 channels
  passive_index_add.py    passive probe (torch): sparse index_add_ sign signature
experiments/
  forge_probe.py          forgery: launch-knob sweep of the per-SM signature (cupy)
  passive_dense.py        dense index_add_ (negative control)
  passive_scatter.py      scatter-overwrite and early sparse index_add_
  transformer_leak.py     vanilla transformer forward/backward leakage
  channel_probe.py        TF32 matmul (tc) and DVFS power/clock channels
  defense_probe.py        passive signature in normal vs deterministic mode
analysis/                 CPU-only; each reads its data from data/
  analyze_puf.py          within/between separation, LOO, permutation test
  pooled_analyze.py       12- and 19-chip pooled A40 analysis
  temporal_analyze.py     stress and reboot re-identification
  cross_model_analyze.py  RTX 3090, RTX 4090, L40S
  passive_analyze.py      passive index_add_, dense control, scatter-overwrite
  roc_capacity.py         AUC, EER, d', bit capacity
  forge_analyze.py        forgery test
  leak_analyze.py         transformer boundary + deterministic-mode cost
  channel_analyze.py      TF32 and DVFS boundary
  defense_analyze.py      deterministic-mode defense
data/                     measurement JSONs, one folder per experiment
  a40_8chip/              puf8_<chip>_r<rep>.json      8 A40s x 4
  a40_stability/          puf_stab_<chip>_t<phase>_r<rep>.json  T1 baseline, T2 after stress
  a40_reboot/             reb_<chip>_t<phase>_r<rep>.json       T1 baseline, T3 after restart
  a40_scale/              scl_<chip>_r<rep>.json       10 A40s x 4
  rtx3090/                d12_<chip>.json              5 RTX 3090s
  rtx4090/                g90_probe_*, g90_pass_*      RTX 4090 active + passive
  cross_model/            xm_probe_*, xm_pass_*        2 RTX 3090s + 2 L40S
  passive_k65536/         r4_*                         6 A40s x 3
  passive_k4096/          r3_*                         8 A40s x 3
  passive_dense/          rfp_*                        6 A40s x 4
  passive_scatter/        rsc_*                        5 A40s x 4
  forge/ transformer_leak/ channels/ defense/
scripts/run_remote.sh     run any probe on several SSH hosts in parallel
docker/                   probe image (CUDA 12.8 + cupy + sshd)
```

Chip numbers in file names are per-experiment indices. The ground-truth chip identity is the GPU UUID inside each JSON, which is how chips that recur across experiments are matched.

## License

MIT. See [LICENSE](LICENSE).
