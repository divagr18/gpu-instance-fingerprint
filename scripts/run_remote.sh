#!/usr/bin/env bash
# Run a probe on one or more remote GPU machines over SSH, in parallel, and
# collect the JSON it prints into a local directory.
#
# Usage:
#   scripts/run_remote.sh <experiment> <label>=<user@host[:port]> [<label>=<user@host[:port]> ...]
#
# Example:
#   OUT_DIR=data/my_run REPS=4 scripts/run_remote.sh probe gpu1=root@gpu-host-1:22 gpu2=root@gpu-host-2
#
# Experiments (output file names use <label> and the replicate index):
#   probe     probes/probe_puf.py EPOCHS rep          -> probe_<label>_r<rep>.json
#   passive   probes/passive_index_add.py rep K       -> passive_<label>_r<rep>.json
#   forge     experiments/forge_probe.py rep RUNS     -> forge_<label>_r<rep>.json
#   dense     experiments/passive_dense.py rep        -> dense_<label>_r<rep>.json
#   scatter   experiments/passive_scatter.py rep      -> scatter_<label>_r<rep>.json
#   leak      experiments/transformer_leak.py rep 0, then 0 1
#                                                    -> tl_<label>_r<rep>.json, tl_<label>_det.json
#   channels  experiments/channel_probe.py tc rep, then dvfs 0
#                                                    -> ch_tc_<label>_r<rep>.json, ch_dvfs_<label>_r0.json
#   defense   experiments/defense_probe.py rep 0|1    -> def_<label>_{norm,det}_r<rep>.json
#
# Environment:
#   OUT_DIR       local output directory              (default: data/new)
#   REPS          replicates per machine              (default: 2; use 4 for probe)
#   EPOCHS        epochs per probe_puf.py run         (default: 3)
#   K             passive_index_add.py signature size (default: 65536)
#   RUNS          forge_probe.py launches per config  (default: 1200)
#   REMOTE_DIR    working directory on the remote     (default: /root)
#   REMOTE_PY     python on the remote                (default: python3)
#   INSTALL_CUPY  1 = pip install cupy-cuda12x on the remote if missing (default: 1)
#   SSH_OPTS      extra ssh options                   (default: connect/keepalive timeouts)
#
# The remote needs an NVIDIA GPU, nvidia-smi, numpy, and cupy (probe, forge)
# or torch (everything else). docker/Dockerfile builds an image with cupy and
# probe_puf.py preinstalled.
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXP="${1:?usage: run_remote.sh <experiment> <label>=<user@host[:port]> ...}"
shift
[ $# -ge 1 ] || { echo "need at least one <label>=<user@host[:port]>" >&2; exit 2; }

OUT_DIR="${OUT_DIR:-$ROOT/data/new}"
REPS="${REPS:-2}"
EPOCHS="${EPOCHS:-3}"
K="${K:-65536}"
RUNS="${RUNS:-1200}"
REMOTE_DIR="${REMOTE_DIR:-/root}"
REMOTE_PY="${REMOTE_PY:-python3}"
INSTALL_CUPY="${INSTALL_CUPY:-1}"
SSH_OPTS="${SSH_OPTS:--o ConnectTimeout=20 -o ServerAliveInterval=15}"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/run_remote.log"

case "$EXP" in
  probe)    SCRIPT=probes/probe_puf.py ;;
  passive)  SCRIPT=probes/passive_index_add.py ;;
  forge)    SCRIPT=experiments/forge_probe.py ;;
  dense)    SCRIPT=experiments/passive_dense.py ;;
  scatter)  SCRIPT=experiments/passive_scatter.py ;;
  leak)     SCRIPT=experiments/transformer_leak.py ;;
  channels) SCRIPT=experiments/channel_probe.py ;;
  defense)  SCRIPT=experiments/defense_probe.py ;;
  *) echo "unknown experiment: $EXP" >&2; exit 2 ;;
esac
NAME="$(basename "$SCRIPT")"

one() {
  local label="${1%%=*}" target="${1#*=}" port=22
  if [[ "$target" == *:* ]]; then port="${target##*:}"; target="${target%:*}"; fi
  local ssh_="ssh $SSH_OPTS -p $port $target"

  for _ in $(seq 1 30); do $ssh_ "echo up" 2>/dev/null | grep -q up && break; sleep 6; done
  if [ "$INSTALL_CUPY" = 1 ] && { [ "$EXP" = probe ] || [ "$EXP" = forge ]; }; then
    $ssh_ "$REMOTE_PY -c 'import cupy' 2>/dev/null || pip install -q --break-system-packages cupy-cuda12x" >>"$LOG" 2>&1
  fi
  $ssh_ "mkdir -p $REMOTE_DIR && cat > $REMOTE_DIR/$NAME" < "$ROOT/$SCRIPT"

  run() {  # run <args> <outfile>
    local args="$1" out="$OUT_DIR/$2"
    $ssh_ "cd $REMOTE_DIR && $REMOTE_PY $NAME $args" > "$out" 2>>"$LOG"
    echo "$label $NAME $args -> $(basename "$out") ($(wc -c < "$out") bytes)" >>"$LOG"
  }

  local rep
  case "$EXP" in
    probe)    for rep in $(seq 0 $((REPS-1))); do run "$EPOCHS $rep" "probe_${label}_r${rep}.json"; done ;;
    passive)  for rep in $(seq 0 $((REPS-1))); do run "$rep $K" "passive_${label}_r${rep}.json"; done ;;
    forge)    for rep in $(seq 0 $((REPS-1))); do run "$rep $RUNS" "forge_${label}_r${rep}.json"; done ;;
    dense)    for rep in $(seq 0 $((REPS-1))); do run "$rep" "dense_${label}_r${rep}.json"; done ;;
    scatter)  for rep in $(seq 0 $((REPS-1))); do run "$rep" "scatter_${label}_r${rep}.json"; done ;;
    leak)     for rep in $(seq 0 $((REPS-1))); do run "$rep 0" "tl_${label}_r${rep}.json"; done
              run "0 1" "tl_${label}_det.json" ;;
    channels) for rep in $(seq 0 $((REPS-1))); do run "tc $rep" "ch_tc_${label}_r${rep}.json"; done
              run "dvfs 0" "ch_dvfs_${label}_r0.json" ;;
    defense)  for rep in $(seq 0 $((REPS-1))); do run "$rep 0" "def_${label}_norm_r${rep}.json"; done
              for rep in $(seq 0 $((REPS-1))); do run "$rep 1" "def_${label}_det_r${rep}.json"; done ;;
  esac
}

for spec in "$@"; do one "$spec" & done
wait
echo "done: $EXP -> $OUT_DIR (log: $LOG)"
