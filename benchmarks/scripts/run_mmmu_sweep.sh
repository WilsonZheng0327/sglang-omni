#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Orchestrate the paired-rep MMMU sweep across both lanes and both backends.
#
# Layout:
#   Lane A: natural EOS, max_tokens=2048 (client-visible latency)
#   Lane B: ignore_eos=True, max_tokens=256 (decode-throughput parity)
#
# Per-host topology:
#   One H200 host (ion8-omni or ion9-omni) runs a pair of named containers:
#     - sglang-omni-hayden-benchmark (image frankleeeee/sglang-omni:dev)
#       hosting sgl-omni serve --text-only on port 30000
#     - sglang-hayden-benchmark (image lmsysorg/sglang)
#       hosting python -m sglang.launch_server on port 30001
#   The benchmark client process runs on the host and connects via
#   localhost:30000 / localhost:30001. Container names are enforced by
#   the preflight gate so AC-7's docker-inspect contract holds.
#
# Modes:
#   parallel-by-lane (default): when both --host-lane-a and --host-lane-b
#     are reachable, Lane A runs on host A while Lane B runs on host B in
#     parallel — wall-clock ~1.5 GPU-h for 3 reps.
#   serial single-host: when only one host is reachable (--serial or
#     auto-detected), both lanes run on the same host sequentially —
#     wall-clock ~3 GPU-h.
#
# Failure policy (AC-10): each rep's success/failure is appended to
# <out>/sweep-status.jsonl. Failed reps are NOT silently retried; the
# report renders cells as "(2/3 successful, 1 failed)" with a link to the
# captured stderr log.

set -euo pipefail

# --------------------------------------------------------------------- args

REPS=3
LANES="both"
OUT_ROOT="results/mmmu_sweep_$(date +%Y%m%d_%H%M%S)"
HOST_LANE_A="ion8-omni"
HOST_LANE_B="ion9-omni"
SERIAL=0
SKIP_PREFLIGHT=0
MODEL_OMNI="Qwen/Qwen3-Omni-30B-A3B-Instruct"
MODEL_SGLANG="Qwen/Qwen3-VL-30B-A3B-Instruct"
PORT_OMNI=30000
PORT_SGLANG=30001
SAMPLES=""        # empty = full MMMU (~900 samples)
CONCURRENCY=8

usage() {
    cat <<'EOF'
Usage: run_mmmu_sweep.sh [options]

Options:
  --reps N              Paired repetitions per cell (default: 3)
  --lanes a|b|both      Which lanes to run (default: both)
  --out PATH            Output root directory (default: results/mmmu_sweep_<ts>)
  --host-lane-a HOST    Host for Lane A in parallel mode (default: ion8-omni)
  --host-lane-b HOST    Host for Lane B in parallel mode (default: ion9-omni)
  --serial              Force single-host serial execution
  --skip-preflight      Skip the preflight gate (NOT recommended)
  --samples N           Cap MMMU sample count (default: full split)
  --concurrency N       Benchmark client max_concurrency (default: 8)
  -h, --help            Show this help

The sweep covers (reps) x (lanes) x (backends) = up to 2 * 2 * REPS runs.
Each run produces a JSON artifact under <out>/<lane>/<backend>/rep-<i>/
plus an entry in <out>/sweep-status.jsonl.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --reps) REPS="$2"; shift 2;;
        --lanes) LANES="$2"; shift 2;;
        --out) OUT_ROOT="$2"; shift 2;;
        --host-lane-a) HOST_LANE_A="$2"; shift 2;;
        --host-lane-b) HOST_LANE_B="$2"; shift 2;;
        --serial) SERIAL=1; shift;;
        --skip-preflight) SKIP_PREFLIGHT=1; shift;;
        --samples) SAMPLES="$2"; shift 2;;
        --concurrency) CONCURRENCY="$2"; shift 2;;
        -h|--help) usage; exit 0;;
        *) echo "Unknown option: $1" >&2; usage; exit 1;;
    esac
done

mkdir -p "$OUT_ROOT"
STATUS_LOG="$OUT_ROOT/sweep-status.jsonl"
: > "$STATUS_LOG"

# ---------------------------------------------------------------- preflight

if [[ "$SKIP_PREFLIGHT" -eq 0 ]]; then
    echo "[sweep] running preflight gate..."
    # AC-7: --strict-log-check makes any missing/non-matching launcher log
    # fatal in sweep mode, and --launcher-log-{omni,sglang} point the gate
    # at the tee'd logs produced by the operator's docker run --launch step.
    python benchmarks/scripts/preflight_mmmu_sweep.py \
        --host-lane-a "$HOST_LANE_A" \
        --host-lane-b "$HOST_LANE_B" \
        --launcher-log-omni /tmp/sglang-omni-benchmark.log \
        --launcher-log-sglang /tmp/sglang-benchmark.log \
        --strict-log-check \
        --output "$OUT_ROOT/preflight.json" \
        || { echo "[sweep] preflight failed; aborting" >&2; exit 2; }
else
    echo "[sweep] WARNING: preflight skipped (--skip-preflight)"
fi

# ------------------------------------------------------------- exec helpers

# run_cell <host> <backend> <lane> <port> <rep_idx> <model>
# Runs one benchmark cell against a remote (or local) host. Writes the
# result JSON under $OUT_ROOT/<lane>/<backend>/rep-<i>/mmmu_results.json
# and appends an entry to $STATUS_LOG.
#
# Remote-host mode (host != localhost / $(hostname)) runs the eval over
# SSH and writes results into a remote tmp dir; the dir is then scp'd
# back into the local cell_dir so $OUT_ROOT always has the artifacts
# the issue #379 follow-up comment needs.
run_cell() {
    local host="$1" backend="$2" lane="$3" port="$4" rep_idx="$5" model="$6"
    local cell_dir="$OUT_ROOT/lane_$lane/$backend/rep_$rep_idx"
    mkdir -p "$cell_dir"
    local stderr_log="$cell_dir/stderr.log"

    local cmd=(
        python -m benchmarks.eval.benchmark_omni_mmmu
        --base-url "http://localhost:$port"
        --model "$model"
        --backend "$backend"
        --lane "$lane"
        --stream
        --seed 42
        --reps "$REPS"
        --repetition-index "$rep_idx"
        --max-concurrency "$CONCURRENCY"
        --warmup 5
    )
    if [[ -n "$SAMPLES" ]]; then
        cmd+=(--max-samples "$SAMPLES")
    fi

    echo "[sweep] cell host=$host backend=$backend lane=$lane rep=$rep_idx"
    local status=success
    local container_name container_image container_digest
    if [[ "$backend" == "omni" ]]; then
        container_name="sglang-omni-hayden-benchmark"
        container_image="frankleeeee/sglang-omni:dev"
    else
        container_name="sglang-hayden-benchmark"
        container_image="lmsysorg/sglang"
    fi
    container_digest="$(docker inspect "$container_name" --format '{{index .Image}}' 2>/dev/null || true)"

    if [[ "$host" == "$(hostname)" ]] || [[ -z "$host" ]]; then
        cmd+=(--output-dir "$cell_dir")
        if ! "${cmd[@]}" 2> "$stderr_log"; then
            status=failed
        fi
    else
        local remote_dir="/tmp/mmmu-sweep-${lane}-${backend}-${rep_idx}-$$"
        cmd+=(--output-dir "$remote_dir")
        # Ensure remote dir exists and is clean before the run.
        ssh "$host" "mkdir -p $remote_dir && rm -rf $remote_dir/*" 2>/dev/null || true
        if ! ssh "$host" "cd /sgl-workspace/sglang-omni && ${cmd[*]}" 2> "$stderr_log"; then
            status=failed
        fi
        # Always attempt to copy back partial artifacts so debugging is possible
        # even on failure. scp non-zero is non-fatal (we still record status).
        scp -q -r "${host}:${remote_dir}/." "${cell_dir}/" 2>>"$stderr_log" || true
        ssh "$host" "rm -rf $remote_dir" 2>/dev/null || true
    fi

    local failure_count
    if [[ -f "${cell_dir}/mmmu_results.json" ]]; then
        failure_count="$(python -c "import json,sys; d=json.load(open('${cell_dir}/mmmu_results.json')); print(d.get('run_metadata',{}).get('failure_count',0))" 2>/dev/null || echo 0)"
    else
        failure_count=0
    fi

    printf '{"host":"%s","backend":"%s","lane":"%s","rep":%d,"status":"%s","cell_dir":"%s","container_name":"%s","container_image":"%s","container_image_digest":"%s","server_port":%d,"failure_count":%s}\n' \
        "$host" "$backend" "$lane" "$rep_idx" "$status" "$cell_dir" \
        "$container_name" "$container_image" "$container_digest" "$port" "$failure_count" \
        >> "$STATUS_LOG"

    if [[ "$status" == "failed" ]]; then
        echo "[sweep] cell FAILED (host=$host backend=$backend lane=$lane rep=$rep_idx); see $stderr_log" >&2
        # AC-10: do NOT silently retry. Continue to next cell.
        return 1
    fi
    return 0
}

# run_paired_rep <host> <lane>
# A single paired rep on one host: omni first, then sglang (so the two
# backends share the same hardware-warmup state within this rep).
run_paired_rep() {
    local host="$1" lane="$2" rep_idx="$3"
    run_cell "$host" "omni" "$lane" "$PORT_OMNI" "$rep_idx" "$MODEL_OMNI" || true
    run_cell "$host" "sglang" "$lane" "$PORT_SGLANG" "$rep_idx" "$MODEL_SGLANG" || true
}

# run_lane_serial <host> <lane>
# All paired reps for one lane on one host.
run_lane_serial() {
    local host="$1" lane="$2"
    for ((rep = 0; rep < REPS; rep++)); do
        run_paired_rep "$host" "$lane" "$rep"
    done
}

# ----------------------------------------------------------- dispatch logic

LANES_TO_RUN=()
case "${LANES,,}" in
    a) LANES_TO_RUN=("A");;
    b) LANES_TO_RUN=("B");;
    both) LANES_TO_RUN=("A" "B");;
    *) echo "Unknown --lanes value: $LANES" >&2; exit 1;;
esac

# Decide parallel-by-lane vs serial-single-host. Parallel needs both hosts
# AND both lanes selected; otherwise we serialize on the first usable host.
PARALLEL=0
if [[ "$SERIAL" -eq 0 ]] && [[ " ${LANES_TO_RUN[*]} " == *" A "* ]] && [[ " ${LANES_TO_RUN[*]} " == *" B "* ]]; then
    if ssh -o BatchMode=yes -o ConnectTimeout=5 "$HOST_LANE_A" true 2>/dev/null \
        && ssh -o BatchMode=yes -o ConnectTimeout=5 "$HOST_LANE_B" true 2>/dev/null; then
        PARALLEL=1
    fi
fi

if [[ "$PARALLEL" -eq 1 ]]; then
    echo "[sweep] parallel-by-lane mode: Lane A on $HOST_LANE_A, Lane B on $HOST_LANE_B"
    run_lane_serial "$HOST_LANE_A" "A" &
    PID_A=$!
    run_lane_serial "$HOST_LANE_B" "B" &
    PID_B=$!
    wait "$PID_A" "$PID_B"
else
    HOST=""
    if ssh -o BatchMode=yes -o ConnectTimeout=5 "$HOST_LANE_A" true 2>/dev/null; then
        HOST="$HOST_LANE_A"
    elif ssh -o BatchMode=yes -o ConnectTimeout=5 "$HOST_LANE_B" true 2>/dev/null; then
        HOST="$HOST_LANE_B"
    else
        HOST="$(hostname)"
    fi
    echo "[sweep] serial single-host mode on $HOST (lanes: ${LANES_TO_RUN[*]})"
    for lane in "${LANES_TO_RUN[@]}"; do
        run_lane_serial "$HOST" "$lane"
    done
fi

# ----------------------------------------------------------------- summary

echo "[sweep] complete. results under $OUT_ROOT"
echo "[sweep] status log: $STATUS_LOG"
TOTAL=$(wc -l < "$STATUS_LOG")
SUCCESS=$(grep -c '"status":"success"' "$STATUS_LOG" || echo 0)
FAILED=$((TOTAL - SUCCESS))
echo "[sweep] cells: total=$TOTAL success=$SUCCESS failed=$FAILED"
