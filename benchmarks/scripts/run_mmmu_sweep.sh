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
#       hosting sgl-omni serve --text-only --disable-radix-cache
#                   --mem-fraction-static <X> on port 30000
#     - sglang-hayden-benchmark (image lmsysorg/sglang:dev)
#       hosting python -m sglang.launch_server --disable-radix-cache
#                   --mem-fraction-static <X> on port 30001
#   Preflight runs ON the host (over SSH from the orchestrator) so all of
#   the readiness probes, docker inspect calls, and launcher-log greps
#   happen on the same machine that issued the docker run.
#
# Modes:
#   parallel-by-lane (default): ion8-omni runs Lane A, ion9-omni runs Lane B
#     in parallel (~1.5 GPU-h for 3 reps).
#   serial single-host (--serial): both lanes on the same host (~3 GPU-h).
#
# Authoritative metadata threading: each cell's benchmark_omni_mmmu invocation
# receives --preflight-json, --launcher-log, --mem-fraction-static so the
# resulting mmmu_results.json carries real model_revision, container
# digest, KV capacity, mem-fraction, and prefix-cache policy. After the
# remote run, scp pulls mmmu_results.json AND the host's preflight.json
# AND the cell's launcher.log back into the local cell_dir. The status
# JSONL row's container_image_digest is sourced from the retained
# run_metadata block, not from an orchestrator-local docker inspect (which
# would return the wrong digest in parallel-by-lane mode).
#
# Failure policy: each rep's success/failure is appended to
# <out>/sweep-status.jsonl. Failed reps are NOT silently retried.

set -euo pipefail

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
REPO_ID=""        # empty = default repo (full MMMU); set to zhaochenyang20/mmmu-ci-50 for curated subset
CONCURRENCY=8
MEM_FRACTION="0.9"
PREFLIGHT_REMOTE="/tmp/preflight.json"
LAUNCHER_LOG_OMNI="/tmp/sglang-omni-benchmark.log"
LAUNCHER_LOG_SGLANG="/tmp/sglang-benchmark.log"
REMOTE_REPO_ROOT="${REMOTE_REPO_ROOT:-/sgl-workspace/sglang-omni}"

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
  --mem-fraction N      Server mem_fraction_static (default: 0.9)
  -h, --help            Show this help

Pre-flight (per host):
  ssh <host> 'preflight_mmmu_sweep.py --launch --download --strict-log-check
              --launcher-log-omni /tmp/sglang-omni-benchmark.log
              --launcher-log-sglang /tmp/sglang-benchmark.log
              --mem-fraction-static <N> --disable-prefix-cache
              --output /tmp/preflight.json'

The sweep covers (reps) x (lanes) x (backends) = up to 2 * 2 * REPS runs.
Each cell produces a JSON artifact under <out>/lane_<lane>/<backend>/rep_<i>/
plus an entry in <out>/sweep-status.jsonl. Status rows include the
container_image_digest sourced from the cell's retained run_metadata.
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
        --repo-id) REPO_ID="$2"; shift 2;;
        --concurrency) CONCURRENCY="$2"; shift 2;;
        --mem-fraction) MEM_FRACTION="$2"; shift 2;;
        -h|--help) usage; exit 0;;
        *) echo "Unknown option: $1" >&2; usage; exit 1;;
    esac
done

LANES="$(printf '%s' "$LANES" | tr '[:upper:]' '[:lower:]')"

mkdir -p "$OUT_ROOT"
STATUS_LOG="$OUT_ROOT/sweep-status.jsonl"
: > "$STATUS_LOG"

# ---------------------------------------------------------------- preflight

# preflight_on_host <host>: ssh into the host and run preflight in --launch
# mode. Writes preflight.json on the remote host at $PREFLIGHT_REMOTE.
preflight_on_host() {
    local host="$1"
    local preflight_cmd=(
        python "$REMOTE_REPO_ROOT/benchmarks/scripts/preflight_mmmu_sweep.py"
        --launch
        --download
        --strict-log-check
        --launcher-log-omni "$LAUNCHER_LOG_OMNI"
        --launcher-log-sglang "$LAUNCHER_LOG_SGLANG"
        --mem-fraction-static "$MEM_FRACTION"
        --disable-prefix-cache
        --output "$PREFLIGHT_REMOTE"
    )
    echo "[sweep] preflight on $host..."
    if [[ "$host" == "$(hostname)" ]] || [[ -z "$host" ]]; then
        cd "$REMOTE_REPO_ROOT" 2>/dev/null || true
        "${preflight_cmd[@]}"
    else
        ssh "$host" "cd $REMOTE_REPO_ROOT && ${preflight_cmd[*]}"
    fi
}

if [[ "$SKIP_PREFLIGHT" -eq 0 ]]; then
    # Always preflight each host that will run cells. Even in serial mode the
    # single chosen host needs --launch + log capture before any benchmark
    # traffic; otherwise the launcher-log verification has no log to read.
    PREFLIGHT_HOSTS=()
    if [[ "$SERIAL" -eq 0 ]] && [[ "$LANES" == "both" ]]; then
        PREFLIGHT_HOSTS+=("$HOST_LANE_A" "$HOST_LANE_B")
    else
        PREFLIGHT_HOSTS+=("$HOST_LANE_A")
    fi
    for host in "${PREFLIGHT_HOSTS[@]}"; do
        preflight_on_host "$host" \
            || { echo "[sweep] preflight failed on $host; aborting" >&2; exit 2; }
    done
else
    echo "[sweep] WARNING: preflight skipped (--skip-preflight)"
fi

# ------------------------------------------------------------- exec helpers

# run_cell <host> <backend> <lane> <port> <rep_idx> <model> <launcher_log>
run_cell() {
    local host="$1" backend="$2" lane="$3" port="$4" rep_idx="$5" model="$6" launcher_log="$7"
    local cell_dir="$OUT_ROOT/lane_$lane/$backend/rep_$rep_idx"
    mkdir -p "$cell_dir"
    local stderr_log="$cell_dir/stderr.log"

    local container_name container_image
    if [[ "$backend" == "omni" ]]; then
        container_name="sglang-omni-hayden-benchmark"
        container_image="frankleeeee/sglang-omni:dev"
    else
        container_name="sglang-hayden-benchmark"
        container_image="lmsysorg/sglang:dev"
    fi

    # All metadata-source flags are threaded into the eval. Paths refer to the
    # host's filesystem (preflight wrote them there).
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
        --preflight-json "$PREFLIGHT_REMOTE"
        --launcher-log "$launcher_log"
        --mem-fraction-static "$MEM_FRACTION"
        --prefix-cache-disabled
    )
    if [[ -n "$SAMPLES" ]]; then
        cmd+=(--max-samples "$SAMPLES")
    fi
    if [[ -n "$REPO_ID" ]]; then
        cmd+=(--repo-id "$REPO_ID")
    fi

    echo "[sweep] cell host=$host backend=$backend lane=$lane rep=$rep_idx"
    local status=success
    local remote_dir
    if [[ "$host" == "$(hostname)" ]] || [[ -z "$host" ]]; then
        remote_dir="$cell_dir"
        cmd+=(--output-dir "$cell_dir")
        if ! (cd "$REMOTE_REPO_ROOT" 2>/dev/null && "${cmd[@]}") 2> "$stderr_log"; then
            status=failed
        fi
    else
        remote_dir="/tmp/mmmu-sweep-${lane}-${backend}-${rep_idx}-$$"
        cmd+=(--output-dir "$remote_dir")
        ssh "$host" "mkdir -p $remote_dir && rm -rf $remote_dir/*" 2>/dev/null || true
        if ! ssh "$host" "cd $REMOTE_REPO_ROOT && ${cmd[*]}" 2> "$stderr_log"; then
            status=failed
        fi
        # Copy back the result JSON, plus preflight + launcher log so the
        # retained bundle is self-contained and the run-metadata in the
        # JSON can be cross-checked against its source files.
        scp -q -r "${host}:${remote_dir}/." "${cell_dir}/" 2>>"$stderr_log" || true
        scp -q "${host}:${PREFLIGHT_REMOTE}" "${cell_dir}/preflight.json" 2>>"$stderr_log" || true
        scp -q "${host}:${launcher_log}" "${cell_dir}/launcher.log" 2>>"$stderr_log" || true
        ssh "$host" "rm -rf $remote_dir" 2>/dev/null || true
    fi

    # Source container digest + failure count from the cell's run_metadata
    # block rather than orchestrator-local docker inspect — that local
    # inspect would return the WRONG digest in parallel-by-lane mode where
    # the orchestrator's docker is not the host that ran the cell.
    local container_digest="" failure_count=0
    if [[ -f "${cell_dir}/mmmu_results.json" ]]; then
        container_digest="$(python -c "import json; d=json.load(open('${cell_dir}/mmmu_results.json'));m=d.get('run_metadata',{});print(m.get('container_image_digest') or '')" 2>/dev/null || true)"
        failure_count="$(python -c "import json; d=json.load(open('${cell_dir}/mmmu_results.json'));m=d.get('run_metadata',{});print(m.get('failure_count', 0))" 2>/dev/null || echo 0)"
    fi

    printf '{"host":"%s","backend":"%s","lane":"%s","rep":%d,"status":"%s","cell_dir":"%s","container_name":"%s","container_image":"%s","container_image_digest":"%s","server_port":%d,"failure_count":%s,"failure_log_path":"%s"}\n' \
        "$host" "$backend" "$lane" "$rep_idx" "$status" "$cell_dir" \
        "$container_name" "$container_image" "$container_digest" "$port" "$failure_count" \
        "$stderr_log" \
        >> "$STATUS_LOG"

    if [[ "$status" == "failed" ]]; then
        echo "[sweep] cell FAILED (host=$host backend=$backend lane=$lane rep=$rep_idx); see $stderr_log" >&2
        return 1
    fi
    return 0
}

# run_paired_rep <host> <lane> <rep_idx>: omni first, then sglang.
run_paired_rep() {
    local host="$1" lane="$2" rep_idx="$3"
    run_cell "$host" "omni" "$lane" "$PORT_OMNI" "$rep_idx" "$MODEL_OMNI" "$LAUNCHER_LOG_OMNI" || true
    run_cell "$host" "sglang" "$lane" "$PORT_SGLANG" "$rep_idx" "$MODEL_SGLANG" "$LAUNCHER_LOG_SGLANG" || true
}

run_lane_serial() {
    local host="$1" lane="$2"
    for ((rep = 0; rep < REPS; rep++)); do
        run_paired_rep "$host" "$lane" "$rep"
    done
}

# ----------------------------------------------------------- dispatch logic

LANES_TO_RUN=()
case "$LANES" in
    a) LANES_TO_RUN=("A");;
    b) LANES_TO_RUN=("B");;
    both) LANES_TO_RUN=("A" "B");;
    *) echo "Unknown --lanes value: $LANES" >&2; exit 1;;
esac

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

# --------------------------------------------------------- artifact validate

echo "[sweep] validating retained artifacts..."
if ! python benchmarks/scripts/validate_mmmu_artifacts.py "$OUT_ROOT" "$STATUS_LOG"; then
    echo "[sweep] artifact validator FAILED — bundle is not safe for reporting" >&2
    echo "[sweep] cells: see $STATUS_LOG" >&2
    exit 3
fi

echo "[sweep] complete. results under $OUT_ROOT"
echo "[sweep] status log: $STATUS_LOG"
TOTAL=$(wc -l < "$STATUS_LOG")
SUCCESS=$(grep -c '"status":"success"' "$STATUS_LOG" || echo 0)
FAILED=$((TOTAL - SUCCESS))
echo "[sweep] cells: total=$TOTAL success=$SUCCESS failed=$FAILED"
