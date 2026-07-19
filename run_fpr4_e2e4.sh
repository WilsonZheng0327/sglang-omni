#!/usr/bin/env bash
# F-PR4 E2-E4 experiment runner. Benchmarks a Fun-ASR server that is
# ALREADY RUNNING (launch it yourself in another terminal, eager or
# compiled). Writes <mode>_{en,zh}[_transcripts].json and
# <mode>_sweep_{en,zh}.json into the repo root.
#
# Usage: ./run_fpr4_e2e4.sh [mode] [port]
#   mode: output filename prefix, "compiled" (default) or "eager"
#   port: server port, default 8000
set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-compiled_cpp}"
PORT="${2:-8000}"
MODEL="FunAudioLLM/Fun-ASR-Nano-2512-hf"

bench() {
    python -m benchmarks.eval.benchmark_asr_seedtts \
        --port "$PORT" --model-path "$MODEL" "$@"
}

echo "[fpr4] waiting for server on port $PORT ..."
until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; do sleep 5; done
echo "[fpr4] server is up"

# E2: transcript dumps for parity (c=1, 1 repeat, full set)
for lang in en zh; do
    echo "[fpr4] E2 dump: $MODE $lang"
    bench --lang "$lang" --concurrencies 1 --repeats 1 \
        --dump-transcripts --output "${MODE}_${lang}.json"
done

# E4 (and E3, which is read from the c=32 rows): full sweeps
for lang in en zh; do
    echo "[fpr4] E4 sweep: $MODE $lang"
    bench --lang "$lang" --concurrencies 1,2,4,8,16,32,64 \
        --repeats 3 --warmup --output "${MODE}_sweep_${lang}.json"
done

echo "[fpr4] done. Next:"
echo "  - parity/WER/perf analysis: compare ${MODE}_* against the other mode's files (f-pr4.md steps 17-19)"
echo "  - E5: grep -ci recompil <server log>  (expect 0)"
