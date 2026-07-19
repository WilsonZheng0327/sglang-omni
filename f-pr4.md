# F-PR4: Fun-ASR Encoder torch.compile (dynamic=True)

Tracking issue: [#924](https://github.com/sgl-project/sglang-omni/issues/924) (Fun-ASR-Nano roadmap).
Scope: *encoder `torch.compile` using `dynamic=True` for variable LFR input
lengths.* Follow-up to F-PR1 (#1078, model support); complementary to F-PR3
(encoder service) and F-PR5 (batched encoder).

## Why

The SANM encoder (`FunAsrNanoAudioEncoder`) is 70 sequential blocks (50 SANM +
20 timestamp-prediction) of hand-written eager PyTorch: per layer, separate
q/k/v/out Linears, explicit matmul+softmax attention, FSMN depthwise conv,
LayerNorms, FFN — thousands of small kernel launches per request on `[1, T,
512]` tensors. Fun-ASR-Nano is encoder-heavy (tiny Qwen3 LLM, short
transcripts), so launch/Python overhead dominates short-clip latency. Inductor
fusion attacks exactly that.

## Why dynamic=True

The feature extractor emits `[1, 560, T_lfr]` with `T_lfr` proportional to
audio duration (~1 frame / 60 ms → ~50–500 frames), and `get_audio_feature`
slices padding off before the encoder runs — so every request presents a
different sequence length. `dynamic=True` compiles one symbolic-shape graph up
front. It is *not* a further speedup over default `torch.compile` (steady
state is the same symbolic kernels); it is reliability: no wasted first static
compile, no mid-traffic recompile stall, no dependence on the
automatic-dynamic heuristic whose failure mode is a silent
recompile-limit fallback to eager.

## Implementation

- `_compile_fun_asr_audio_encoder(model, warmup_lfr_frames=128)` in
  `sglang_omni/models/fun_asr/stages.py` (repo convention: owned compile
  helpers are private `_compile_<model>_<component>` functions in `stages.py`
  — see `_compile_qwen3_tts_backbone`, `_compile_s2pro_codebook_decoder`;
  `sglang_model.py` stays pure model definition):
  - calls sglang's `set_torch_compile_config()` first, as both precedents do
    — enables `fx_graph_cache` (much faster warm-restart compiles) and lifts
    Dynamo cache limits;
  - compiles `audio_tower.forward` and `multi_modal_projector.forward` with
    `torch.compile(dynamic=True)`, default mode — matching the closest
    precedent, Higgs's variable-length acoustic encoder
    (`higgs_tts/stages.py`: `mode="default", dynamic=True` + warmup). The
    `SGLANG_TORCH_COMPILE_MODE`/max-autotune env pattern is only used for
    fixed-shape AR decode-layer compiles;
  - compiles the **bound forwards**, not `torch.compile(module)` — an
    `OptimizedModule` wrapper would prefix parameter names with `_orig_mod.`,
    breaking `load_weights` (strict name matching) and weight updates.
    (qwen3_tts preserves names by storing wrapped layers in a separate
    `_compiled_decode_layers` list; fishaudio also compiles bound methods);
  - runs one warmup forward at startup so the one-time compile cost never
    lands on a user request. `dynamic=True` means a single warmup length
    covers all lengths (must be ≥ 2: sizes 0/1 are always shape-specialized).
    The warmup uses no explicit grad context because Dynamo guards on grad
    mode and the omni scheduler event loop invokes forwards in ambient mode;
  - logs an INFO line on completion (fishaudio pattern).
- New stage factory arg `enable_encoder_torch_compile` (default `False`,
  opt-in) in `stages.py`, applied after weights load / CUDA-graph init.
  Deliberately separate from `enable_torch_compile`, which flows into sglang
  ServerArgs and compiles only the LLM decode path. This diverges from the
  qwen3_tts/fishaudio convention of intercepting `enable_torch_compile` (run
  owned compile, clear the flag) — fun_asr's `enable_torch_compile` has meant
  "LLM decode compile" since F-PR1, and the parity eval needs the two paths
  independently toggleable.
- `config.py` factory_args documents the flag (`False`).
- Not added: a `CAPABILITIES` declaration (`supports_torch_compile=True`) in
  `fun_asr/__init__.py`. It is startup-log-only and no ASR model declares one
  yet; reasonable follow-up if ASR capability logging becomes useful.

## Verification already done

- Unit tests (`tests/unit_test/fun_asr/test_encoder_compile.py`): plumbing —
  both forwards compiled with `dynamic=True`, parameter names unchanged,
  warmup executes with the configured length; degenerate warmup length
  rejected before compiling.
- GPU smoke (real dims, random weights, bf16, H200, batch 1). Results:
  - eager is length-independent at ~32 ms (launch-overhead-bound, as
    predicted); compiled is ~22 ms → **~1.46× encoder speedup**;
  - first calls after warmup: 22–38 ms at every length — **no per-length
    recompile stalls**, confirming `dynamic=True` serves all lengths from one
    graph;
  - one-time compile+warmup: ~281 s (measured without `fx_graph_cache`; the
    landed helper enables it via `set_torch_compile_config`, so warm restarts
    should be far cheaper);
  - numerics: max abs diff ~0.047, max relative ~2% vs eager — bf16 kernel
    reassociation across 70 layers, expected scale; transcript-level parity
    on real weights is experiment E2 below.

## Experiments

| # | Experiment | Gate for merge? |
|---|---|---|
| E1 | Startup compile cost, cold vs warm cache | no (informational) |
| E2 | Transcript parity, eager vs compiled, EN + ZH | **yes** |
| E3 | Corpus WER, both modes, EN + ZH | **yes** |
| E4 | Concurrency sweep: throughput / latency / RTF | **yes** (no regression; quantifies the win) |
| E5 | No-recompile check under full length variety | **yes** |
| E6 | GPU memory headroom | no (sanity) |

### How the pieces fit

Every experiment is a client/server pair. The `serve` command starts the
inference server: it loads the model onto the GPU, binds
`http://127.0.0.1:8000`, and then **blocks that terminal**, printing logs
until you stop it with Ctrl-C (`2>&1 | tee <file>` just saves a copy of
those logs for later grepping). The benchmark commands are separate client
processes you run **in a second terminal** while the server is up; they send
audio to port 8000 and score the responses.

"Eager" and "compiled" are two launch variants of the *same* server — never
run both at once (same port, same GPU). The whole matrix is three server
sessions: eager → compiled(cold) → compiled(warm). Keep
`enable_torch_compile` (the LLM flag) untouched in both modes so the encoder
change is measured in isolation.

### Runbook

**Prep (once, no server needed)**

1. Pick a free GPU and use it for every session:
   `nvidia-smi` → below assumes GPU 2 is free.
2. Download the eval set (EN + ZH splits; skip if already present):
   ```bash
   python -m benchmarks.dataset.prepare --dataset seedtts
   ```
3. Write the compiled-mode config (eager needs no file — the flag defaults
   to false; `runtime_overrides` is the documented per-stage factory-arg
   override, stage name `asr`):
   ```bash
   cat > /tmp/fun_asr_compiled.yaml <<'YAML'
   config_cls: FunASRPipelineConfig
   model_path: FunAudioLLM/Fun-ASR-Nano-2512-hf
   runtime_overrides:
     asr:
       enable_encoder_torch_compile: true
   YAML
   ```

**Session A — eager baseline**

4. Terminal 1 — start the eager server; leave it running:
   ```bash
   CUDA_VISIBLE_DEVICES=2 python -m sglang_omni.cli serve \
       --model-path FunAudioLLM/Fun-ASR-Nano-2512-hf --port 8000 \
       2>&1 | tee /tmp/fun_asr_eager.log
   ```
5. Terminal 2 — wait until the server answers, then everything below runs
   here:
   ```bash
   until curl -sf http://127.0.0.1:8000/health; do sleep 5; done
   ```
6. (Recommended) 1-minute plumbing smoke before committing to full runs:
   ```bash
   python -m benchmarks.eval.benchmark_asr_seedtts \
       --port 8000 --model-path FunAudioLLM/Fun-ASR-Nano-2512-hf \
       --lang en --max-samples 20 --concurrencies 2 --repeats 1
   ```
7. E2 parity inputs — full-set transcript dumps at concurrency 1:
   ```bash
   python -m benchmarks.eval.benchmark_asr_seedtts \
       --port 8000 --model-path FunAudioLLM/Fun-ASR-Nano-2512-hf \
       --lang en --concurrencies 1 --repeats 1 --dump-transcripts \
       --output eager_en.json     # also writes eager_en_transcripts.json
   python -m benchmarks.eval.benchmark_asr_seedtts \
       --port 8000 --model-path FunAudioLLM/Fun-ASR-Nano-2512-hf \
       --lang zh --concurrencies 1 --repeats 1 --dump-transcripts \
       --output eager_zh.json
   ```
   (An EN eager dump may already exist as `eager_transcripts.json`; redoing
   it under the clean name keeps the four files symmetric.)
8. E4 sweeps (these also produce E3's WER: c=32 × 3 repeats is in the
   sweep):
   ```bash
   python -m benchmarks.eval.benchmark_asr_seedtts \
       --port 8000 --model-path FunAudioLLM/Fun-ASR-Nano-2512-hf \
       --lang en --concurrencies 1,2,4,8,16,32,64 --repeats 3 --warmup \
       --output eager_sweep_en.json
   python -m benchmarks.eval.benchmark_asr_seedtts \
       --port 8000 --model-path FunAudioLLM/Fun-ASR-Nano-2512-hf \
       --lang zh --concurrencies 1,2,4,8,16,32,64 --repeats 3 --warmup \
       --output eager_sweep_zh.json
   ```
9. E6 eager memory reading (right after step 8, server still busy-warm):
   ```bash
   nvidia-smi --query-gpu=memory.used --format=csv -i 2   # note the number
   memory.used [MiB]
   133663 MiB
   ```
10. Terminal 1 — Ctrl-C the server. Wait for the prompt to return.

**Session B — compiled, cold cache**

11. Terminal 1 — force a cold Inductor cache, stamp the time, launch the
    compiled server with recompile logging on (E1 + E5 ride along free):
    ```bash
    rm -rf /tmp/torchinductor_$USER
    date
    TORCH_LOGS=recompiles CUDA_VISIBLE_DEVICES=2 python -m sglang_omni.cli serve \
        --config /tmp/fun_asr_compiled.yaml --port 8000 \
        2>&1 | tee /tmp/fun_asr_compiled.log
    ```
12. Watch terminal 1. When
    `Compiled Fun-ASR audio encoder + adaptor (dynamic=True, warmup_lfr_frames=128)`
    appears, run `date` in terminal 2. **E1 cold = second date − first
    date** (expect minutes).
    06:08:35
    06:14:10

13. Terminal 2 — repeat steps 5–9 exactly, with `eager` → `compiled` in
    every `--output` name (`compiled_en.json`, `compiled_sweep_en.json`, …).
    Note the step-9 memory number as the compiled reading for E6.
14. E5 — after the runs, count recompiles (server can stay up):
    ```bash
    grep -ci "recompil" /tmp/fun_asr_compiled.log
    ```
    **Pass: 0** after the startup warmup. Any hits mean `dynamic=True`
    isn't holding one graph — stop and investigate before trusting E4.
15. Terminal 1 — Ctrl-C the server.

**Session C — compiled, warm cache (E1 warm, ~2 min total)**

16. Terminal 1 — relaunch **without** clearing the cache, timing it the
    same way as steps 11–12:
    ```bash
    date
    CUDA_VISIBLE_DEVICES=2 python -m sglang_omni.cli serve \
        --config /tmp/fun_asr_compiled.yaml --port 8000 \
        2>&1 | tee /tmp/fun_asr_warm.log
    07:01:56
    07:03:42
    ```
    When the marker line appears: `date` again. **E1 warm** should be far
    smaller than E1 cold (fx_graph_cache hit). Ctrl-C; no benchmarks
    needed.

**Analysis (no server needed)**

17. E2 — transcript parity, once per language:
    ```bash
    python - eager_en_transcripts.json compiled_en_transcripts.json <<'EOF'
    import json, sys
    a, b = (json.load(open(p))["transcripts"]["1"] for p in sys.argv[1:3])
    ah = {r["id"]: r["hyp_text"] for r in a}
    bh = {r["id"]: r["hyp_text"] for r in b}
    diff = [k for k in ah if ah[k] != bh.get(k)]
    print(f"{len(diff)}/{len(ah)} transcripts differ")
    for k in diff[:20]:
        print(f"--- {k}\n  eager:    {ah[k]!r}\n  compiled: {bh[k]!r}")
    EOF
    ```
    **Pass:** 0 differ, or a handful of flips that are WER-neutral on
    manual review (punctuation/ITN variants). Anything systematic —
    repeated words, truncation — fails; greedy decode turns bf16 logit
    drift into visible token flips, which is exactly why this compares
    transcripts and not just WER.
18. E3 — open `eager_sweep_*.json` vs `compiled_sweep_*.json`, compare
    corpus WER at c=32. **Pass:** equal per language (EN reference: 0.0171
    from the F-PR1 table). Exact E2 parity makes this automatic.
19. E4 — from the same sweep files, tabulate per concurrency level:
    samples/s, mean latency, mean RTF, `rtf_p95`, eager vs compiled side by
    side (same shape as the reference table in the benchmark docstring).
    Expect the largest gain at c=1 (encoder step was 32 ms → 22 ms in the
    kernel smoke), shrinking toward saturation. **Pass:** no regression at
    any level; quote the c=1 and c=32 deltas in the PR description.
20. E6 — compare the two step-9 memory readings. **Pass:** compiled delta
    is modest (hundreds of MB, not GB) and the c=64 sweep completed without
    OOM.

## Results (2026-07-19, single H200, full SeedTTS EN=1088 / ZH=2020)

| Gate | Outcome |
|---|---|
| E1 startup | cold 331 s → warm 102 s (`fx_graph_cache` works) |
| E2 parity | **pass with notes** — EN 18/1088, ZH 24/2020 transcripts differ; greedy tie-flips (word substitutions, ITN/punctuation variants), no loops/truncation; bidirectional WER impact |
| E3 WER | **pass** — EN 0.017100→0.017252, ZH 0.014144→0.013725 (improved); identical across c=1..32 within each mode |
| E4 perf | **FAIL (EN)** — EN regresses at every level: −14% samples/s at c=1 worsening to −22% at c=64 (and 70 vs 31 skips at c=64); ZH −8% at c=1, then +2..4% at c≥8 |
| E5 recompiles | **pass** — 0 recompile events across the full compiled session |
| E6 memory | no red flag (readings dominated by the mem_fraction_static pool; c=64 completed without OOM) |

### Root cause of the E4 regression

Every op costs CPU time to *launch* and GPU time to *run*; the two overlap
and the slower side sets throughput. The encoder is thousands of tiny
kernels per request, so it is **launch-bound**: measured eager at T=77, ~74 ms
of CPU to issue vs ~66 ms of GPU work — the GPU waits on the CPU even in
isolation. Eager launches happen in C++ and **release the GIL**; the
compiled callable instead runs Dynamo guard checks and Triton's **Python**
kernel launchers, which **hold the GIL for the whole call** and cost at
least as much CPU as eager (measured ≥74 ms/call, noisy). The server runs
the encoder inside one multi-threaded, GIL-contended process (scheduler
loop + request-build workers; see the GIL note at `_event_loop_normal`), so
serving throughput tracks GIL-held CPU cost, not GPU time: compiled wins
GPU wall (~49 vs ~66 ms) yet loses throughput, and loses more as
concurrency raises contention (EN −14% at c=1 → −22% at c=64). The isolated
smoke measured single-thread wall time — effectively GPU time — which is
why it showed 1.46×. (Residual open question: why ZH mildly wins at c≥8
while EN never does; clip durations are near-identical.)

### Fix attempt: cpp_wrapper (implemented)

`_compile_fun_asr_audio_encoder` now sets
`torch._inductor.config.cpp_wrapper = True` before compiling: Inductor
emits a C++ launcher instead of Python Triton launchers, so issuing the
compiled kernels no longer holds the GIL per launch — directly attacking
the measured overhead while keeping the fused kernels. Guard evaluation
stays in Python but is micro-seconds per call. Note: cpp_wrapper changes
the Inductor cache key, so the first startup after this change recompiles
(cold-compile cost again, no need to clear caches manually) — and the
cpp_wrapper cold compile is much slower than the Python-wrapper one
(observed 20+ min vs ~5 min for this encoder; it C++-compiles a launcher
per kernel). This exceeds the launcher's default 600 s stage-readiness
budget, so the first compiled-mode launch after this change needs
`SGLANG_OMNI_STARTUP_TIMEOUT=3600` or the `asr` stage is killed mid-compile
with "Process asr did not become ready within 600s". Warm restarts hit the
cache and fit the default budget.

### Verdict so far

Quality, stability, and startup are fine; throughput as originally landed
was not. Do **not** flip the default until the cpp_wrapper re-run (below)
shows E4 flipping to a win. If it does not, reframe the PR as opt-in
infrastructure for F-PR3 — in a dedicated encoder process (own GIL), the
~49 vs ~66 ms GPU-side win is what matters.

### Re-run after the cpp_wrapper change

Eager baselines (`eager_*`) remain valid — only compiled-mode runs need
redoing, one server session (runbook steps 11–15, warmup is automatic at
startup via the marker line):
1. relaunch the compiled server with a raised startup budget for the
   one-time cpp_wrapper compile (20+ min):
   ```bash
   SGLANG_OMNI_STARTUP_TIMEOUT=3600 TORCH_LOGS=recompiles \
   CUDA_VISIBLE_DEVICES=2 python -m sglang_omni.cli serve \
       --config /tmp/fun_asr_compiled.yaml --port 8000 \
       2>&1 | tee /tmp/fun_asr_compiled_cpp.log
   ```
2. E2: re-dump `compiled_en` / `compiled_zh` transcripts (c=1, 1 repeat) —
   cpp_wrapper changes codegen, so parity must be re-checked;
3. E4/E3: re-run `compiled_sweep_en` (decisive) and `compiled_sweep_zh`;
4. E5: `grep -ci recompil` on the new server log;
5. compare against the existing eager files (analysis steps 17–19).

### Afterwards: flipping the default

If E2–E5 pass: flip `enable_encoder_torch_compile` to `True` in
`config.py` factory_args, refresh the reference-results table in the
`benchmark_asr_seedtts.py` docstring (repo convention: reference numbers
live in the benchmark file and are refreshed via the running-eval-suite
flow), and recalibrate the Fun-ASR CI speed gates against the new numbers
(same convention as the #1087 Qwen3-Omni speed-gate recalibration). That can
be this PR or a follow-up; keeping the flip separate matches how F-PR1
landed conservative defaults first.

## Notes / risks

- Startup cost: one-time compile (~minutes) when the flag is on; warmup keeps
  it off the request path.
- Flag stays opt-in until WER parity + RTFx gains are confirmed on the eval
  sets (F-PR8 refreshes reference numbers); then flip the default in
  `config.py`.
- If sglang ever wraps the omni event loop in `inference_mode`, the first
  request after startup pays one extra (symbolic, one-time) recompile from
  the grad-mode guard mismatch — worth revisiting the warmup context then.
