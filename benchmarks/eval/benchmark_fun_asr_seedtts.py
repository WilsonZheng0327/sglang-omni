# SPDX-License-Identifier: Apache-2.0
"""Fun-ASR speed + accuracy benchmark on SeedTTS (EN + ZH).

Transcribes SeedTTS evaluation clips against a *running* ``sgl-omni serve``
Fun-ASR server (OpenAI-compatible ``/v1/audio/transcriptions`` endpoint) and
reports, per concurrency level:

  * **Speed** — throughput (samples/s), latency mean/p95/p99, RTF mean/p95.
  * **Accuracy** — corpus WER + per-sample WER (max/mean/p95) against the
    dataset reference text.

This is the Fun-ASR counterpart of
``benchmarks.eval.benchmark_qwen3_asr_concurrency`` (same SeedTTS corpus, same
``BenchmarkRunner`` / ``normalize_text`` / ``calculate_wer_metrics`` /
``compute_speed_metrics`` primitives) but Fun-ASR-specific:

  * greedy by default (no ``temperature`` sent — Fun-ASR is correct under
    pure greedy, unlike Qwen3-ASR which needs 0.01);
  * ``language`` defaults to the split's language (``en``→en, ``zh``→zh);
  * ``max_new_tokens`` defaults to 256 (Fun-ASR stages.py default).

Both EN and ZH splits of ``zhaochenyang20/seed-tts-eval-arrow`` are supported.
WER normalization is language-aware via ``normalize_text`` (English uses the
openai-whisper ``EnglishTextNormalizer``; Mandarin uses ``zhon`` char-level).

Usage::

    # 0. Download the dataset once:
    python -m benchmarks.dataset.prepare --dataset seedtts

    # 1. Start the Fun-ASR server:
    sgl-omni serve --model-path FunAudioLLM/Fun-ASR-Nano-2512-hf --port 8731

    # 2. EN full sweep (1088 clips):
    python -m benchmarks.eval.benchmark_fun_asr_seedtts \
        --port 8731 --split en --concurrencies 1,8,32 --repeats 1 --warmup

    # 3. ZH full sweep (2020 clips):
    python -m benchmarks.eval.benchmark_fun_asr_seedtts \
        --port 8731 --split zh --concurrencies 1,8,32 --repeats 1 --warmup

    # Quick smoke on a 20-clip subset:
    python -m benchmarks.eval.benchmark_fun_asr_seedtts \
        --port 8731 --split en --max-samples 20 --concurrencies 2 --repeats 1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from typing import Any, Callable, Coroutine

import aiohttp
from jiwer import process_words

from benchmarks.benchmarker.data import RequestResult
from benchmarks.benchmarker.runner import BenchmarkRunner, RunConfig, SendFn
from benchmarks.dataset.prepare import DATASETS
from benchmarks.dataset.seedtts import SampleInput, load_seedtts_samples
from benchmarks.metrics.performance import compute_speed_metrics
from benchmarks.metrics.wer import (
    calculate_asr_speed_metrics,
    calculate_wer_metrics,
    print_asr_speed_summary,
    print_asr_wer_summary,
)
from benchmarks.tasks.tts import SampleOutput, normalize_text

DEFAULT_CONCURRENCIES = "1,8,32"
DEFAULT_META = DATASETS["seedtts"]  # zhaochenyang20/seed-tts-eval-arrow
DEFAULT_MODEL = "FunAudioLLM/Fun-ASR-Nano-2512-hf"
# Fun-ASR stages.py default max_new_tokens=256. SeedTTS clips are short
# sentences (a few seconds); 256 is ample and matches the model default.
DEFAULT_MAX_NEW_TOKENS = 256
REQUEST_TIMEOUT_S = 300


def get_audio_duration(path: str | os.PathLike) -> float:
    """Return playback length in seconds for a staged SeedTTS wav clip.

    ``benchmarks.benchmarker.utils.get_wav_duration`` parses raw WAV headers
    and would work here too, but ``soundfile.info`` is codec-agnostic (handles
    wav/opus/flac) and exposes ``.duration`` directly.
    """
    import soundfile as sf

    return float(sf.info(str(path)).duration)


def make_fun_asr_send_fn(
    model_name: str,
    api_url: str,
    *,
    lang: str = "en",
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> SendFn:
    """Return a ``send_fn(session, sample) -> RequestResult`` that transcribes
    one SeedTTS clip via the Omni ``/v1/audio/transcriptions`` endpoint.

    Fun-ASR transcribes greedily by default (``temperature=0.0`` in
    ``request_builders.py``), so — unlike Qwen3-ASR (which needs 0.01) — we do
    NOT send ``temperature``. ``language`` selects the Fun-ASR prompt:
    ``en`` → ``语音转写成英文：``; ``zh``/``cn``/``auto`` → bare
    ``语音转写：`` (multilingual verbatim), per ``_resolve_language``.
    """

    async def send_fn(
        session: aiohttp.ClientSession, sample: SampleInput
    ) -> RequestResult:
        result = RequestResult(request_id=sample.sample_id)
        try:
            with open(sample.ref_audio, "rb") as audio_file:
                audio_bytes = audio_file.read()
        except OSError as exc:
            result.error = str(exc)
            return result
        try:
            result.audio_duration_s = get_audio_duration(sample.ref_audio)
        except Exception:
            # non-fatal: RTF is skipped for this sample when duration==0
            result.audio_duration_s = 0.0

        form = aiohttp.FormData()
        form.add_field("model", model_name)
        form.add_field("language", lang)
        form.add_field("response_format", "json")
        form.add_field("max_new_tokens", str(max_new_tokens))
        form.add_field(
            "file",
            audio_bytes,
            filename=os.path.basename(sample.ref_audio),
            content_type="audio/wav",
        )

        start_time = time.perf_counter()
        try:
            async with session.post(api_url, data=form) as response:
                if response.status != 200:
                    result.error = (
                        f"HTTP {response.status}: {await response.text()}"
                    )
                else:
                    payload = await response.json()
                    result.text = str(payload.get("text", ""))
                    result.is_success = True
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            result.error = str(exc)
        finally:
            result.latency_s = time.perf_counter() - start_time
        if result.is_success and result.audio_duration_s > 0:
            result.rtf = result.latency_s / result.audio_duration_s
        return result

    return send_fn


async def run_fun_asr_transcription(
    samples: list[SampleInput],
    *,
    host: str = "127.0.0.1",
    port: int,
    model_name: str = DEFAULT_MODEL,
    lang: str = "en",
    concurrency: int = 1,
    warmup: int = 0,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    disable_tqdm: bool = True,
) -> tuple[list[RequestResult], float]:
    """Transcribe ``samples`` against a running Fun-ASR server at one
    concurrency. Returns ``(outputs, wall_clock_s)`` via ``BenchmarkRunner``.
    """
    api_url = f"http://{host}:{port}/v1/audio/transcriptions"
    send_fn = make_fun_asr_send_fn(
        model_name,
        api_url,
        lang=lang,
        max_new_tokens=max_new_tokens,
    )
    runner = BenchmarkRunner(
        RunConfig(
            max_concurrency=concurrency,
            warmup=warmup,
            disable_tqdm=disable_tqdm,
            timeout_s=REQUEST_TIMEOUT_S,
        )
    )
    outputs = await runner.run(samples, send_fn)
    return outputs, runner.wall_clock_s


def build_fun_asr_eval_results(
    samples: list[SampleInput],
    outputs: list[RequestResult],
    wall_clock_s: float,
    lang: str,
    *,
    model_name: str = DEFAULT_MODEL,
    concurrency: int = 1,
) -> dict:
    """Score transcriptions and assemble WER + speed metrics.

    Mirrors ``benchmark_qwen3_asr_concurrency.build_asr_eval_results`` but
    keeps the Fun-ASR-specific defaults. Returns
    ``{"summary": wer, "speed": speed, "per_sample": [...]}``.
    """
    result_by_id = {result.request_id: result for result in outputs}
    sample_outputs: list[SampleOutput] = []
    per_sample: list[dict] = []
    for sample in samples:
        result = result_by_id.get(sample.sample_id)
        output = SampleOutput(
            sample_id=sample.sample_id,
            target_text=sample.ref_text,
        )
        if result is None or not result.is_success:
            output.error = (result.error if result else "") or "No transcription"
        else:
            output.latency_s = result.latency_s
            output.asr_latency_s = result.latency_s
            output.audio_duration_s = result.audio_duration_s
            output.whisper_text = result.text
            output.ref_norm = normalize_text(sample.ref_text, lang)
            output.hyp_norm = normalize_text(result.text, lang)
            if output.ref_norm:
                measures = process_words(output.ref_norm, output.hyp_norm)
                output.wer = measures.wer
                output.substitutions = measures.substitutions
                output.deletions = measures.deletions
                output.insertions = measures.insertions
                output.hits = measures.hits
                output.is_success = True
            else:
                output.error = "Empty reference after normalization"
        sample_outputs.append(output)
        per_sample.append(
            {
                "id": output.sample_id,
                "is_success": output.is_success,
                "wer": output.wer if output.is_success else None,
                "ref_text": output.target_text,
                "hyp_text": output.whisper_text,
                "audio_duration_s": output.audio_duration_s,
                "latency_s": output.latency_s,
                "error": output.error,
            }
        )

    wer_summary = calculate_wer_metrics(sample_outputs, lang)
    wer_summary["corpus_wer"] = wer_summary["wer_corpus"]

    asr_speed = calculate_asr_speed_metrics(
        sample_outputs, wall_time_s=wall_clock_s
    )
    perf = compute_speed_metrics(outputs, wall_clock_s=wall_clock_s)
    speed = {
        **asr_speed,
        "asr_model": model_name,
        "asr_concurrency": concurrency,
        "asr_rtf_p95": perf.get("rtf_p95"),
        "throughput_samples_per_s": asr_speed["asr_throughput_samples_per_s"],
        "latency_mean_s": asr_speed["asr_latency_mean_s"],
        "latency_median_s": asr_speed["asr_latency_median_s"],
        "latency_p95_s": asr_speed["asr_latency_p95_s"],
        "latency_p99_s": asr_speed["asr_latency_p99_s"],
        "rtf_mean": asr_speed["asr_rtf_mean"],
        "rtf_median": asr_speed["asr_rtf_median"],
        "rtf_p95": perf.get("rtf_p95"),
    }
    return {"summary": wer_summary, "speed": speed, "per_sample": per_sample}


def _aggregate(repeats: list[dict]) -> dict:
    """Mean/best/worst across repeats for the headline metrics."""

    def _stat(key: str) -> dict:
        values = [r[key] for r in repeats]
        return {"mean": statistics.mean(values), "min": min(values), "max": max(values)}

    return {
        "concurrency": repeats[0]["concurrency"],
        "repeats": len(repeats),
        "evaluated": repeats[0]["evaluated"],
        "total": repeats[0]["total"],
        "skipped": repeats[0]["skipped"],
        "corpus_wer": _stat("corpus_wer"),
        "per_sample_wer_max": _stat("per_sample_wer_max"),
        "wall_clock_s": _stat("wall_clock_s"),
        "throughput_samples_per_s": _stat("throughput_samples_per_s"),
        "latency_mean_s": _stat("latency_mean_s"),
        "latency_p95_s": _stat("latency_p95_s"),
        "latency_p99_s": _stat("latency_p99_s"),
        "rtf_mean": _stat("rtf_mean"),
        "rtf_p95": _stat("rtf_p95"),
        "per_repeat": repeats,
    }


async def _run_repeat(
    args: argparse.Namespace,
    samples: list[SampleInput],
    concurrency: int,
    repeat: int,
) -> dict:
    outputs, wall_clock_s = await run_fun_asr_transcription(
        samples,
        host=args.host,
        port=args.port,
        model_name=args.model,
        lang=args.lang,
        concurrency=concurrency,
        max_new_tokens=args.max_new_tokens,
    )
    results = build_fun_asr_eval_results(
        samples,
        outputs,
        wall_clock_s,
        args.lang,
        model_name=args.model,
        concurrency=concurrency,
    )
    summary = results["summary"]
    speed = results["speed"]
    return {
        "concurrency": concurrency,
        "repeat": repeat,
        "evaluated": summary["evaluated"],
        "total": summary["total_samples"],
        "skipped": summary["skipped"],
        "corpus_wer": summary["corpus_wer"],
        "per_sample_wer_max": summary["wer_per_sample_max"],
        "wall_clock_s": wall_clock_s,
        "throughput_samples_per_s": speed["throughput_samples_per_s"],
        "latency_mean_s": speed["latency_mean_s"],
        "latency_p95_s": speed["latency_p95_s"],
        "latency_p99_s": speed["latency_p99_s"],
        "rtf_mean": speed["rtf_mean"],
        "rtf_p95": speed["rtf_p95"],
    }


async def _sweep(
    args: argparse.Namespace,
    samples: list[SampleInput],
    concurrencies: list[int],
) -> list[dict]:
    aggregates: list[dict] = []
    for concurrency in concurrencies:
        if args.warmup:
            print(f"[conc={concurrency}] warmup pass ...")
            await run_fun_asr_transcription(
                samples,
                host=args.host,
                port=args.port,
                model_name=args.model,
                lang=args.lang,
                concurrency=concurrency,
                max_new_tokens=args.max_new_tokens,
            )
        repeats: list[dict] = []
        for repeat in range(1, args.repeats + 1):
            result = await _run_repeat(args, samples, concurrency, repeat)
            repeats.append(result)
            print(
                f"[conc={concurrency} rep={repeat}] "
                f"wall={result['wall_clock_s']:.3f}s "
                f"thrpt={result['throughput_samples_per_s']:.3f}/s "
                f"lat_mean={result['latency_mean_s']:.3f}s "
                f"lat_p95={result['latency_p95_s']:.3f}s "
                f"rtf_mean={result['rtf_mean']:.4f} "
                f"corpus_wer={result['corpus_wer']:.4f} "
                f"skipped={result['skipped']}"
            )
        aggregates.append(_aggregate(repeats))
    return aggregates


def _print_table(aggregates: list[dict]) -> None:
    header = (
        "| conc | reps | wall(s) mean | thrpt mean | thrpt best | "
        "lat mean(s) | lat p95(s) | rtf mean | rtf p95 | corpus WER | max WER |"
    )
    sep = "|---:" * 11 + "|"
    print("\n" + header)
    print(sep)
    for agg in aggregates:
        print(
            f"| {agg['concurrency']} | {agg['repeats']} "
            f"| {agg['wall_clock_s']['mean']:.3f} "
            f"| {agg['throughput_samples_per_s']['mean']:.3f} "
            f"| {agg['throughput_samples_per_s']['max']:.3f} "
            f"| {agg['latency_mean_s']['mean']:.3f} "
            f"| {agg['latency_p95_s']['mean']:.3f} "
            f"| {agg['rtf_mean']['mean']:.4f} "
            f"| {agg['rtf_p95']['mean']:.4f} "
            f"| {agg['corpus_wer']['max']:.4f} "
            f"| {agg['per_sample_wer_max']['max']:.4f} |"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--port",
        type=int,
        required=True,
        help="Port of the running Fun-ASR SGLang Omni server.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model id reported in /v1/models (server-side label only).",
    )
    parser.add_argument(
        "--meta",
        default=DEFAULT_META,
        help="SeedTTS source (HF repo id or local meta.lst). "
        "Default: zhaochenyang20/seed-tts-eval-arrow (EN 1088 + ZH 2020).",
    )
    parser.add_argument(
        "--split",
        default="en",
        choices=["en", "zh"],
        help="SeedTTS split (en=1088 clips, zh=2020 clips).",
    )
    parser.add_argument(
        "--lang",
        default=None,
        help="Language for WER normalization + Fun-ASR prompt. "
        "Defaults to the --split value (en or zh).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Limit samples (0 = full split; 1088 for EN, 2020 for ZH).",
    )
    parser.add_argument(
        "--concurrencies",
        default=DEFAULT_CONCURRENCIES,
        help="Comma-separated concurrency levels to sweep.",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="max_new_tokens sent to the transcription endpoint.",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Run one discarded warmup pass before timing each concurrency.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Where to write the full JSON results. "
        "Default: fun_asr_seedtts_{split}_results.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.lang = args.lang or args.split
    lang = args.lang
    concurrencies = [int(c) for c in args.concurrencies.split(",") if c.strip()]
    max_samples = args.max_samples if args.max_samples > 0 else None

    samples = load_seedtts_samples(args.meta, max_samples=max_samples, split=args.split)
    if not samples:
        raise SystemExit(
            f"No SeedTTS samples loaded for meta={args.meta!r} split={args.split!r}"
        )
    print(
        f"Loaded {len(samples)} SeedTTS {args.split} samples; "
        f"sweeping concurrency={concurrencies} x {args.repeats} repeats "
        f"against {args.host}:{args.port} (model={args.model}, lang={lang})"
    )

    aggregates = asyncio.run(_sweep(args, samples, concurrencies))
    _print_table(aggregates)

    # Per-concurrency detail for the last repeat (WER/speed summary tables).
    print("\n=== last-repeat detail ===")
    for agg in aggregates:
        last = agg["per_repeat"][-1]
        print(
            f"\n[conc={agg['concurrency']}] corpus_wer={last['corpus_wer']:.4f} "
            f"evaluated={last['evaluated']}/{last['total']} skipped={last['skipped']}"
        )

    output_path = os.path.abspath(
        args.output or f"fun_asr_seedtts_{args.split}_results.json"
    )
    payload = {
        "config": {
            "host": args.host,
            "port": args.port,
            "meta": args.meta,
            "split": args.split,
            "lang": lang,
            "model": args.model,
            "max_new_tokens": args.max_new_tokens,
            "num_samples": len(samples),
            "concurrencies": concurrencies,
            "repeats": args.repeats,
            "warmup": args.warmup,
        },
        "results": aggregates,
    }
    with open(output_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"\nWrote results to {output_path}")


if __name__ == "__main__":
    main()
