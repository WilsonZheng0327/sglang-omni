# SPDX-License-Identifier: Apache-2.0
"""MMMU benchmark for sglang-omni models.

Evaluates VLM accuracy and performance on the MMMU validation set via
/v1/chat/completions with image input.

Usage:
    # Text-only
    python benchmarks/eval/benchmark_omni_mmmu.py \
        --model qwen3-omni --port 8000 --max-samples 20

    # With concurrency
    python benchmarks/eval/benchmark_omni_mmmu.py \
        --model qwen3-omni --port 8000 --max-samples 50 --max-concurrency 16

    # With audio (requires speech server)
    # Note (Yifei, Chenyang): Concurrency=1 only for now since code_predictor and
    # code2wav modules serialize GPU access, so they run serially even when
    # concurrency > 1. And, audio output is still slow at this stage.

    python benchmarks/eval/benchmark_omni_mmmu.py \
        --model qwen3-omni --port 8000 --max-samples 5 --enable-audio --max-tokens 50


H200 Full-Set Reference Results

Reproducibility references for the FULL eval set — NOT CI thresholds.
CI runs on a subset and has its own thresholds elsewhere (see tasks/*.py).

Benchmark: MMMU     |  Dataset: MMMU_val (900 samples, all 30 subjects)
Hardware:  1 x H200 (default; non-H200 sources are tagged in Source column)
Last verified: 2026-05-04

Accuracy (summary)

| Model      | Config             | accuracy | correct | failed | mc_fallback | Source                                                 |
| ---------- | ------------------ | -------- | ------- | ------ | ----------- | ------------------------------------------------------ |
| Qwen3-Omni | enable_audio=False | 66.33%   | 597/900 | 0      | 22          | PR #393 [H200, V1-pipeline, full-set, c=8, max_tokens=2048]         |
| Qwen3-Omni | enable_audio=True  | 60.00%   | 30/50   | 0      | 2           | PR #393 [H200, V1-pipeline, 50-sample subset, c=1, max_tokens=2048] |
| Qwen3-Omni | enable_audio=False | 66.11%   | 595/900 | 0      | 28          | PR #351 [H100, full-set, c=8, max_tokens=2048, text-only server] |
| Qwen3-Omni | enable_audio=True  | 18.00%   | 9/50    | 21     | 20          | PR #351 [H100, 50-sample subset, c=1, max_tokens=64, timeout=120s] |

Note (Xuesong): full 900 not runfor enable_audio = True — Issue #276 talker is c=1 only and ~2 min/sample (~30 h for full set). 15/50 requests failed
 in audio generation (Issue #276); on the 35 completed requests accuracy = 65.7%.

Speed (speed)

| Model      | Config             | latency_mean_s | latency_p95_s | throughput_qps | tok_per_s_mean | tok_per_s_agg | Source                                                     |
| ---------- | ------------------ | -------------- | ------------- | -------------- | -------------- | ------------- | ---------------------------------------------------------- |
| Qwen3-Omni | enable_audio=False | 5.724          | 20.134        | 1.377          | 83.5           | 88.4          | PR #393 [H200, V1-pipeline, full-set, c=8, max_tokens=2048]             |
| Qwen3-Omni | enable_audio=True  | 70.927         | 197.541       | 0.014          | 10.2           | 8.2           | PR #393 [H200, V1-pipeline, **50-sample subset**, c=1, max_tokens=2048] |
| Qwen3-Omni | enable_audio=False | 20.297         | 74.122        | 0.392          | 24.9           | 25.4          | PR #351 [H100, full-set, c=8, max_tokens=2048, text-only server] |
| Qwen3-Omni | enable_audio=True  | 19.579         | 23.147        | 0.009          | 3.3            | 3.3           | PR #351 [H100, 50-sample subset, c=1, max_tokens=64, timeout=120s] |

Local v1 Pipeline Result (this workspace, 2026-05-01)

Accuracy (summary)

| Model      | Config             | accuracy | correct | failed | mc_fallback | Source                                                       |
| ---------- | ------------------ | -------- | ------- | ------ | ----------- | ------------------------------------------------------------ |
| Qwen3-Omni | enable_audio=False | 67.11%   | 604/900 | 0      | 26          | local v1 sweep [H200, full-set, c=8, max_tokens=2048]       |

Speed (speed)

| Model      | Config             | latency_mean_s | latency_p95_s | throughput_qps | tok_per_s_mean | tok_per_s_agg | Source                                                       |
| ---------- | ------------------ | -------------- | ------------- | -------------- | -------------- | ------------- | ------------------------------------------------------------ |
| Qwen3-Omni | enable_audio=False | 6.542          | 21.356        | 1.202          | 76.3           | 76.5          | local v1 sweep [H200, full-set, c=8, max_tokens=2048]       |
"""


from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.benchmarker.runner import BenchmarkRunner, RunConfig
from benchmarks.benchmarker.utils import save_json_results, wait_for_service
from benchmarks.dataset.mmmu import load_mmmu_samples
from benchmarks.metrics.mmmu import compute_mmmu_metrics, print_mmmu_accuracy_summary
from benchmarks.metrics.performance import compute_speed_metrics, print_speed_summary
from benchmarks.metrics.wer import print_wer_summary
# compute_text_audio_consistency is imported lazily inside the --enable-audio
# branch because benchmarks.tasks.tts pulls in soundfile/jiwer.
from benchmarks.tasks.visual_understand import (
    build_mmmu_result_records,
    make_mmmu_send_fn,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class MMMUEvalConfig:
    model: str
    base_url: str | None = None
    host: str = "localhost"
    port: int = 8000
    max_samples: int | None = None
    max_tokens: int = 2048
    temperature: float = 0.0
    output_dir: str | None = None
    max_concurrency: int = 1
    warmup: int = 0
    request_rate: float = float("inf")
    disable_tqdm: bool = False
    enable_audio: bool = False
    asr_device: str = "cuda:0"
    lang: str = "en"
    repo_id: str | None = None
    prompt_override: str | None = None
    timeout_s: int = 300
    # Evidence-based metadata sources: preflight.json supplies model_revision
    # + container digest; launcher.log supplies kv_cache_capacity_tokens; the
    # two policy flags below are cross-checked against the retained launch_command.
    preflight_json: str | None = None
    launcher_log: str | None = None
    mem_fraction_static: float | None = None
    prefix_cache_disabled: bool = True
    # "omni" → top-level images field; "sglang" → OAI messages[].content with image_url.
    backend: str = "omni"
    # When True, consume SSE response and populate TTFT + inter-chunk metrics.
    stream: bool = False
    seed: int | None = 42
    ignore_eos: bool = False
    # "A" = natural EOS + default max_tokens; "B" forces ignore_eos + max_tokens=256.
    lane: str = "A"
    reps: int = 3
    repetition_index: int = 0
    # None = use benchmarks/dataset/mmmu_revisions.json.
    dataset_revisions: str | None = None


def _build_base_url(config: MMMUEvalConfig) -> str:
    return config.base_url or f"http://{config.host}:{config.port}"


def _load_preflight_merge(preflight_json: str | None) -> dict:
    """Load preflight.json output (if any) for model_revision + digest merge."""
    if not preflight_json:
        return {}
    try:
        return json.loads(Path(preflight_json).read_text())
    except (OSError, json.JSONDecodeError):
        return {}


class LaunchPolicyMismatch(RuntimeError):
    """Eval's declared policy disagrees with the preflight launch_command evidence."""


def _derive_launch_policy_from_preflight(
    preflight: dict, container_name: str, declared_mem_fraction: float | None,
    declared_prefix_cache_disabled: bool,
    *,
    preflight_supplied: bool = False,
) -> tuple[float | None, bool, bool]:
    """Parse retained ``launch_command`` and return ``(mem_fraction, prefix_cache_disabled, claimed_unverified)``.

    Raises ``LaunchPolicyMismatch`` when launch_command flags disagree with
    declared values, or when ``preflight_supplied=True`` but launch_command
    is missing (closes the evidence-dropped-silently failure mode).
    """
    containers = (preflight.get("containers") or {}) if preflight else {}
    container_record = containers.get(container_name) or {}
    launch_cmd = container_record.get("launch_command")
    if not launch_cmd:
        if preflight_supplied:
            raise LaunchPolicyMismatch(
                f"--preflight-json was provided but the retained preflight record "
                f"for container {container_name!r} is missing `launch_command`. "
                f"The eval cannot derive `prefix_cache_disabled` or "
                f"`mem_fraction_static_configured` from evidence in this state. "
                f"Re-run preflight with --launch so the launch command is "
                f"captured, or omit --preflight-json for an unverified dev run."
            )
        # No evidence and no preflight supplied; the eval's declared values
        # stand but are marked claimed-unverified.
        return declared_mem_fraction, declared_prefix_cache_disabled, True

    # Parse flags out of the recorded list. The launch command is a list of
    # tokens that includes the docker run wrapper plus the server args; we
    # just scan for the two policy flags we care about.
    evidence_mem_fraction: float | None = None
    evidence_prefix_cache_disabled = False
    tokens = list(launch_cmd)
    for i, tok in enumerate(tokens):
        if tok == "--mem-fraction-static" and i + 1 < len(tokens):
            try:
                evidence_mem_fraction = float(tokens[i + 1])
            except ValueError:
                evidence_mem_fraction = None
        if tok == "--disable-radix-cache":
            evidence_prefix_cache_disabled = True

    # Mismatch checks: fail fast so the retained artifact cannot claim
    # a policy the launch command did not enforce.
    if (
        declared_mem_fraction is not None
        and evidence_mem_fraction is not None
        and abs(declared_mem_fraction - evidence_mem_fraction) > 1e-9
    ):
        raise LaunchPolicyMismatch(
            f"Eval CLI declares --mem-fraction-static={declared_mem_fraction} "
            f"but preflight launch_command for {container_name} carries "
            f"--mem-fraction-static={evidence_mem_fraction}. The artifact "
            f"cannot self-assert an unverified mem-fraction policy."
        )
    if declared_prefix_cache_disabled and not evidence_prefix_cache_disabled:
        raise LaunchPolicyMismatch(
            f"Eval CLI declares prefix_cache_disabled=True but preflight "
            f"launch_command for {container_name} does not include "
            f"--disable-radix-cache. The artifact cannot self-assert an "
            f"unverified prefix-cache policy."
        )

    return evidence_mem_fraction, evidence_prefix_cache_disabled, False


def _load_dataset_revisions(revisions_path: str | None) -> dict[str, str]:
    """Load the per-repo dataset revision dict for the metadata block."""
    from benchmarks.dataset.mmmu import _load_revision_map

    return _load_revision_map(revisions_path)


def _build_run_metadata(
    config: MMMUEvalConfig,
    *,
    request_results: list | None = None,
    steady_state_gpu_gb: list[float] | None = None,
) -> dict:
    """Populate the run-metadata block from authoritative sources.

    - ``model_revision`` and ``container_image_digest``: merged from
      ``preflight.json`` when ``config.preflight_json`` is set (the
      preflight gate is the source of truth for these). Otherwise falls
      back to live ``docker inspect``.
    - ``dataset_revisions``: loaded from
      ``benchmarks/dataset/mmmu_revisions.json`` (or
      ``config.dataset_revisions`` override).
    - ``mem_fraction_static_configured`` and ``prefix_cache_disabled``:
      sourced from the explicit CLI flags so they reflect the launch
      contract, not a per-eval guess.
    - ``kv_cache_capacity_tokens``: regex-scraped from
      ``config.launcher_log`` when provided.
    - ``steady_state_gpu_gb``: passed in by the caller after sampling at
      ``warmup_complete + 30s`` (or freshly sampled if not provided).
    - ``failure_count``: derived from ``request_results`` (count of
      ``not is_success``).
    """
    from benchmarks.scripts.run_metadata import (
        RunMetadata,
        get_commit_sha,
        get_container_image_digest,
        get_current_branch,
        get_gpu_topology,
        get_sglang_version,
        sample_gpu_memory_used_gb,
        scrape_kv_cache_capacity_from_log,
        to_dict,
    )

    repo_root = Path(__file__).resolve().parents[2]
    container_name = (
        "sglang-omni-hayden-benchmark"
        if config.backend == "omni"
        else "sglang-hayden-benchmark"
    )
    container_image = (
        "frankleeeee/sglang-omni:dev"
        if config.backend == "omni"
        else "lmsysorg/sglang"
    )

    preflight = _load_preflight_merge(config.preflight_json)
    preflight_models = (preflight.get("model_revisions") or {})
    preflight_containers = (preflight.get("containers") or {})
    model_repo = (
        "Qwen/Qwen3-Omni-30B-A3B-Instruct"
        if config.backend == "omni"
        else "Qwen/Qwen3-VL-30B-A3B-Instruct"
    )
    model_revision = preflight_models.get(model_repo)
    container_digest = (
        preflight_containers.get(container_name, {}).get("container_image_digest")
    )
    if container_digest is None:
        container_digest = get_container_image_digest(container_name)

    (
        mem_fraction_evidence,
        prefix_cache_disabled_evidence,
        claimed_unverified,
    ) = _derive_launch_policy_from_preflight(
        preflight,
        container_name,
        declared_mem_fraction=config.mem_fraction_static,
        declared_prefix_cache_disabled=config.prefix_cache_disabled,
        preflight_supplied=bool(config.preflight_json),
    )

    dataset_revisions = _load_dataset_revisions(config.dataset_revisions)

    if config.launcher_log:
        kv_capacity = scrape_kv_cache_capacity_from_log(Path(config.launcher_log))
    else:
        kv_capacity = None

    if steady_state_gpu_gb is None:
        # Caller didn't pre-sample at warmup+30s; fall back to a right-now sample.
        steady_state_gpu_gb = sample_gpu_memory_used_gb()

    failure_count = 0
    if request_results is not None:
        failure_count = sum(1 for r in request_results if not getattr(r, "is_success", False))

    meta = RunMetadata(
        commit_sha=get_commit_sha(repo_root),
        branch=get_current_branch(repo_root),
        sglang_version=get_sglang_version(),
        backend=config.backend,
        model_id=config.model,
        model_revision=model_revision,
        dataset_revisions=dataset_revisions,
        seed=config.seed,
        ignore_eos=config.ignore_eos,
        lane=config.lane,
        stream=config.stream,
        max_tokens=config.max_tokens,
        max_concurrency=config.max_concurrency,
        temperature=config.temperature,
        warmup=config.warmup,
        request_rate=(
            None if config.request_rate == float("inf") else config.request_rate
        ),
        timeout_s=config.timeout_s,
        repo_id=config.repo_id,
        max_samples=config.max_samples,
        mem_fraction_static_configured=mem_fraction_evidence,
        kv_cache_capacity_tokens=kv_capacity,
        steady_state_gpu_gb=steady_state_gpu_gb,
        prefix_cache_disabled=prefix_cache_disabled_evidence,
        encoder_patches_active=False,
        host=os.environ.get("HOSTNAME") or os.environ.get("HOST"),
        container_name=container_name,
        container_image=container_image,
        container_image_digest=container_digest,
        server_port=config.port,
        gpu_topology=get_gpu_topology(),
        repetition_index=config.repetition_index,
        failure_count=failure_count,
    )
    return to_dict(meta)


async def run_mmmu_eval(config: MMMUEvalConfig) -> dict:
    """Run full MMMU evaluation and return results dict.

    Returns a dict with keys: summary, speed, config,
    per_sample, and wer (only when enable_audio is True).
    """
    base_url = _build_base_url(config)
    api_url = f"{base_url}/v1/chat/completions"

    samples = load_mmmu_samples(
        config.max_samples,
        repo_id=config.repo_id,
        instruction_override=config.prompt_override,
        revisions_path=config.dataset_revisions,
    )
    logger.info(f"Prepared {len(samples)} MMMU samples")

    audio_dir: str | None = None
    if config.enable_audio and config.output_dir:
        audio_dir = str(Path(config.output_dir) / "audio")
        Path(audio_dir).mkdir(parents=True, exist_ok=True)

    send_fn = make_mmmu_send_fn(
        config.model,
        api_url,
        backend=config.backend,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        stream=config.stream,
        seed=config.seed,
        ignore_eos=config.ignore_eos,
        enable_audio=config.enable_audio,
        audio_dir=audio_dir,
    )

    runner = BenchmarkRunner(
        RunConfig(
            max_concurrency=config.max_concurrency,
            request_rate=config.request_rate,
            warmup=config.warmup,
            disable_tqdm=config.disable_tqdm,
            timeout_s=config.timeout_s,
            # Streaming runs use read_bufsize=1 per the plan's literal
            # parser contract: per-chunk SSE arrivals are visible to the
            # parser immediately, without a prefetch window coalescing them.
            read_bufsize=1 if config.stream else None,
        )
    )

    # Schedule a steady-state GPU memory sample for warmup_complete + 30s.
    # The sampler thread is launched from BenchmarkRunner's post-warmup
    # hook so its 30s sleep is anchored at warmup completion, not at run
    # start (which would include the warmup wall time).
    steady_state_gpu_holder: dict[str, list[float]] = {"value": []}
    sampler_thread_holder: dict[str, threading.Thread | None] = {"thread": None}

    def _sample_gpu_after_warmup() -> None:
        time.sleep(30)
        from benchmarks.scripts.run_metadata import sample_gpu_memory_used_gb

        steady_state_gpu_holder["value"] = sample_gpu_memory_used_gb()

    def _start_sampler_post_warmup() -> None:
        thread = threading.Thread(target=_sample_gpu_after_warmup, daemon=True)
        thread.start()
        sampler_thread_holder["thread"] = thread

    request_results = await runner.run(
        samples, send_fn, post_warmup_hook=_start_sampler_post_warmup
    )
    # Wait for the warmup+30s GPU sample before building run_metadata.
    if sampler_thread_holder["thread"] is not None:
        sampler_thread_holder["thread"].join(timeout=35)

    per_sample = build_mmmu_result_records(samples, request_results, lane=config.lane)
    summary = compute_mmmu_metrics(per_sample)
    speed_metrics = compute_speed_metrics(
        request_results, wall_clock_s=runner.wall_clock_s
    )

    config_dict = {
        "model": config.model,
        "base_url": base_url,
        "max_samples": config.max_samples,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "max_concurrency": config.max_concurrency,
        "warmup": config.warmup,
        "enable_audio": config.enable_audio,
        "backend": config.backend,
        "stream": config.stream,
        "seed": config.seed,
        "ignore_eos": config.ignore_eos,
        "lane": config.lane,
        "reps": config.reps,
        "repetition_index": config.repetition_index,
    }

    run_metadata = _build_run_metadata(
        config,
        request_results=request_results,
        steady_state_gpu_gb=steady_state_gpu_holder["value"] or None,
    )
    results = {
        "summary": summary,
        "speed": speed_metrics,
        "config": config_dict,
        "run_metadata": run_metadata,
        "per_sample": per_sample,
    }

    if config.enable_audio:
        from benchmarks.tasks.tts import compute_text_audio_consistency

        results["wer"] = compute_text_audio_consistency(
            request_results, config.lang, config.asr_device
        )

    if config.output_dir:
        save_json_results(results, config.output_dir, "mmmu_results.json")

    return results


def _config_from_args(args: argparse.Namespace) -> MMMUEvalConfig:
    """Resolve MMMUEvalConfig from argparse, enforcing lane contracts.

    Lane A: natural EOS; max_tokens defaults to 2048 (overridable).
    Lane B: ignore_eos=True + max_tokens=256, both NON-NEGOTIABLE (overrides rejected).
    """
    lane = args.lane.upper()
    if lane == "B":
        if args.max_tokens is not None and args.max_tokens != 256:
            raise SystemExit(
                f"Lane B is the fixed-length decode-throughput parity lane and "
                f"requires --max-tokens 256 (got {args.max_tokens}). Either drop "
                f"the override or switch to --lane A."
            )
        if args.ignore_eos is False:
            # ignore_eos is implied by Lane B and cannot be opted out of.
            # We accept --ignore-eos as a no-op redundancy but never let it
            # land False here.
            pass
        ignore_eos = True
        max_tokens = 256
    elif lane == "A":
        ignore_eos = bool(args.ignore_eos)
        max_tokens = args.max_tokens if args.max_tokens is not None else 2048
    else:
        raise SystemExit(
            f"--lane must be 'A' (natural EOS) or 'B' (ignore_eos + 256 tokens), got {args.lane!r}"
        )

    if args.stream and args.enable_audio:
        raise SystemExit(
            "--stream and --enable-audio cannot be combined: the audio response shape "
            "does not flow through the per-token SSE path. This combination is "
            "explicitly out of scope for this PR."
        )

    return MMMUEvalConfig(
        base_url=args.base_url,
        host=args.host,
        port=args.port,
        model=args.model,
        max_samples=args.max_samples,
        max_tokens=max_tokens,
        temperature=args.temperature,
        output_dir=args.output_dir,
        max_concurrency=args.max_concurrency,
        warmup=args.warmup,
        request_rate=args.request_rate,
        disable_tqdm=args.disable_tqdm,
        enable_audio=args.enable_audio,
        asr_device=args.asr_device,
        lang=args.lang,
        repo_id=args.repo_id,
        backend=args.backend,
        stream=args.stream,
        seed=args.seed,
        ignore_eos=ignore_eos,
        lane=lane,
        reps=args.reps,
        repetition_index=args.repetition_index,
        dataset_revisions=args.dataset_revisions,
        preflight_json=args.preflight_json,
        launcher_log=args.launcher_log,
        mem_fraction_static=args.mem_fraction_static,
        prefix_cache_disabled=args.prefix_cache_disabled,
    )


async def benchmark(args: argparse.Namespace) -> dict:
    config = _config_from_args(args)
    results = await run_mmmu_eval(config)
    print_mmmu_accuracy_summary(results["summary"], config.model)
    print_speed_summary(
        results["speed"],
        config.model,
        config.max_concurrency,
        title="MMMU Speed",
    )
    if "wer" in results:
        print_wer_summary(results["wer"]["summary"], config.model)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MMMU benchmark for VLM models served by sglang-omni."
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL (e.g. http://localhost:8000). Overrides --host/--port.",
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3-omni",
        help="Model name for the API request.",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    # Sentinel so _config_from_args can tell whether --max-tokens was passed
    # explicitly. Default semantics depend on --lane.
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Maximum concurrent requests.",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Requests per second (inf = send all at once).",
    )
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument(
        "--enable-audio",
        action="store_true",
        help="Request audio output and compute text-audio WER.",
    )
    parser.add_argument(
        "--asr-device",
        type=str,
        default="cuda:0",
        help="Device for ASR model (default: cuda:0).",
    )
    parser.add_argument(
        "--lang",
        choices=["en", "zh"],
        default="en",
        help="Language for ASR transcription (default: en).",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="HuggingFace dataset repo (e.g. 'zhaochenyang20/mmmu-ci-50'). "
        "Defaults to loading the full MMMU/MMMU (all 30 subjects).",
    )
    parser.add_argument(
        "--backend",
        choices=["omni", "sglang"],
        default="omni",
        help=(
            "Which backend payload shape to emit. 'omni' uses sglang-omni's "
            "top-level images field; 'sglang' uses OpenAI-style messages[]"
            ".content image_url parts ordered [image, image, ..., text] "
            "mirroring Qwen3OmniPreprocessor._build_multimodal_messages."
        ),
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help=(
            "Enable per-token SSE streaming and capture client-side TTFT + "
            "inter-content-chunk latency. Incompatible with --enable-audio."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Sampling seed forwarded to upstream SGLang SamplingParams via "
            "sampling_seed. Default 42 for reproducibility."
        ),
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help=(
            "Force the sampler to keep emitting until max_tokens by ignoring "
            "EOS. Implied by --lane B (which cannot be opted out of). "
            "Allowed but unusual on --lane A."
        ),
    )
    parser.add_argument(
        "--lane",
        choices=["A", "B", "a", "b"],
        default="A",
        help=(
            "A = natural EOS with --max-tokens 2048 (user-visible MMMU "
            "latency). B = --ignore-eos with --max-tokens 256 (fixed-length "
            "decode-throughput parity)."
        ),
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=3,
        help=(
            "Number of paired repetitions the sweep orchestrator runs per "
            "cell. Carried into the run-metadata block; this CLI runs one "
            "sweep per invocation."
        ),
    )
    parser.add_argument(
        "--repetition-index",
        type=int,
        default=0,
        help="Index of this run within the paired-rep cycle.",
    )
    parser.add_argument(
        "--dataset-revisions",
        type=str,
        default=None,
        help=(
            "Path to the per-repo dataset revision JSON. Defaults to "
            "benchmarks/dataset/mmmu_revisions.json. The loader fails closed "
            "when the chosen repo lacks an entry; populate it via the "
            "preflight gate."
        ),
    )
    parser.add_argument(
        "--preflight-json",
        type=str,
        default=None,
        help=(
            "Path to the preflight gate's JSON output. When provided, the "
            "eval merges model_revision and container_image_digest from "
            "the preflight record instead of recomputing them."
        ),
    )
    parser.add_argument(
        "--launcher-log",
        type=str,
        default=None,
        help=(
            "Path to the server launcher log. When provided, the eval "
            "scrapes kv_cache_capacity_tokens from it into the run-metadata block."
        ),
    )
    parser.add_argument(
        "--mem-fraction-static",
        type=float,
        default=None,
        help=(
            "Configured mem_fraction_static at server launch (passed through "
            "to the run-metadata block). The eval does NOT enforce this on "
            "the server; the operator must have launched the server with "
            "the matching --mem-fraction-static flag."
        ),
    )
    parser.add_argument(
        "--prefix-cache-disabled",
        action="store_true",
        default=True,
        help=(
            "Record that prefix caching is disabled (default: True). The "
            "operator must have launched the server with prefix caching "
            "actually disabled; this flag records the policy in metadata."
        ),
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = "results/mmmu_audio" if args.enable_audio else "results/mmmu"

    base_url = args.base_url or f"http://{args.host}:{args.port}"
    wait_for_service(base_url)

    asyncio.run(benchmark(args))


if __name__ == "__main__":
    main()
