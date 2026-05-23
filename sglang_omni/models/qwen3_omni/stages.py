# SPDX-License-Identifier: Apache-2.0
"""Stage factories for Qwen3-Omni pipelines.

Each factory returns either:
- A callable (compute_fn) for simple stages
- An OmniScheduler for AR stages
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from sglang_omni.models.qwen3_omni.bootstrap import create_thinker_scheduler
from sglang_omni.models.qwen3_omni.components.audio_encoder import Qwen3OmniAudioEncoder
from sglang_omni.models.qwen3_omni.components.image_encoder import Qwen3OmniImageEncoder
from sglang_omni.models.qwen3_omni.components.preprocessor import Qwen3OmniPreprocessor
from sglang_omni.models.qwen3_omni.components.streaming_detokenizer import (
    create_streaming_detokenize_scheduler,
)
from sglang_omni.models.qwen3_omni.encoder_model_runner import (
    Qwen3OmniAudioEncoderModelRunner,
    Qwen3OmniImageEncoderModelRunner,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.sglang_backend import (
    apply_encoder_mem_reserve,
    build_sglang_server_args,
)
from sglang_omni.scheduling.stage_cache import StageOutputCache
from sglang_omni.utils.gpu_memory import format_bytes_gib, get_process_gpu_memory_bytes
from sglang_omni.utils.misc import avail_gpu_mem

IMAGE_STAGE = "image_encoder"
AUDIO_STAGE = "audio_encoder"
THINKER_STAGE = "thinker"

logger = logging.getLogger(__name__)

# Image-encoder batching budget; the multiplier accounts for transient activations.
QWEN3_IMAGE_ENCODER_BATCH_BUDGET_BYTES = 10 * 1024**3

# CPU LRU cap for repeated-media encoder outputs.
QWEN3_ENCODER_CACHE_MAX_BYTES = 4 * 1024**3
QWEN3_ENCODER_CACHE_MAX_ENTRIES = 64


@dataclass(frozen=True)
class _ArMemoryContract:
    mem_fraction_static_pinned: bool
    effective_total_gpu_memory_fraction: float | None
    applied_encoder_mem_reserve: float


def _apply_qwen_thinker_encoder_reserve(
    server_args: Any,
    *,
    has_explicit_mem_fraction_static: bool,
    encoder_mem_reserve: float,
) -> bool:
    if has_explicit_mem_fraction_static:
        return False
    apply_encoder_mem_reserve(server_args, encoder_mem_reserve)
    return True


def _apply_colocated_ar_memory_contract(
    overrides: dict[str, Any],
    *,
    stage_name: str,
    total_gpu_memory_fraction: float | None,
    encoder_mem_reserve: float = 0.0,
) -> _ArMemoryContract:
    """Derive or validate SGLang AR memory args for a colocated stage."""

    if total_gpu_memory_fraction is None:
        return _ArMemoryContract(
            mem_fraction_static_pinned=overrides.get("mem_fraction_static") is not None,
            effective_total_gpu_memory_fraction=None,
            applied_encoder_mem_reserve=0.0,
        )

    explicit_mem_fraction = overrides.get("mem_fraction_static")
    if explicit_mem_fraction is not None:
        if encoder_mem_reserve:
            raise ValueError(
                f"Stage {stage_name} cannot apply encoder_mem_reserve when "
                "runtime.sglang_server_args.mem_fraction_static is explicitly set."
            )
        if abs(float(explicit_mem_fraction) - total_gpu_memory_fraction) > 1e-3:
            raise ValueError(
                f"Stage {stage_name} sets conflicting colocated memory "
                "contracts: runtime.resources.total_gpu_memory_fraction="
                f"{total_gpu_memory_fraction:.3f} and "
                "runtime.sglang_server_args.mem_fraction_static="
                f"{float(explicit_mem_fraction):.3f}. Use one value or make "
                "the explicit SGLang override match the stage total budget."
            )
        return _ArMemoryContract(
            mem_fraction_static_pinned=True,
            effective_total_gpu_memory_fraction=total_gpu_memory_fraction,
            applied_encoder_mem_reserve=0.0,
        )

    effective_total_gpu_memory_fraction = _apply_colocated_encoder_mem_reserve(
        total_gpu_memory_fraction,
        encoder_mem_reserve,
    )
    overrides["mem_fraction_static"] = effective_total_gpu_memory_fraction
    applied_encoder_mem_reserve = (
        encoder_mem_reserve
        if effective_total_gpu_memory_fraction != total_gpu_memory_fraction
        else 0.0
    )
    return _ArMemoryContract(
        mem_fraction_static_pinned=True,
        effective_total_gpu_memory_fraction=effective_total_gpu_memory_fraction,
        applied_encoder_mem_reserve=applied_encoder_mem_reserve,
    )


def _apply_colocated_encoder_mem_reserve(
    total_gpu_memory_fraction: float,
    encoder_mem_reserve: float,
) -> float:
    if not 0.0 <= encoder_mem_reserve < 1.0:
        raise ValueError("encoder_mem_reserve must be in [0, 1)")
    if encoder_mem_reserve == 0:
        return total_gpu_memory_fraction

    effective_total_gpu_memory_fraction = (
        total_gpu_memory_fraction - encoder_mem_reserve
    )
    if effective_total_gpu_memory_fraction < 0.1:
        raise ValueError(
            f"colocated total_gpu_memory_fraction {total_gpu_memory_fraction:.3f} "
            f"minus encoder_mem_reserve {encoder_mem_reserve:.3f} = "
            f"{effective_total_gpu_memory_fraction:.3f} is below the safe floor "
            "0.1; lower encoder_mem_reserve or increase the thinker stage budget."
        )
    return round(effective_total_gpu_memory_fraction, 3)


# ---------------------------------------------------------------------------
# Simple stages — return SimpleScheduler
# ---------------------------------------------------------------------------


def create_preprocessing_executor(
    model_path: str,
    *,
    thinker_max_seq_len: int | None = None,
    video_fps: float | None = None,
    video_max_frames: int | None = None,
    video_min_pixels: int | None = None,
    video_max_pixels: int | None = None,
    video_total_pixels: int | None = None,
):
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

    preprocessor = Qwen3OmniPreprocessor(
        model_path=model_path,
        max_seq_len=thinker_max_seq_len,
        video_fps=video_fps,
        video_max_frames=video_max_frames,
        video_min_pixels=video_min_pixels,
        video_max_pixels=video_max_pixels,
        video_total_pixels=video_total_pixels,
    )

    async def _preprocess(payload: StagePayload) -> StagePayload:
        return await preprocessor(payload)

    return SimpleScheduler(_preprocess)


def create_aggregate_executor():
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

    def _identity(payload: StagePayload) -> StagePayload:
        return payload

    return SimpleScheduler(_identity)


def create_image_encoder_executor(
    model_path: str,
    *,
    device: str = "cuda",
    dtype: str | None = None,
):
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

    model = Qwen3OmniImageEncoder(model_path=model_path, device=device, dtype=dtype)
    cache = StageOutputCache(
        max_size=QWEN3_ENCODER_CACHE_MAX_ENTRIES,
        max_bytes=QWEN3_ENCODER_CACHE_MAX_BYTES,
        cache_device="cpu",
    )
    runner = Qwen3OmniImageEncoderModelRunner(model=model, cache=cache)

    def _encode(payload: StagePayload) -> StagePayload:
        return runner.execute(payload)

    def _encode_batch(payloads: list[StagePayload]) -> list[StagePayload]:
        return runner.execute_batch(payloads)

    # Preserve the calibrated image-encoder batching shape and add a small
    # batch_wait so video benchmarks at concurrency=16 batch together.
    return SimpleScheduler(
        _encode,
        batch_compute_fn=_encode_batch,
        max_batch_size=32,
        max_batch_wait_ms=50,
        request_cost_fn=runner.estimate_payload_cost,
        max_batch_cost=QWEN3_IMAGE_ENCODER_BATCH_BUDGET_BYTES,
    )


def create_audio_encoder_executor(
    model_path: str,
    *,
    device: str = "cuda",
    dtype: str | None = None,
):
    from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

    model = Qwen3OmniAudioEncoder(model_path=model_path, device=device, dtype=dtype)
    cache = StageOutputCache(
        max_size=QWEN3_ENCODER_CACHE_MAX_ENTRIES,
        max_bytes=QWEN3_ENCODER_CACHE_MAX_BYTES,
        cache_device="cpu",
    )
    runner = Qwen3OmniAudioEncoderModelRunner(model=model, cache=cache)

    def _encode(payload: StagePayload) -> StagePayload:
        return runner.execute(payload)

    def _encode_batch(payloads: list[StagePayload]) -> list[StagePayload]:
        return runner.execute_batch(payloads)

    return SimpleScheduler(
        _encode,
        batch_compute_fn=_encode_batch,
        max_batch_size=32,
        max_batch_wait_ms=50,
    )


def create_decode_executor(model_path: str):
    return create_streaming_detokenize_scheduler(model_path)


# ---------------------------------------------------------------------------
# AR stages — return OmniScheduler
# ---------------------------------------------------------------------------


def create_sglang_thinker_executor_from_config(
    model_path: str,
    *,
    gpu_id: int = 0,
    tp_rank: int = 0,
    tp_size: int = 1,
    nccl_port: int | None = None,
    thinker_max_seq_len: int = 8192,
    server_args_overrides: dict[str, Any] | None = None,
    encoder_mem_reserve: float = 0.05,
    speech_enabled: bool = False,
    total_gpu_memory_fraction: float | None = None,
):
    """Returns OmniScheduler for thinker."""

    overrides: dict[str, Any] = {"disable_cuda_graph": False}
    if server_args_overrides:
        overrides.update(server_args_overrides)
    overrides["tp_size"] = tp_size
    has_explicit_colocated_mem_fraction = (
        total_gpu_memory_fraction is not None
        and overrides.get("mem_fraction_static") is not None
    )
    colocated_encoder_mem_reserve = (
        encoder_mem_reserve
        if total_gpu_memory_fraction is not None
        and not has_explicit_colocated_mem_fraction
        else 0.0
    )
    memory_contract = _apply_colocated_ar_memory_contract(
        overrides,
        stage_name="thinker",
        total_gpu_memory_fraction=total_gpu_memory_fraction,
        encoder_mem_reserve=colocated_encoder_mem_reserve,
    )
    server_args = build_sglang_server_args(
        model_path,
        context_length=thinker_max_seq_len,
        **overrides,
    )
    if total_gpu_memory_fraction is None:
        encoder_reserve_applied = _apply_qwen_thinker_encoder_reserve(
            server_args,
            has_explicit_mem_fraction_static=(
                memory_contract.mem_fraction_static_pinned
            ),
            encoder_mem_reserve=encoder_mem_reserve,
        )
        effective_total_gpu_memory_fraction = total_gpu_memory_fraction
        applied_encoder_reserve = (
            encoder_mem_reserve if encoder_reserve_applied else 0.0
        )
    else:
        effective_total_gpu_memory_fraction = (
            memory_contract.effective_total_gpu_memory_fraction
        )
        applied_encoder_reserve = memory_contract.applied_encoder_mem_reserve

    pre_load_avail_mem = avail_gpu_mem(gpu_id)
    pre_load_process_mem = get_process_gpu_memory_bytes(gpu_id)
    logger.info(
        f"sglang_ar_startup stage=thinker gpu_id={gpu_id} tp_rank={tp_rank}/{tp_size} "
        f"context_length={thinker_max_seq_len} "
        f"total_gpu_memory_fraction={total_gpu_memory_fraction} "
        f"effective_total_gpu_memory_fraction={effective_total_gpu_memory_fraction} "
        f"mem_fraction_static={server_args.mem_fraction_static} "
        f"encoder_mem_reserve={applied_encoder_reserve} "
        f"pre_load_avail_mem={pre_load_avail_mem} "
        f"pid={os.getpid()} "
        f"pre_load_process_mem={format_bytes_gib(pre_load_process_mem)}"
    )
    scheduler = create_thinker_scheduler(
        server_args,
        gpu_id,
        speech_enabled=speech_enabled,
        tp_rank=tp_rank,
        nccl_port=nccl_port,
        total_gpu_memory_fraction=effective_total_gpu_memory_fraction,
    )
    post_load_process_mem = get_process_gpu_memory_bytes(gpu_id)
    logger.info(
        f"sglang_ar_started stage=thinker gpu_id={gpu_id} tp_rank={tp_rank}/{tp_size} "
        f"context_length={thinker_max_seq_len} "
        f"total_gpu_memory_fraction={total_gpu_memory_fraction} "
        f"effective_total_gpu_memory_fraction={effective_total_gpu_memory_fraction} "
        f"mem_fraction_static={server_args.mem_fraction_static} "
        f"pre_load_avail_mem={pre_load_avail_mem} "
        f"post_load_avail_mem={avail_gpu_mem(gpu_id)} "
        f"pid={os.getpid()} "
        f"pre_load_process_mem={format_bytes_gib(pre_load_process_mem)}"
        f" post_load_process_mem={format_bytes_gib(post_load_process_mem)}"
    )
    return scheduler


def create_talker_ar_executor_from_config(
    model_path: str,
    *,
    gpu_id: int = 0,
    tp_rank: int = 0,
    tp_size: int = 1,
    nccl_port: int | None = None,
    talker_max_seq_len: int = 4096,
    server_args_overrides: dict[str, Any] | None = None,
    speech_enabled: bool = True,
    feedback_enabled: bool = True,
    weight_prefix: str = "talker.",
    total_gpu_memory_fraction: float | None = None,
):
    """Returns OmniScheduler for talker."""
    from sglang_omni.models.qwen3_omni.bootstrap import create_talker_scheduler

    # Note (Xuesong, Chenyang): cuda_graph defaults to ON for the talker
    # after #384, which routed talker MoE through `self.experts` (FusedMoE)
    # — the `fused_experts (full graph)` backend picked in #344. Caller can
    # override via factory_args or the `--talker-cuda-graph off` CLI flag.
    # Note (Xuesong): pytorch backend works around an sglang upstream gap —
    # Sampler.forward doesn't forward sampling_seed to flashinfer, so
    # under cuda graph the captured RNG is boot-dependent and ~5% of prompts
    # trigger degenerate AR loops (see #408). Revert once upstream lands.
    overrides: dict[str, Any] = {
        "disable_cuda_graph": False,
        "sampling_backend": "pytorch",
    }
    if server_args_overrides:
        overrides.update(server_args_overrides)
    overrides["tp_size"] = tp_size
    _apply_colocated_ar_memory_contract(
        overrides,
        stage_name="talker_ar",
        total_gpu_memory_fraction=total_gpu_memory_fraction,
    )
    server_args = build_sglang_server_args(
        model_path,
        context_length=talker_max_seq_len,
        **overrides,
    )
    pre_load_avail_mem = avail_gpu_mem(gpu_id)
    pre_load_process_mem = get_process_gpu_memory_bytes(gpu_id)
    logger.info(
        f"sglang_ar_startup stage=talker_ar gpu_id={gpu_id} tp_rank={tp_rank}/{tp_size} "
        f"context_length={talker_max_seq_len} "
        f"total_gpu_memory_fraction={total_gpu_memory_fraction} "
        f"mem_fraction_static={server_args.mem_fraction_static} "
        f"pre_load_avail_mem={pre_load_avail_mem} "
        f"pid={os.getpid()} "
        f"pre_load_process_mem={format_bytes_gib(pre_load_process_mem)}"
    )
    scheduler = create_talker_scheduler(
        server_args,
        gpu_id,
        weight_prefix=weight_prefix,
        speech_enabled=speech_enabled,
        feedback_enabled=feedback_enabled,
        tp_rank=tp_rank,
        nccl_port=nccl_port,
        total_gpu_memory_fraction=total_gpu_memory_fraction,
    )
    post_load_process_mem = get_process_gpu_memory_bytes(gpu_id)
    logger.info(
        f"sglang_ar_started stage=talker_ar gpu_id={gpu_id} tp_rank={tp_rank}/{tp_size} "
        f"context_length={talker_max_seq_len} "
        f"total_gpu_memory_fraction={total_gpu_memory_fraction} "
        f"mem_fraction_static={server_args.mem_fraction_static} "
        f"pre_load_avail_mem={pre_load_avail_mem} "
        f"post_load_avail_mem={avail_gpu_mem(gpu_id)} "
        f"pid={os.getpid()} "
        f"pre_load_process_mem={format_bytes_gib(pre_load_process_mem)}"
        f" post_load_process_mem={format_bytes_gib(post_load_process_mem)}"
    )
    return scheduler
