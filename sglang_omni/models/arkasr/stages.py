# SPDX-License-Identifier: Apache-2.0
"""Stage factory for SGLang-backed ARK-ASR-3B inference."""

from __future__ import annotations

import contextlib
import logging
from typing import Any

import torch
from sglang.srt.managers.mm_utils import init_mm_embedding_cache
from transformers import AutoConfig, AutoTokenizer, WhisperFeatureExtractor

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.arkasr.encoder_service import (
    ArkASRPreLMEncoderService,
    build_cache_namespace,
)
from sglang_omni.models.arkasr.request_builders import make_arkasr_scheduler_adapters
from sglang_omni.scheduling.bootstrap import (
    create_sglang_infrastructure_defer_cuda_graph,
)
from sglang_omni.scheduling.generation_batch_policy import (
    build_generation_batch_overrides,
    validate_generation_batch_policy,
)
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.sglang_backend import (
    SGLangOutputProcessor,
    build_sglang_server_args,
)
from sglang_omni.utils.gpu_compat import get_visible_gpu_sm_version

logger = logging.getLogger(__name__)


def _compile_arkasr_audio_encoder(
    model: Any, *, warmup_mel_frames: int = 256, warmup_inference_mode: bool = True
) -> None:
    """Compile the Whisper tower and MLP adapter with a symbolic sequence length.

    ARK builds mels with ``padding="longest"``, so the frame count varies per
    request; ``dynamic=True`` builds one symbolic-shape graph instead of
    specializing per length, which would recompile until Dynamo's limit silently
    falls back to eager.

    Only the tower and the ``adapting`` MLP are compiled, not
    ``ArkAudioMLPAdapter.forward`` as a whole: the adapter's frame-merge step
    branches on ``seq_len % merge_factor``, and with a symbolic length that guard
    would specialize the graph per remainder. Those branches are cheap reshapes;
    the conv frontend, the encoder layers and the projection MLP are the work.

    The bound forwards are compiled rather than wrapping the modules in
    ``OptimizedModule`` so parameter names stay stable for ``load_weights``. The
    warmup pays the compile cost at startup instead of on the first request;
    Dynamo guards on grad mode, so it must run in the same mode as the serving
    caller — ``torch.inference_mode`` for the pre-LM encoder service, ambient
    mode for inline prefill on the scheduler loop.
    """
    from sglang.srt.compilation.torch_compile_decoration import set_torch_compile_config

    if warmup_mel_frames < 4:
        # After conv2's stride-2 the downsampled length must still be >= 2:
        # Dynamo shape-specializes sizes 0 and 1, so a smaller warmup would not
        # build the symbolic-length graph.
        raise ValueError(f"warmup_mel_frames must be >= 4, got {warmup_mel_frames}")

    set_torch_compile_config()
    tower = model.audio_encoder.whisper
    tower.forward = torch.compile(tower.forward, dynamic=True)
    model.audio_encoder.adapting.forward = torch.compile(
        model.audio_encoder.adapting.forward, dynamic=True
    )

    param = next(model.audio_encoder.parameters())
    num_mel_bins = int(model.config.whisper_config.num_mel_bins)
    warmup_ctx = (
        torch.inference_mode() if warmup_inference_mode else contextlib.nullcontext()
    )
    with warmup_ctx:
        # The tensors must be created inside the context, not merely passed
        # through it: tensors allocated under inference_mode lack the
        # ADInplaceOrView dispatch key and Dynamo guards on the key set, so a
        # normal tensor here would compile a graph the service's inference-mode
        # tensors fail, forcing a full recompile on the first real request.
        #
        # B=1 is today's path (the encoder service encodes per item); B=2 covers
        # the batched-tower graph so a later batched get_audio_feature does not
        # recompile at serving time.
        for batch in (1, 2):
            mel = torch.zeros(
                (batch, num_mel_bins, int(warmup_mel_frames)),
                device=param.device,
                dtype=param.dtype,
            )
            model.audio_encoder(mel)
    logger.info(
        "Compiled ARK-ASR audio tower + adapter MLP (dynamic=True, "
        "warmup_mel_frames=%d, warmup_inference_mode=%s, signatures=B1+B2)",
        warmup_mel_frames,
        warmup_inference_mode,
    )


def create_sglang_arkasr_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    max_running_requests: int = 32,
    max_new_tokens: int = 256,
    mem_fraction_static: float | None = None,
    mm_embedding_cache_size_bytes: int = 0,
    enable_torch_compile: bool = False,
    enable_encoder_torch_compile: bool = False,
    mm_attention_backend: str | None = None,
    enable_pre_lm_encoder: bool = True,
    pre_lm_cache_max_entries: int = 4096,
    pre_lm_cache_size_bytes: int = 2 * 1024**3,
    pre_lm_max_batch_size: int = 8,
    pre_lm_max_batch_wait_ms: int = 4,
    # The pre-LM service encodes synchronously inside the request builder, so
    # the builder pool is what feeds it: with only 2 workers the service could
    # never assemble a batch wider than 2. Matches Fun-ASR's pool for the same
    # reason.
    request_build_max_workers: int = 8,
    request_build_max_pending: int | None = 16,
    server_args_overrides: dict[str, Any] | None = None,
):
    if pre_lm_max_batch_size < 1:
        raise ValueError(
            f"pre_lm_max_batch_size must be >= 1, got {pre_lm_max_batch_size}"
        )
    if pre_lm_max_batch_wait_ms < 0:
        raise ValueError(
            f"pre_lm_max_batch_wait_ms must be >= 0, got {pre_lm_max_batch_wait_ms}"
        )

    gpu_id = int(device.split(":")[-1]) if ":" in device else 0

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path)
    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    merge_factor = int(getattr(hf_config, "merge_factor", 4))
    audio_token_id = int(getattr(hf_config, "audio_token_id", 151663))

    encoder_token_count = int(getattr(feature_extractor, "nb_max_frames", 3000) // 2)

    defaults: dict[str, Any] = {
        "disable_cuda_graph": False,
        "disable_overlap_schedule": True,
        "enable_torch_compile": enable_torch_compile,
        "mem_fraction_static": mem_fraction_static,
        "max_prefill_tokens": 4096,
        "chunked_prefill_size": 4096,
        "sampling_backend": "pytorch",
        "dtype": dtype,
    }
    if mm_attention_backend is not None:
        defaults["mm_attention_backend"] = mm_attention_backend
    else:
        sm_version = get_visible_gpu_sm_version(gpu_id)
        if sm_version is not None and sm_version >= 100:
            defaults["mm_attention_backend"] = "triton_attn"
    overrides = build_generation_batch_overrides(
        max_running_requests=max_running_requests,
        server_args_overrides=server_args_overrides,
        **defaults,
    )

    server_args = build_sglang_server_args(
        model_path,
        context_length=encoder_token_count + int(max_new_tokens) + 8,
        **overrides,
    )
    validate_generation_batch_policy(model_name="ARK-ASR", server_args=server_args)

    want_cuda_graph, (
        model_worker,
        tree_cache,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        prefill_mgr,
        decode_mgr,
        model_config,
    ) = create_sglang_infrastructure_defer_cuda_graph(
        server_args,
        gpu_id,
        model_arch_override="ArkasrForConditionalGeneration",
    )

    if want_cuda_graph:
        model_worker.model_runner.init_cuda_graphs()

    if enable_encoder_torch_compile:
        _compile_arkasr_audio_encoder(
            model_worker.model_runner.model,
            warmup_inference_mode=enable_pre_lm_encoder,
        )

    init_mm_embedding_cache(mm_embedding_cache_size_bytes)

    output_proc = SGLangOutputProcessor(
        capture_hidden=False,
        capture_hidden_layers=None,
        model=model_worker.model_runner.model,
    )
    audio_encoder_service = None
    if enable_pre_lm_encoder:
        model = model_worker.model_runner.model
        audio_encoder_service = ArkASRPreLMEncoderService(
            model,
            cache_namespace=build_cache_namespace(
                model,
                model_path=model_path,
                feature_extractor=feature_extractor,
                mm_attention_backend=server_args.mm_attention_backend,
            ),
            cache_max_entries=pre_lm_cache_max_entries,
            cache_max_bytes=pre_lm_cache_size_bytes,
            max_batch_size=pre_lm_max_batch_size,
            max_batch_wait_ms=pre_lm_max_batch_wait_ms,
        )

    try:
        request_builder, result_adapter = make_arkasr_scheduler_adapters(
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            max_new_tokens=max_new_tokens,
            merge_factor=merge_factor,
            audio_token_id=audio_token_id,
            audio_encoder_service=audio_encoder_service,
        )

        return OmniScheduler(
            tp_worker=model_worker,
            tree_cache=tree_cache,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            server_args=server_args,
            model_config=model_config,
            prefill_manager=prefill_mgr,
            decode_manager=decode_mgr,
            model_runner=ModelRunner(model_worker, output_proc),
            request_builder=request_builder,
            result_adapter=result_adapter,
            request_build_max_workers=request_build_max_workers,
            request_build_max_pending=request_build_max_pending,
            shutdown_callback=(
                audio_encoder_service.close
                if audio_encoder_service is not None
                else None
            ),
        )
    except Exception:
        # The service owns a live worker thread; leaking it would keep the
        # process alive after a failed startup.
        if audio_encoder_service is not None:
            audio_encoder_service.close()
        raise


def create_arkasr_executor(*args, **kwargs):
    return create_sglang_arkasr_executor(*args, **kwargs)


__all__ = ["create_sglang_arkasr_executor", "create_arkasr_executor"]
