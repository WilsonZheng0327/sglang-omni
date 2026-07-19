# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import Any

import torch
from sglang.srt.managers.mm_utils import init_mm_embedding_cache
from transformers import AutoFeatureExtractor, AutoTokenizer

# note(LauraGPT): Auto* loading depends on these local registrations.
import sglang_omni.models.fun_asr.configuration_fun_asr  # noqa: F401
from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.fun_asr.request_builders import (
    fun_asr_prompt_overhead_tokens,
    make_fun_asr_scheduler_adapters,
)
from sglang_omni.models.fun_asr.tool_funcs.audio_lengths import (
    fun_asr_low_frame_rate_length,
)
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


def _compile_fun_asr_audio_encoder(model: Any, *, warmup_lfr_frames: int = 128) -> None:
    """Compile the SANM encoder and adaptor with a symbolic sequence length.

    The LFR frame count varies with audio duration, so ``dynamic=True`` builds
    one symbolic-shape graph instead of specializing per length (which would
    recompile per new length until Dynamo's recompile limit silently falls
    back to eager). The bound forwards are compiled rather than wrapping the
    modules in ``OptimizedModule`` so parameter names stay stable for
    ``load_weights`` and weight updates. The warmup forward pays the one-time
    compile cost at startup instead of on the first request; it runs without
    an explicit grad context because Dynamo guards on grad mode and the omni
    scheduler event loop invokes forwards in ambient mode.
    """
    from sglang.srt.model_executor.cuda_graph_runner import set_torch_compile_config

    if warmup_lfr_frames < 2:
        # Sizes 0/1 are always shape-specialized by Dynamo; warming up with
        # them would not build the symbolic-length graph.
        raise ValueError(f"warmup_lfr_frames must be >= 2, got {warmup_lfr_frames}")
    set_torch_compile_config()
    model.audio_tower.forward = torch.compile(model.audio_tower.forward, dynamic=True)
    model.multi_modal_projector.forward = torch.compile(
        model.multi_modal_projector.forward, dynamic=True
    )
    param = next(model.audio_tower.parameters())
    warmup = torch.zeros(
        (1, int(warmup_lfr_frames), int(model.config.encoder_config.input_size)),
        device=param.device,
        dtype=param.dtype,
    )
    model.multi_modal_projector(model.audio_tower(warmup))
    logger.info(
        "Compiled Fun-ASR audio encoder + adaptor "
        "(dynamic=True, warmup_lfr_frames=%d)",
        warmup_lfr_frames,
    )


def create_sglang_fun_asr_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    max_running_requests: int = 32,
    max_new_tokens: int = 200,
    mem_fraction_static: float | None = None,
    mm_embedding_cache_size_bytes: int = 0,
    enable_torch_compile: bool = False,
    enable_encoder_torch_compile: bool = False,
    mm_attention_backend: str | None = None,
    request_build_max_workers: int = 2,
    request_build_max_pending: int | None = 16,
    server_args_overrides: dict[str, Any] | None = None,
):

    gpu_id = int(device.split(":")[-1]) if ":" in device else 0

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_path, trust_remote_code=True
    )

    encoder_token_count = int(
        fun_asr_low_frame_rate_length(feature_extractor.nb_max_frames)
    )

    # Size the text-prompt overhead from the actual tokenized prompt template
    # (system/user/assistant wrappers + base prompt_text, no hotwords) instead
    # of a fixed guess, so context_length is accurate. Per-request hotword
    # overflow is guarded in the request builder against this context_length.
    prompt_overhead_tokens = fun_asr_prompt_overhead_tokens(tokenizer)
    context_length = encoder_token_count + int(max_new_tokens) + prompt_overhead_tokens

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
        context_length=context_length,
        **overrides,
    )
    validate_generation_batch_policy(
        model_name="Fun-ASR",
        server_args=server_args,
    )

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
        model_arch_override="FunAsrNanoForConditionalGeneration",
    )

    if want_cuda_graph:
        model_worker.model_runner.init_device_graphs()

    # Separate from enable_torch_compile: that flag reaches the sglang server
    # args and compiles only the LLM decode path; the audio encoder runs in
    # eager prefill and is compiled here, after weights are loaded.
    if enable_encoder_torch_compile:
        _compile_fun_asr_audio_encoder(model_worker.model_runner.model)

    init_mm_embedding_cache(mm_embedding_cache_size_bytes)

    output_proc = SGLangOutputProcessor(
        capture_hidden=False,
        capture_hidden_layers=None,
        model=model_worker.model_runner.model,
    )
    request_builder, result_adapter = make_fun_asr_scheduler_adapters(
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        max_new_tokens=max_new_tokens,
        context_length=context_length,
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
    )


def create_fun_asr_executor(*args, **kwargs):
    return create_sglang_fun_asr_executor(*args, **kwargs)


__all__ = ["create_sglang_fun_asr_executor", "create_fun_asr_executor"]
