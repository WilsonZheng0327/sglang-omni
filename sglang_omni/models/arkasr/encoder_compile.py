# SPDX-License-Identifier: Apache-2.0
"""A-PR5: dynamic-shape ``torch.compile`` for the ARK-ASR audio encoder.

The generic ``enable_torch_compile`` server arg only reaches the LM. After A-PR2
moved audio encoding off the LM forward path and onto a dedicated pre-LM worker
thread and CUDA stream, the encoder is the isolated, measurable stage that this
compile targets.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)

# Dynamo shape-specializes sizes 0 and 1, so every stage of the warmup has to
# end up with at least this many frames or the symbolic-length graph is never
# built -- the warmup would compile a specialization that no real request hits.
_MIN_DYNAMIC_FRAMES = 2

# 256 mel frames (~2.6 s of audio) exercises every stage at a size Dynamo will
# not specialize, while keeping startup cheap.
_DEFAULT_WARMUP_MEL_FRAMES = 256


def _min_warmup_mel_frames(merge_factor: int) -> int:
    """Smallest mel length that leaves a non-specialized shape everywhere.

    conv2 has stride 2, so N mel frames leave ``ceil(N / 2)`` post-conv frames,
    and the adapter merges those in groups of ``merge_factor`` before the
    projection MLP. The merged length is the binding constraint:
    ``ceil(N / 2) // merge_factor >= 2`` holds from ``N >= 4 * merge_factor``.
    """
    return 2 * _MIN_DYNAMIC_FRAMES * max(int(merge_factor), 1)


def compile_arkasr_audio_encoder(
    model: Any,
    *,
    warmup_mel_frames: int = _DEFAULT_WARMUP_MEL_FRAMES,
    warmup_inference_mode: bool = True,
) -> None:
    """Compile the Whisper tower and the adapter MLP with a symbolic length.

    ARK builds mels with ``padding="longest"``, so the frame count varies per
    request. ``dynamic=True`` builds one symbolic-shape graph instead of
    specializing per length, which would recompile on every new length until
    Dynamo's recompile limit silently falls back to eager.

    Only ``ArkAudioTower`` and the ``adapting`` MLP are compiled, not
    ``ArkAudioMLPAdapter.forward`` as a whole. The adapter's frame-merge step
    branches on ``seq_len % merge_factor`` and on the trimmed length, which with
    a symbolic length turns into guards that specialize the graph for no gain --
    both arms are a slice and a reshape, while the conv front-end, the encoder
    layers and the projection MLP are the actual work.

    The bound forwards are compiled rather than wrapping the modules in
    ``OptimizedModule`` so parameter names stay stable: ``load_weights`` matches
    checkpoint names against ``named_parameters`` and would miss every audio
    weight behind an ``_orig_mod.`` prefix.

    Default compile mode only. ``reduce-overhead``'s cudagraph trees share an
    allocator pool with the SGLang generation CUDA graphs that always run in
    this process, and the two corrupt each other's memory; bucketed capture of
    the encoder is A-PR7's job, not this one's.

    The warmup pays the compile cost at startup instead of on the first request.
    Dynamo guards on grad mode, so it must run in the same mode as the serving
    caller -- ``torch.inference_mode`` for the pre-LM encoder service
    (``ArkasrPreLMEncoderService._batch_context``), ambient mode for the inline
    ``get_audio_feature`` call on the scheduler loop when the service is off.
    """
    from sglang.srt.compilation.torch_compile_decoration import set_torch_compile_config

    audio_encoder = model.audio_encoder
    merge_factor = int(audio_encoder.merge_factor)
    minimum = _min_warmup_mel_frames(merge_factor)
    if warmup_mel_frames < minimum:
        raise ValueError(
            f"warmup_mel_frames must be >= {minimum} at merge_factor="
            f"{merge_factor}, got {warmup_mel_frames}"
        )

    set_torch_compile_config()
    audio_encoder.whisper.forward = torch.compile(
        audio_encoder.whisper.forward, dynamic=True
    )
    audio_encoder.adapting.forward = torch.compile(
        audio_encoder.adapting.forward, dynamic=True
    )

    param = next(audio_encoder.parameters())
    num_mel_bins = int(model.config.whisper_config.num_mel_bins)
    frames = int(warmup_mel_frames)
    warmup_ctx = (
        torch.inference_mode() if warmup_inference_mode else contextlib.nullcontext()
    )
    with warmup_ctx:
        # The tensors must be *created* inside the context, not merely passed
        # through it: tensors allocated under inference_mode lack the
        # ADInplaceOrView dispatch key and Dynamo guards on the key set, so a
        # normal tensor here would compile a graph that the service's
        # inference-mode tensors fail, forcing a full recompile on the first
        # real request.
        #
        # get_audio_feature produces exactly two signatures: a single item
        # encodes with attention_mask=None, and a group of two or more encodes
        # with a ragged mask. Dynamo specializes batch 1 and treats batch 2 as
        # symbolic, so B1/None + B2/mask covers every serving shape.
        for batch, with_mask in ((1, False), (2, True)):
            mel = torch.zeros(
                (batch, num_mel_bins, frames), device=param.device, dtype=param.dtype
            )
            if with_mask:
                lengths = torch.tensor(
                    [frames, frames // 2], device=param.device, dtype=torch.long
                )
                positions = torch.arange(frames, device=param.device)
                mask = positions.unsqueeze(0) < lengths.unsqueeze(1)
            else:
                mask = None
            audio_encoder(mel, attention_mask=mask)
    logger.info(
        "Compiled ARK-ASR audio tower + adapter MLP (dynamic=True, "
        "warmup_mel_frames=%d, warmup_inference_mode=%s, "
        "signatures=B1/None+B2/mask)",
        frames,
        warmup_inference_mode,
    )


__all__ = ["compile_arkasr_audio_encoder"]
