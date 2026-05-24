# SPDX-License-Identifier: Apache-2.0
"""Standalone execute pipeline for non-AR encoder stages."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import torch

from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.stage_cache import StageOutputCache
from sglang_omni.utils.cuda_capture import cuda_capture_guard

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class EncoderBatchItem:
    index: int
    payload: StagePayload
    state: Any
    request: Any


@dataclass(slots=True)
class EncoderCudaGraphStats:
    hits: int = 0
    misses: int = 0
    captures: int = 0
    fallbacks: int = 0


def tensor_bytes(value: Any) -> int:
    if not isinstance(value, torch.Tensor):
        return 0
    return int(value.numel() * value.element_size())


def nested_tensor_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return tensor_bytes(value)
    if isinstance(value, dict):
        return sum(nested_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(nested_tensor_bytes(item) for item in value)
    return 0


def _encoder_cache_trace_enabled() -> bool:
    value = os.getenv("SGLANG_OMNI_TRACE_ENCODER_CACHE", "")
    return value.lower() not in ("", "0", "false", "no")


def _short_cache_key(cache_key: str | None) -> str:
    if not cache_key:
        return "-"
    if len(cache_key) <= 32:
        return cache_key
    return f"{cache_key[:16]}...{cache_key[-8:]}"


class EncoderModelRunner:
    """Common request/cache/batch lifecycle for encoder stages.

    Encoders are not AR model runners: they do not build ``ForwardBatch``, sample
    tokens, or own KV-cache state. Subclasses provide model-specific request
    building and the ``prepare -> forward -> post`` batch hooks.
    """

    def __init__(
        self,
        *,
        model: Any,
        stage_name: str,
        cache: StageOutputCache | None = None,
        enable_cuda_graph: bool = True,
    ) -> None:
        self.model = model
        self.stage_name = stage_name
        self.cache = cache
        self.enable_cuda_graph = enable_cuda_graph
        self.cuda_graphs: dict[Any, torch.cuda.CUDAGraph] = {}
        self.cuda_graph_static_inputs: dict[Any, Any] = {}
        self.cuda_graph_input_buffers: dict[Any, dict[str, torch.Tensor]] = {}
        self.cuda_graph_metadata_buffers: dict[Any, dict[str, torch.Tensor]] = {}
        self.cuda_graph_output_buffers: dict[Any, Any] = {}
        self.cuda_graph_stats = EncoderCudaGraphStats()

    def execute(self, payload: StagePayload) -> StagePayload:
        return self.execute_batch([payload])[0]

    def execute_batch(self, payloads: list[StagePayload]) -> list[StagePayload]:
        results: list[StagePayload | None] = [None] * len(payloads)
        active: list[EncoderBatchItem] = []
        duplicate_waiters: dict[str, list[EncoderBatchItem]] = {}
        active_cache_keys: set[str] = set()
        active_cache_leaders: dict[str, str] = {}

        for idx, payload in enumerate(payloads):
            state = self.load_state(payload)
            request = self.build_encoder_request(payload, state)
            skip_result = self.request_skip_result(request)
            if skip_result is not None:
                self._apply_payload_result(
                    EncoderBatchItem(idx, payload, state, request),
                    skip_result,
                    results,
                )
                continue

            cached = self.lookup_cached_output(
                request=request,
                request_id=payload.request_id,
            )
            if cached is not None:
                self._apply_payload_result(
                    EncoderBatchItem(idx, payload, state, request),
                    cached,
                    results,
                )
                continue

            item = EncoderBatchItem(idx, payload, state, request)
            if not self.is_batchable(request):
                self._compute_single_item(item, results)
                continue

            cache_key = self.request_cache_key(request)
            if cache_key is not None and cache_key in active_cache_keys:
                duplicate_waiters.setdefault(cache_key, []).append(item)
                self.trace_cache_event(
                    "dedup_same_batch",
                    request_id=payload.request_id,
                    cache_key=cache_key,
                    input_bytes=nested_tensor_bytes(self.request_model_inputs(request)),
                    detail=f"leader={active_cache_leaders[cache_key]}",
                )
                continue

            active.append(item)
            if cache_key is not None:
                active_cache_keys.add(cache_key)
                active_cache_leaders[cache_key] = payload.request_id

        computed_by_cache_key: dict[str, Any] = {}
        if active:
            prepared = self.prepare(active)
            with torch.no_grad():
                combined = self.forward(prepared)
            batch_results = self.post(prepared, combined)
            if len(batch_results) != len(active):
                raise ValueError(
                    f"{type(self).__name__}.post returned {len(batch_results)} "
                    f"results for {len(active)} encoder requests"
                )

            for item, result in zip(active, batch_results):
                self.store_cached_output(
                    request=item.request,
                    request_id=item.payload.request_id,
                    result=result,
                )
                cache_key = self.request_cache_key(item.request)
                if cache_key is not None:
                    computed_by_cache_key[cache_key] = result
                self._apply_payload_result(item, result, results)

        for cache_key, waiters in duplicate_waiters.items():
            if cache_key not in computed_by_cache_key:
                continue
            result = computed_by_cache_key[cache_key]
            for item in waiters:
                self._apply_payload_result(item, result, results)

        missing = [
            payloads[idx].request_id
            for idx, payload_result in enumerate(results)
            if payload_result is None
        ]
        if missing:
            raise RuntimeError(
                f"{type(self).__name__} did not produce results for {missing}"
            )
        return [
            payload_result for payload_result in results if payload_result is not None
        ]

    def estimate_payload_cost(self, payload: StagePayload) -> int:
        state = self.load_state(payload)
        request = self.build_encoder_request(payload, state)
        if self.request_skip_result(request) is not None:
            return 0
        return self.estimate_request_cost(request)

    def load_state(self, payload: StagePayload) -> Any:
        raise NotImplementedError

    def store_state(self, payload: StagePayload, state: Any) -> StagePayload:
        raise NotImplementedError

    def build_encoder_request(self, payload: StagePayload, state: Any) -> Any:
        raise NotImplementedError

    def apply_result(self, state: Any, result: Any) -> None:
        raise NotImplementedError

    def is_batchable(self, request: Any) -> bool:
        del request
        return False

    def estimate_request_cost(self, request: Any) -> int:
        del request
        return 0

    def prepare(self, items: list[EncoderBatchItem]) -> Any:
        return items

    def forward(self, prepared: Any) -> Any:
        if self.can_run_cuda_graph(prepared):
            return self.forward_cuda_graph(prepared)
        return self.forward_eager(prepared)

    def forward_eager(self, prepared: Any) -> Any:
        items = prepared
        if not isinstance(items, list) or len(items) != 1:
            raise NotImplementedError(
                f"{type(self).__name__}.forward_eager must handle batched inputs"
            )
        return self.forward_single(items[0].request)

    def can_run_cuda_graph(self, prepared: Any) -> bool:
        return (
            self.enable_cuda_graph
            and torch.cuda.is_available()
            and self.cuda_graph_key(prepared) is not None
        )

    def cuda_graph_key(self, prepared: Any) -> Any | None:
        del prepared
        return None

    def forward_cuda_graph(self, prepared: Any) -> Any:
        graph_key = self.cuda_graph_key(prepared)
        if graph_key is None:
            return self.forward_eager(prepared)

        return self.run_cuda_graph_piece(graph_key, prepared)

    def run_cuda_graph_piece(self, graph_key: Any, prepared: Any) -> Any:
        if graph_key not in self.cuda_graphs:
            self.cuda_graph_stats.misses += 1
            self.capture_cuda_graph(graph_key, prepared)
        else:
            self.cuda_graph_stats.hits += 1

        self.prepare_cuda_graph_replay(graph_key, prepared)
        self.cuda_graphs[graph_key].replay()
        return self.get_cuda_graph_output(graph_key, prepared)

    def prepare_between_graphs(self, value: Any, prepared: Any) -> Any:
        """Hook for encoders with multiple captured islands in one forward."""
        del prepared
        return value

    def capture_cuda_graph(self, graph_key: Any, prepared: Any) -> None:
        if graph_key in self.cuda_graphs:
            return
        static_prepared = self.prepare_cuda_graph_capture(graph_key, prepared)
        self.cuda_graph_static_inputs[graph_key] = static_prepared

        graph = torch.cuda.CUDAGraph()
        with cuda_capture_guard():
            torch.cuda.synchronize()
            with torch.cuda.graph(graph):
                self.cuda_graph_output_buffers[graph_key] = (
                    self.forward_cuda_graph_capture(graph_key, static_prepared)
                )
        self.cuda_graphs[graph_key] = graph
        self.cuda_graph_stats.captures += 1
        logger.info(
            "encoder_cuda_graph_captured stage=%s graph_key=%r",
            self.stage_name,
            graph_key,
        )

    def prepare_cuda_graph_capture(self, graph_key: Any, prepared: Any) -> Any:
        del graph_key, prepared
        raise NotImplementedError

    def prepare_cuda_graph_replay(self, graph_key: Any, prepared: Any) -> None:
        del graph_key, prepared
        raise NotImplementedError

    def forward_cuda_graph_capture(self, graph_key: Any, static_prepared: Any) -> Any:
        del graph_key
        return self.forward_eager(static_prepared)

    def get_cuda_graph_output(self, graph_key: Any, prepared: Any) -> Any:
        del prepared
        return self.cuda_graph_output_buffers[graph_key]

    def static_input_buffer(
        self,
        graph_key: Any,
        name: str,
        *,
        shape: tuple[int, ...] | torch.Size,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> torch.Tensor:
        return self._static_buffer(
            self.cuda_graph_input_buffers,
            graph_key,
            name,
            shape=shape,
            dtype=dtype,
            device=device,
        )

    def static_metadata_buffer(
        self,
        graph_key: Any,
        name: str,
        *,
        shape: tuple[int, ...] | torch.Size,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> torch.Tensor:
        return self._static_buffer(
            self.cuda_graph_metadata_buffers,
            graph_key,
            name,
            shape=shape,
            dtype=dtype,
            device=device,
        )

    def _static_buffer(
        self,
        buffer_store: dict[Any, dict[str, torch.Tensor]],
        graph_key: Any,
        name: str,
        *,
        shape: tuple[int, ...] | torch.Size,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> torch.Tensor:
        buffers = buffer_store.setdefault(graph_key, {})
        requested_shape = tuple(shape)
        requested_device = torch.device(device)
        buffer = buffers.get(name)
        if buffer is not None:
            if (
                tuple(buffer.shape) == requested_shape
                and buffer.dtype == dtype
                and buffer.device == requested_device
            ):
                return buffer
            if graph_key in self.cuda_graphs:
                raise RuntimeError(
                    f"Cannot resize CUDA graph buffer {name!r} for captured "
                    f"graph key {graph_key!r}"
                )

        buffers[name] = torch.empty(
            requested_shape,
            dtype=dtype,
            device=requested_device,
        )
        return buffers[name]

    def post(self, prepared: Any, combined: Any) -> list[Any]:
        items = prepared
        if not isinstance(items, list) or len(items) != 1:
            raise NotImplementedError(
                f"{type(self).__name__}.post must split batched outputs"
            )
        return [self.post_single(items[0].request, combined)]

    def forward_single(self, request: Any) -> Any:
        return self.model(**self.request_model_inputs(request))

    def post_single(self, request: Any, result: Any) -> Any:
        del request
        return result

    def request_model_inputs(self, request: Any) -> dict[str, Any]:
        return request.model_inputs

    def request_cache_key(self, request: Any) -> str | None:
        cache_key = request.cache_key
        return str(cache_key) if cache_key is not None else None

    def request_skip_result(self, request: Any) -> Any | None:
        return request.skip_result

    def lookup_cached_output(self, *, request: Any, request_id: str) -> Any | None:
        cache_key = self.request_cache_key(request)
        if self.cache is None or cache_key is None:
            return None
        cached = self.cache.get(cache_key)
        if cached is None:
            self.trace_cache_event(
                "miss",
                request_id=request_id,
                cache_key=cache_key,
                input_bytes=nested_tensor_bytes(self.request_model_inputs(request)),
            )
            return None
        self.trace_cache_event(
            "hit",
            request_id=request_id,
            cache_key=cache_key,
            input_bytes=nested_tensor_bytes(self.request_model_inputs(request)),
            output_bytes=nested_tensor_bytes(cached),
        )
        return cached

    def store_cached_output(
        self, *, request: Any, request_id: str, result: Any
    ) -> None:
        cache_key = self.request_cache_key(request)
        if self.cache is None or cache_key is None:
            return
        self.cache.put(cache_key, result)
        self.trace_cache_event(
            "store",
            request_id=request_id,
            cache_key=cache_key,
            input_bytes=nested_tensor_bytes(self.request_model_inputs(request)),
            output_bytes=nested_tensor_bytes(result),
        )

    def trace_cache_event(
        self,
        action: str,
        *,
        request_id: str,
        cache_key: str | None,
        input_bytes: int | None = None,
        output_bytes: int | None = None,
        detail: str | None = None,
    ) -> None:
        if not _encoder_cache_trace_enabled():
            return
        parts = [
            f"stage={self.stage_name}",
            f"action={action}",
            f"req={request_id}",
            f"key={_short_cache_key(cache_key)}",
        ]
        if input_bytes is not None:
            parts.append(f"input_bytes={input_bytes}")
        if output_bytes is not None:
            parts.append(f"output_bytes={output_bytes}")
        if detail:
            parts.append(detail)
        logger.info("encoder_cache %s", " ".join(parts))

    def _compute_single_item(
        self,
        item: EncoderBatchItem,
        results: list[StagePayload | None],
    ) -> None:
        with torch.no_grad():
            result = self.forward_single(item.request)
        result = self.post_single(item.request, result)
        self.store_cached_output(
            request=item.request,
            request_id=item.payload.request_id,
            result=result,
        )
        self._apply_payload_result(item, result, results)

    def _apply_payload_result(
        self,
        item: EncoderBatchItem,
        result: Any,
        results: list[StagePayload | None],
    ) -> None:
        self.apply_result(item.state, result)
        results[item.index] = self.store_state(item.payload, item.state)
