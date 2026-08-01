# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.arkasr.encoder_service import (
    ArkASRPreLMEncoderService,
    _expected_audio_tokens,
    build_cache_namespace,
)

_HIDDEN_SIZE = 4
_NAMESPACE = "testns"
_SERVICES: list[ArkASRPreLMEncoderService] = []


@pytest.fixture(autouse=True)
def _close_services() -> Iterator[None]:
    yield
    for service in _SERVICES:
        service.close()
    _SERVICES.clear()


class _StubModel(torch.nn.Module):
    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        # ARK's audio stack is exposed as ``audio_encoder`` (tower + adapter),
        # not ``audio_tower``.
        self.audio_encoder = torch.nn.Linear(2, 2).to(dtype)
        self.config = SimpleNamespace(hidden_size=_HIDDEN_SIZE)
        self.dtype = dtype
        self.encode_calls = 0
        self.fail = False
        self.fail_multi_item = False
        self.encode_gate: threading.Event | None = None
        self.row_offset = 0
        self.encode_delay_s = 0.0
        self.grad_enabled_during_encode: bool | None = None

    def get_audio_feature(self, items):  # noqa: ANN001
        self.grad_enabled_during_encode = torch.is_grad_enabled()
        self.encode_calls += 1
        gate = self.encode_gate
        if gate is not None:
            self.encode_gate = None
            gate.wait(timeout=10)
        if self.encode_delay_s:
            time.sleep(self.encode_delay_s)
        if self.fail:
            raise RuntimeError("boom")
        if self.fail_multi_item and len(items) > 1:
            raise RuntimeError("multi-item boom")
        parts = []
        for item in items:
            rows = _expected_audio_tokens(item) + self.row_offset
            fill = float((getattr(item, "hash", None) or 0) % 97 + 1)
            parts.append(torch.full((rows, _HIDDEN_SIZE), fill, dtype=self.dtype))
        return torch.cat(parts, dim=0)


def _make_service(
    model: _StubModel | None = None,
    *,
    cache_max_entries: int = 16,
    cache_max_bytes: int = 1 << 20,
) -> ArkASRPreLMEncoderService:
    service = ArkASRPreLMEncoderService(
        model or _StubModel(),
        cache_namespace=_NAMESPACE,
        cache_max_entries=cache_max_entries,
        cache_max_bytes=cache_max_bytes,
    )
    _SERVICES.append(service)
    return service


def _item(
    audio_hash: int | None,
    num_audio_tokens: int,
    *,
    with_feature: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        hash=audio_hash,
        audio_fingerprint=str(audio_hash) if audio_hash is not None else None,
        num_audio_tokens=num_audio_tokens,
        feature=torch.zeros(1, 128, 8) if with_feature else None,
        precomputed_embeddings=None,
    )


def test_encode_attaches_lm_ready_embedding_and_clears_feature() -> None:
    model = _StubModel()
    service = _make_service(model)
    item = _item(7, 3)

    service.encode_item(item)

    assert item.precomputed_embeddings.shape == (3, _HIDDEN_SIZE)
    assert item.precomputed_embeddings.dtype == model.dtype
    assert (
        item.precomputed_embeddings.device
        == next(model.audio_encoder.parameters()).device
    )
    assert item.feature is None
    assert item.format.name == "PRECOMPUTED_EMBEDDING"
    assert model.encode_calls == 1
    # inference_mode, not just no_grad: the encode runs off the scheduler loop.
    assert model.grad_enabled_during_encode is False
    assert service.stats()["misses"] == 1


def test_close_stops_worker() -> None:
    service = _make_service()

    service.close()

    assert not service._thread.is_alive()


def test_cache_hit_skips_reencode() -> None:
    model = _StubModel()
    service = _make_service(model)

    first = _item(11, 3)
    second = _item(11, 3)
    service.encode_item(first)
    service.encode_item(second)

    assert model.encode_calls == 1
    assert torch.equal(first.precomputed_embeddings, second.precomputed_embeddings)
    assert second.feature is None
    assert service.stats()["hits"] == 1


def test_distinct_audio_never_reuses_another_clip_embedding() -> None:
    model = _StubModel()
    service = _make_service(model)

    short = _item(111, 3)
    extended = _item(222, 5)
    service.encode_item(short)
    service.encode_item(extended)

    assert model.encode_calls == 2
    assert extended.precomputed_embeddings.shape == (5, _HIDDEN_SIZE)
    assert len(service._cache) == 2
    assert not torch.equal(
        short.precomputed_embeddings[0], extended.precomputed_embeddings[0]
    )


def test_cache_key_uses_full_waveform_fingerprint() -> None:
    model = _StubModel()
    service = _make_service(model)
    first = _item(7, 3)
    second = _item(7, 3)
    first.audio_fingerprint = "full-hash-a"
    second.audio_fingerprint = "full-hash-b"

    service.encode_item(first)
    service.encode_item(second)

    assert model.encode_calls == 2
    assert len(service._cache) == 2


def test_concurrent_identical_requests_encode_once() -> None:
    model = _StubModel()
    model.encode_delay_s = 0.05
    service = _make_service(model)
    n_threads = 8
    barrier = threading.Barrier(n_threads)
    items = [_item(123, 3) for _ in range(n_threads)]
    errors: list[BaseException] = []

    def worker(item: SimpleNamespace) -> None:
        try:
            barrier.wait(timeout=10)
            service.encode_item(item)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(item,)) for item in items]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors, errors
    assert model.encode_calls == 1
    for item in items:
        assert item.precomputed_embeddings.shape == (3, _HIDDEN_SIZE)
        assert torch.equal(item.precomputed_embeddings, items[0].precomputed_embeddings)
    stats = service.stats()
    assert stats["merged"] + stats["hits"] == n_threads - 1


def test_encode_failure_propagates_without_poisoning_cache() -> None:
    model = _StubModel()
    model.fail = True
    model.encode_delay_s = 0.05
    service = _make_service(model)
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            barrier.wait(timeout=10)
            service.encode_item(_item(55, 3))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert len(errors) == 2
    assert all(isinstance(exc, RuntimeError) and "boom" in str(exc) for exc in errors)
    assert len(service._cache) == 0
    assert service.stats()["failed"] == 2

    model.fail = False
    item = _item(55, 3)
    service.encode_item(item)
    assert item.precomputed_embeddings.shape == (3, _HIDDEN_SIZE)


def test_multi_item_batch_failure_retries_per_item_and_counts_stats() -> None:
    model = _StubModel()
    model.fail_multi_item = True
    gate = threading.Event()
    model.encode_gate = gate
    service = _make_service(model)
    items = [_item(31, 3), _item(32, 3), _item(33, 4)]
    errors: list[BaseException] = []

    def worker(item: SimpleNamespace) -> None:
        try:
            service.encode_item(item)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(item,)) for item in items]
    for thread in threads:
        thread.start()
    # Queue every leader before releasing the gate so the next drain exercises
    # the multi-item retry path.
    deadline = time.monotonic() + 5
    while len(service._inflight) < 3 and time.monotonic() < deadline:
        time.sleep(0.005)
    assert len(service._inflight) == 3, "items never queued"
    gate.set()
    for thread in threads:
        thread.join(timeout=30)

    assert not errors, errors
    for item in items:
        assert item.precomputed_embeddings.shape == (
            item.num_audio_tokens,
            _HIDDEN_SIZE,
        )
    stats = service.stats()
    assert stats["failed"] == 0
    assert stats["items"] == 3
    assert stats["batches"] == 3
    assert model.encode_calls == 4
    assert len(service._cache) == 3


def test_eviction_under_byte_budget_triggers_reencode() -> None:
    model = _StubModel()
    service = _make_service(model, cache_max_bytes=100)

    for audio_hash in (1, 2, 3):
        service.encode_item(_item(audio_hash, 3))
    assert model.encode_calls == 3
    assert service._cache.eviction_count >= 1
    assert len(service._cache) == 2

    service.encode_item(_item(1, 3))
    assert model.encode_calls == 4


def test_token_count_mismatch_fails_loudly() -> None:
    model = _StubModel()
    model.row_offset = 1
    service = _make_service(model)
    item = _item(9, 3)

    with pytest.raises(RuntimeError, match="!= expected rows"):
        service.encode_item(item)

    assert item.precomputed_embeddings is None
    assert len(service._cache) == 0


def test_missing_token_count_raises() -> None:
    service = _make_service()
    item = SimpleNamespace(hash=1, feature=None, precomputed_embeddings=None)

    with pytest.raises(RuntimeError, match="num_audio_tokens"):
        service.encode_item(item)


def test_item_without_fingerprint_encodes_without_caching() -> None:
    model = _StubModel()
    service = _make_service(model)

    first = _item(1, 2)
    second = _item(1, 2)
    first.audio_fingerprint = None
    second.audio_fingerprint = None
    service.encode_item(first)
    service.encode_item(second)

    assert model.encode_calls == 2
    assert first.feature is None
    assert first.precomputed_embeddings.shape == (2, _HIDDEN_SIZE)
    assert len(service._cache) == 0


def test_expected_audio_tokens_uses_request_metadata() -> None:
    explicit = SimpleNamespace(num_audio_tokens=5, feature=torch.zeros(1, 128, 17))
    assert _expected_audio_tokens(explicit) == 5
    assert _expected_audio_tokens(SimpleNamespace()) is None


def test_build_cache_namespace_is_stable_and_scoped() -> None:
    model = _StubModel()
    frontend = SimpleNamespace(
        feature_size=128,
        sampling_rate=16000,
        hop_length=160,
        n_fft=400,
        chunk_length=30,
        n_samples=480000,
        nb_max_frames=3000,
    )
    base = dict(
        model_path="AutoArk-AI/ARK-ASR-3B",
        feature_extractor=frontend,
        mm_attention_backend=None,
    )

    namespace = build_cache_namespace(model, **base)
    assert namespace == build_cache_namespace(model, **base)
    assert namespace != build_cache_namespace(
        model, **{**base, "model_path": "other/revision"}
    )
    assert namespace != build_cache_namespace(
        model, **{**base, "mm_attention_backend": "triton_attn"}
    )
    assert namespace != build_cache_namespace(_StubModel(dtype=torch.bfloat16), **base)
    changed_frontend = SimpleNamespace(**{**vars(frontend), "hop_length": 320})
    assert namespace != build_cache_namespace(
        model, **{**base, "feature_extractor": changed_frontend}
    )
    changed_config = _StubModel()
    changed_config.config = SimpleNamespace(hidden_size=_HIDDEN_SIZE, marker="other")
    assert namespace != build_cache_namespace(changed_config, **base)
