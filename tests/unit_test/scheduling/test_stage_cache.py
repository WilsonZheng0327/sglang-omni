# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from sglang_omni.scheduling.stage_cache import StageOutputCache, _value_size_bytes


def test_value_size_bytes_counts_byte_buffers() -> None:
    assert _value_size_bytes(b"x" * 1024) == 1024
    assert _value_size_bytes(bytearray(16)) == 16
    assert _value_size_bytes({"a": b"xx", "b": [b"yyy"]}) == 5
    assert _value_size_bytes(torch.zeros(4, dtype=torch.float32)) == 16


def test_stage_output_cache_evicts_on_byte_buffer_size() -> None:
    cache = StageOutputCache(max_bytes=1024)
    cache.put("a", {"payload": b"x" * 800})
    cache.put("b", {"payload": b"y" * 800})
    assert cache.get("a") is None
    assert cache.get("b") is not None
