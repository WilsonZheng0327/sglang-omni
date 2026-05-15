# SPDX-License-Identifier: Apache-2.0
"""CLI argparse + lane-semantics tests for benchmark_omni_mmmu.

The lane-A / lane-B contract is non-negotiable (AC-10): Lane B forces
``ignore_eos=True`` and ``max_tokens=256``; explicit overrides are
rejected. The ``--stream`` × ``--enable-audio`` cross-product is also
rejected (AC-1). These tests exercise the argparse → MMMUEvalConfig
resolution to lock those guarantees.
"""

from __future__ import annotations

import argparse

import pytest


def _parser_namespace(**overrides) -> argparse.Namespace:
    """Build an argparse.Namespace with the same fields the eval CLI parses."""
    defaults = dict(
        base_url=None,
        host="localhost",
        port=8000,
        model="qwen3-omni",
        output_dir=None,
        max_samples=None,
        max_tokens=None,
        temperature=0.0,
        warmup=0,
        max_concurrency=1,
        request_rate=float("inf"),
        disable_tqdm=False,
        enable_audio=False,
        asr_device="cuda:0",
        lang="en",
        repo_id=None,
        backend="omni",
        stream=False,
        seed=42,
        ignore_eos=False,
        lane="A",
        reps=3,
        repetition_index=0,
        dataset_revisions=None,
        preflight_json=None,
        launcher_log=None,
        mem_fraction_static=None,
        prefix_cache_disabled=True,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_lane_a_defaults() -> None:
    from benchmarks.eval.benchmark_omni_mmmu import _config_from_args

    cfg = _config_from_args(_parser_namespace(lane="A"))
    assert cfg.lane == "A"
    assert cfg.ignore_eos is False
    assert cfg.max_tokens == 2048


def test_lane_b_locks_ignore_eos_true_and_max_tokens_256() -> None:
    from benchmarks.eval.benchmark_omni_mmmu import _config_from_args

    cfg = _config_from_args(_parser_namespace(lane="B"))
    assert cfg.lane == "B"
    assert cfg.ignore_eos is True
    assert cfg.max_tokens == 256


def test_lane_b_with_explicit_max_tokens_256_is_accepted() -> None:
    from benchmarks.eval.benchmark_omni_mmmu import _config_from_args

    cfg = _config_from_args(_parser_namespace(lane="B", max_tokens=256))
    assert cfg.max_tokens == 256


def test_lane_b_rejects_max_tokens_override() -> None:
    from benchmarks.eval.benchmark_omni_mmmu import _config_from_args

    with pytest.raises(SystemExit, match="Lane B"):
        _config_from_args(_parser_namespace(lane="B", max_tokens=2048))


def test_lane_b_accepts_redundant_ignore_eos_true() -> None:
    """--ignore-eos with --lane B is a no-op redundancy, not a contradiction."""
    from benchmarks.eval.benchmark_omni_mmmu import _config_from_args

    cfg = _config_from_args(_parser_namespace(lane="B", ignore_eos=True))
    assert cfg.ignore_eos is True


def test_stream_with_enable_audio_is_rejected() -> None:
    from benchmarks.eval.benchmark_omni_mmmu import _config_from_args

    with pytest.raises(SystemExit, match="stream"):
        _config_from_args(_parser_namespace(stream=True, enable_audio=True))


def test_invalid_lane_is_rejected() -> None:
    from benchmarks.eval.benchmark_omni_mmmu import _config_from_args

    with pytest.raises(SystemExit, match="lane"):
        _config_from_args(_parser_namespace(lane="C"))


def test_lane_a_allows_max_tokens_override() -> None:
    from benchmarks.eval.benchmark_omni_mmmu import _config_from_args

    cfg = _config_from_args(_parser_namespace(lane="A", max_tokens=512))
    assert cfg.max_tokens == 512


def test_lane_a_optional_ignore_eos() -> None:
    """Lane A normally has ignore_eos=False, but the flag is allowed."""
    from benchmarks.eval.benchmark_omni_mmmu import _config_from_args

    cfg = _config_from_args(_parser_namespace(lane="A", ignore_eos=True))
    assert cfg.ignore_eos is True


def test_run_metadata_contains_all_ac9_fields() -> None:
    """AC-9 validator: the emitted run-metadata block has every required key."""
    from benchmarks.eval.benchmark_omni_mmmu import (
        MMMUEvalConfig,
        _build_run_metadata,
    )
    from benchmarks.scripts.run_metadata import REQUIRED_FIELDS, validate

    cfg = MMMUEvalConfig(model="qwen3-omni", lane="A")
    meta = _build_run_metadata(cfg)
    missing = validate(meta)
    assert missing == [], f"run_metadata missing fields: {missing}"
    # Sanity check that REQUIRED_FIELDS is comprehensive (no shrinkage from
    # the dataclass definition).
    for field_name in REQUIRED_FIELDS:
        assert field_name in meta


def test_run_metadata_routes_container_by_backend() -> None:
    """Container name + image fields follow the --backend choice."""
    from benchmarks.eval.benchmark_omni_mmmu import (
        MMMUEvalConfig,
        _build_run_metadata,
    )

    omni_meta = _build_run_metadata(MMMUEvalConfig(model="m", backend="omni"))
    assert omni_meta["container_name"] == "sglang-omni-hayden-benchmark"
    assert omni_meta["container_image"] == "frankleeeee/sglang-omni:dev"

    sglang_meta = _build_run_metadata(MMMUEvalConfig(model="m", backend="sglang"))
    assert sglang_meta["container_name"] == "sglang-hayden-benchmark"
    assert sglang_meta["container_image"] == "lmsysorg/sglang"


def test_run_metadata_merges_model_revision_from_preflight_json(tmp_path) -> None:
    """AC-9 value-source: model_revision comes from preflight.json when provided."""
    import json

    from benchmarks.eval.benchmark_omni_mmmu import (
        MMMUEvalConfig,
        _build_run_metadata,
    )

    preflight = {
        "ok": True,
        "model_revisions": {
            "Qwen/Qwen3-Omni-30B-A3B-Instruct": "abc123def456",
        },
        "containers": {
            "sglang-omni-hayden-benchmark": {
                "container_image_digest": "sha256:deadbeef",
            },
        },
    }
    p = tmp_path / "preflight.json"
    p.write_text(json.dumps(preflight))

    cfg = MMMUEvalConfig(model="qwen3-omni", backend="omni", preflight_json=str(p))
    meta = _build_run_metadata(cfg)
    assert meta["model_revision"] == "abc123def456"
    assert meta["container_image_digest"] == "sha256:deadbeef"


def test_run_metadata_loads_dataset_revisions() -> None:
    """AC-9 value-source: dataset_revisions comes from the JSON pin file."""
    from benchmarks.eval.benchmark_omni_mmmu import (
        MMMUEvalConfig,
        _build_run_metadata,
    )

    cfg = MMMUEvalConfig(model="qwen3-omni", backend="omni")
    meta = _build_run_metadata(cfg)
    # The checked-in mmmu_revisions.json has MMMU/MMMU populated; the
    # metadata must reflect it, not an empty dict.
    assert "MMMU/MMMU" in meta["dataset_revisions"]
    assert len(meta["dataset_revisions"]["MMMU/MMMU"]) == 40  # SHA-1 length


def test_run_metadata_scrapes_kv_capacity_from_launcher_log(tmp_path) -> None:
    """AC-9 value-source: kv_cache_capacity_tokens comes from launcher log."""
    from benchmarks.eval.benchmark_omni_mmmu import (
        MMMUEvalConfig,
        _build_run_metadata,
    )

    log = tmp_path / "launcher.log"
    log.write_text(
        "[INFO] startup ...\n"
        "[INFO] KV-cache pool: capacity 123456 tokens (bf16)\n"
        "[INFO] ready\n"
    )
    cfg = MMMUEvalConfig(model="x", backend="omni", launcher_log=str(log))
    meta = _build_run_metadata(cfg)
    assert meta["kv_cache_capacity_tokens"] == 123456


def test_run_metadata_records_mem_fraction_and_prefix_cache_policy() -> None:
    """AC-9 value-source: mem_fraction + prefix_cache come from launch flags."""
    from benchmarks.eval.benchmark_omni_mmmu import (
        MMMUEvalConfig,
        _build_run_metadata,
    )

    cfg = MMMUEvalConfig(
        model="x",
        backend="omni",
        mem_fraction_static=0.87,
        prefix_cache_disabled=True,
    )
    meta = _build_run_metadata(cfg)
    assert meta["mem_fraction_static_configured"] == 0.87
    assert meta["prefix_cache_disabled"] is True


def test_run_metadata_failure_count_from_request_results() -> None:
    """AC-9 value-source: failure_count comes from request_results, not 0."""
    from benchmarks.benchmarker.data import RequestResult
    from benchmarks.eval.benchmark_omni_mmmu import (
        MMMUEvalConfig,
        _build_run_metadata,
    )

    cfg = MMMUEvalConfig(model="x", backend="omni")
    results = [
        RequestResult(request_id="1", is_success=True),
        RequestResult(request_id="2", is_success=False, error="boom"),
        RequestResult(request_id="3", is_success=False, error="oom"),
        RequestResult(request_id="4", is_success=True),
    ]
    meta = _build_run_metadata(cfg, request_results=results)
    assert meta["failure_count"] == 2
