# SPDX-License-Identifier: Apache-2.0
"""Framework-level tests for PipelineConfig.enabled_stages whitelist filtering.

The mechanism lets users select a subset of stages from a pipeline; required
stages are auto-included; broken edges either resolve via per-stage fallback
declarations or raise actionable compile-time errors.
"""

from __future__ import annotations

import pytest

from sglang_omni.config import PipelineConfig, StageConfig
from sglang_omni.models.qwen3_omni.config import (
    Qwen3OmniPipelineConfig,
    Qwen3OmniSpeechPipelineConfig,
)

_FACTORY = "tests.unit_test.config.test_stage_filter._noop_factory"


def _noop_factory(*args, **kwargs):
    return None


def _stage(
    name: str,
    *,
    next_=None,
    terminal: bool = False,
    wait_for=None,
    merge_fn=None,
    required: bool = False,
    next_fallback=None,
    project_payload=None,
    project_payload_fallback=None,
    stream_to=None,
    process: str | None = None,
) -> StageConfig:
    kwargs: dict = {
        "name": name,
        "factory": _FACTORY,
        "required": required,
        "process": process or name,
    }
    if next_ is not None:
        kwargs["next"] = next_
    if terminal:
        kwargs["terminal"] = True
    if wait_for is not None:
        kwargs["wait_for"] = wait_for
    if merge_fn is not None:
        kwargs["merge_fn"] = merge_fn
    if next_fallback is not None:
        kwargs["next_fallback"] = next_fallback
    if project_payload is not None:
        kwargs["project_payload"] = project_payload
    if project_payload_fallback is not None:
        kwargs["project_payload_fallback"] = project_payload_fallback
    if stream_to is not None:
        kwargs["stream_to"] = stream_to
    return StageConfig(**kwargs)


def _pipeline(stages, *, enabled_stages=None, entry_stage=None) -> PipelineConfig:
    kwargs: dict = {"model_path": "fake/model", "stages": stages}
    if enabled_stages is not None:
        kwargs["enabled_stages"] = enabled_stages
    if entry_stage is not None:
        kwargs["entry_stage"] = entry_stage
    return PipelineConfig(**kwargs)


class TestSchemaFields:
    def test_stage_config_has_required_default_false(self):
        s = _stage("a", terminal=True)
        assert s.required is False

    def test_stage_config_has_next_fallback_default_none(self):
        s = _stage("a", terminal=True)
        assert s.next_fallback is None

    def test_stage_config_has_project_payload_fallback_default_empty(self):
        s = _stage("a", terminal=True)
        assert s.project_payload_fallback == {}

    def test_pipeline_config_has_enabled_stages_default_none(self):
        p = _pipeline([_stage("a", terminal=True)])
        assert p.enabled_stages is None


class TestBackCompat:
    def test_no_filter_preserves_stages(self):
        p = _pipeline(
            [
                _stage("a", next_="b"),
                _stage("b", terminal=True),
            ]
        )
        assert [s.name for s in p.stages] == ["a", "b"]


class TestWhitelistBasics:
    def test_whitelist_keeps_only_listed_stages(self):
        """Parallel branches: dropping one branch leaves the others intact."""
        p = _pipeline(
            [
                _stage("a", next_=["b", "c"]),
                _stage("b", terminal=True),
                _stage("c", terminal=True),
            ],
            enabled_stages=["a", "b"],
        )
        names = {s.name for s in p.stages}
        assert names == {"a", "b"}
        assert "c" not in names

    def test_required_stage_auto_included_when_omitted(self):
        p = _pipeline(
            [
                _stage("a", next_="b"),
                _stage("b", terminal=True, required=True),
            ],
            enabled_stages=["a"],
        )
        names = [s.name for s in p.stages]
        assert "b" in names

    def test_unknown_stage_in_whitelist_raises(self):
        with pytest.raises(ValueError, match="unknown"):
            _pipeline(
                [_stage("a", terminal=True)],
                enabled_stages=["a", "missing"],
            )


class TestNextPruning:
    def test_prunes_disabled_targets_from_next_list(self):
        p = _pipeline(
            [
                _stage("a", next_=["b", "c"]),
                _stage("b", terminal=True),
                _stage("c", terminal=True),
            ],
            enabled_stages=["a", "b"],
        )
        a = next(s for s in p.stages if s.name == "a")
        next_targets = a.next if isinstance(a.next, list) else [a.next]
        assert "c" not in next_targets
        assert "b" in next_targets

    def test_collapse_to_single_target_when_only_one_remains(self):
        p = _pipeline(
            [
                _stage("a", next_=["b", "c"]),
                _stage("b", terminal=True),
                _stage("c", terminal=True),
            ],
            enabled_stages=["a", "b"],
        )
        a = next(s for s in p.stages if s.name == "a")
        if isinstance(a.next, list):
            assert a.next == ["b"]
        else:
            assert a.next == "b"


class TestNextFallback:
    def test_uses_fallback_when_all_next_targets_disabled(self):
        p = _pipeline(
            [
                _stage(
                    "a",
                    next_=["b", "c"],
                    next_fallback=["d"],
                ),
                _stage("b", terminal=True),
                _stage("c", terminal=True),
                _stage("d", terminal=True),
            ],
            enabled_stages=["a", "d"],
        )
        a = next(s for s in p.stages if s.name == "a")
        targets = a.next if isinstance(a.next, list) else [a.next]
        assert "d" in targets
        assert "b" not in targets
        assert "c" not in targets

    def test_no_fallback_when_next_collapses_raises_with_stage_name(self):
        with pytest.raises(ValueError, match=r"\ba\b.*next.*fallback"):
            _pipeline(
                [
                    _stage("a", next_=["b"]),
                    _stage("b", terminal=True),
                    _stage("c", terminal=True),
                ],
                enabled_stages=["a", "c"],
            )

    def test_fallback_target_not_in_whitelist_raises(self):
        with pytest.raises(ValueError, match="next_fallback"):
            _pipeline(
                [
                    _stage(
                        "a",
                        next_=["b"],
                        next_fallback=["d"],
                    ),
                    _stage("b", terminal=True),
                    _stage("d", terminal=True),
                ],
                enabled_stages=["a"],
            )


class TestProjectPayloadFiltering:
    def test_prunes_project_payload_for_disabled_targets(self):
        p = _pipeline(
            [
                _stage(
                    "a",
                    next_=["b", "c"],
                    project_payload={
                        "b": "tests.unit_test.config.test_stage_filter._noop_factory",
                        "c": "tests.unit_test.config.test_stage_filter._noop_factory",
                    },
                ),
                _stage("b", terminal=True),
                _stage("c", terminal=True),
            ],
            enabled_stages=["a", "b"],
        )
        a = next(s for s in p.stages if s.name == "a")
        assert "c" not in a.project_payload
        assert "b" in a.project_payload

    def test_fallback_project_payload_swapped_in_when_next_collapses(self):
        p = _pipeline(
            [
                _stage(
                    "a",
                    next_=["b"],
                    next_fallback=["d"],
                    project_payload={
                        "b": "tests.unit_test.config.test_stage_filter._noop_factory",
                    },
                    project_payload_fallback={
                        "d": "tests.unit_test.config.test_stage_filter._noop_factory",
                    },
                ),
                _stage("b", terminal=True),
                _stage("d", terminal=True),
            ],
            enabled_stages=["a", "d"],
        )
        a = next(s for s in p.stages if s.name == "a")
        assert "d" in a.project_payload
        assert "b" not in a.project_payload


class TestWaitForFiltering:
    def test_prunes_disabled_stages_from_wait_for(self):
        p = _pipeline(
            [
                _stage("a", next_="merge"),
                _stage("b", next_="merge"),
                _stage(
                    "merge",
                    wait_for=["a", "b"],
                    merge_fn="tests.unit_test.config.test_stage_filter._noop_factory",
                    terminal=True,
                ),
            ],
            enabled_stages=["a", "merge"],
        )
        merge = next(s for s in p.stages if s.name == "merge")
        assert merge.wait_for == ["a"]
        assert (
            merge.merge_fn == "tests.unit_test.config.test_stage_filter._noop_factory"
        )


class TestStreamToFiltering:
    def test_prunes_disabled_targets_from_stream_to(self):
        p = _pipeline(
            [
                _stage("a", next_="b", stream_to=["b", "c"]),
                _stage("b", terminal=True),
                _stage("c", terminal=True),
            ],
            enabled_stages=["a", "b"],
        )
        a = next(s for s in p.stages if s.name == "a")
        assert "c" not in a.stream_to
        assert "b" in a.stream_to


class TestRequiredEnforcement:
    def test_required_stage_auto_inclusion_visible_in_final_stage_list(self):
        p = _pipeline(
            [
                _stage("a", next_="b"),
                _stage("b", next_="c", required=True),
                _stage("c", terminal=True, required=True),
            ],
            enabled_stages=["a"],
        )
        names = {s.name for s in p.stages}
        assert {"a", "b", "c"} <= names


class TestBrokenGraphValidation:
    def test_qwen_text_filter_rejects_omitted_original_entry(self):
        with pytest.raises(ValueError, match="entry stage 'preprocessing'"):
            Qwen3OmniPipelineConfig(
                model_path="fake/Qwen3-Omni-30B-A3B-Instruct",
                enabled_stages=["decode"],
            )

    def test_qwen_speech_filter_rejects_unreachable_retained_terminal(self):
        with pytest.raises(ValueError, match="unreachable.*code2wav"):
            Qwen3OmniSpeechPipelineConfig(
                model_path="fake/Qwen3-Omni-30B-A3B-Instruct",
                enabled_stages=["preprocessing", "code2wav"],
            )

    def test_direct_empty_next_list_is_rejected(self):
        with pytest.raises(ValueError, match="next.*empty"):
            _stage("a", next_=[])

    def test_empty_next_fallback_list_is_rejected(self):
        with pytest.raises(ValueError, match="next_fallback.*empty"):
            _stage("a", next_="b", next_fallback=[])
