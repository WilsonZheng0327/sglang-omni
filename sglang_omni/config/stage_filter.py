# SPDX-License-Identifier: Apache-2.0
"""Whitelist-based stage selection for pipeline configs."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang_omni.config.schema import StageConfig


def _as_list(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _apply_stage_filter(
    stages: "list[StageConfig]",
    enabled_stages: list[str],
    *,
    entry_stage: str | None,
) -> "list[StageConfig]":
    """Return a new stage list filtered by the whitelist.

    Required stages are auto-included. Edges are pruned or rewired with
    per-stage fallbacks, then the retained graph must remain reachable from
    the original resolved entry stage.
    """
    all_names = {s.name for s in stages}
    requested = set(enabled_stages)

    unknown = requested - all_names
    if unknown:
        raise ValueError(
            f"enabled_stages references unknown stages: {sorted(unknown)}. "
            f"Known stages: {sorted(all_names)}."
        )

    auto_included = {s.name for s in stages if s.required}
    effective = requested | auto_included
    if entry_stage is not None and entry_stage not in all_names:
        raise ValueError(f"entry_stage {entry_stage!r} is not defined")
    if entry_stage is not None and entry_stage not in effective:
        raise ValueError(
            f"enabled_stages must retain entry stage {entry_stage!r}. "
            f"Add {entry_stage!r} to enabled_stages or set entry_stage to a "
            f"retained stage that can safely accept new requests."
        )

    result: list[StageConfig] = []
    for stage in stages:
        if stage.name not in effective:
            continue
        result.append(_rewire_stage(stage, effective))
    _validate_reachable_from_entry(result, entry_stage=entry_stage)
    return result


def _rewire_stage(stage: "StageConfig", effective: set[str]) -> "StageConfig":
    updates: dict = {}

    # Prune wait_for
    if stage.wait_for is not None:
        kept_wait = [w for w in stage.wait_for if w in effective]
        if kept_wait != stage.wait_for:
            if kept_wait:
                updates["wait_for"] = kept_wait
            else:
                # All upstream removed; the fan-in collapses. Drop merge_fn too
                # since there is nothing to merge.
                updates["wait_for"] = None
                updates["merge_fn"] = None

    # Prune stream_to
    pruned_stream = [t for t in stage.stream_to if t in effective]
    if pruned_stream != stage.stream_to:
        updates["stream_to"] = pruned_stream

    # Prune next; apply fallback if it fully collapses
    if stage.next is not None:
        targets = _as_list(stage.next)
        if not targets:
            raise ValueError(f"Stage {stage.name!r}: next must not be empty")
        kept_next = [t for t in targets if t in effective]

        if kept_next:
            new_next: str | list[str] = (
                kept_next[0]
                if isinstance(stage.next, str) and len(kept_next) == 1
                else kept_next
            )
            updates["next"] = new_next
            project_payload = {
                k: v for k, v in stage.project_payload.items() if k in effective
            }
            pruned_targets = set(targets) - set(kept_next)
            if pruned_targets:
                for target in kept_next:
                    fallback_projection = stage.project_payload_fallback.get(target)
                    if fallback_projection is not None:
                        project_payload[target] = fallback_projection
            updates["project_payload"] = project_payload
        else:
            # All `next` targets disabled — fall back if declared.
            if stage.next_fallback is None:
                raise ValueError(
                    f"Stage {stage.name!r}: all next targets {targets} are "
                    f"disabled by enabled_stages, and no next_fallback is "
                    f"declared on the stage. Either include one of "
                    f"{targets} in enabled_stages, or declare next_fallback "
                    f"(and project_payload_fallback if applicable) on the "
                    f"stage so the framework can rewire the DAG."
                )
            fallback_targets = _as_list(stage.next_fallback)
            if not fallback_targets:
                raise ValueError(
                    f"Stage {stage.name!r}: next_fallback must not be empty"
                )
            missing_fb = [t for t in fallback_targets if t not in effective]
            if missing_fb:
                raise ValueError(
                    f"Stage {stage.name!r}: next_fallback {fallback_targets} "
                    f"references stages not in the effective set "
                    f"(missing: {sorted(missing_fb)}). Add the missing "
                    f"stages to enabled_stages, or revise the fallback "
                    f"declaration to use stages that will remain after "
                    f"filtering."
                )
            updates["next"] = (
                stage.next_fallback
                if isinstance(stage.next_fallback, str)
                else fallback_targets
            )
            # Replace project_payload entirely with the fallback projection.
            updates["project_payload"] = {
                k: v
                for k, v in stage.project_payload_fallback.items()
                if k in fallback_targets
            }

    if not updates:
        return stage
    return stage.model_copy(update=updates)


def _validate_reachable_from_entry(
    stages: "list[StageConfig]",
    *,
    entry_stage: str | None,
) -> None:
    if entry_stage is None or not stages:
        return

    by_name = {stage.name: stage for stage in stages}
    if entry_stage not in by_name:
        raise ValueError(
            f"enabled_stages must retain entry stage {entry_stage!r}. "
            f"Retained stages: {sorted(by_name)}."
        )

    reachable: set[str] = set()
    stack = [entry_stage]
    while stack:
        name = stack.pop()
        if name in reachable:
            continue
        reachable.add(name)
        stage = by_name[name]
        for target in _as_list(stage.next):
            if target in by_name and target not in reachable:
                stack.append(target)

    unreachable = sorted(set(by_name) - reachable)
    if unreachable:
        terminals = sorted(stage.name for stage in stages if stage.terminal)
        raise ValueError(
            f"enabled_stages leaves unreachable retained stages from entry "
            f"{entry_stage!r}: {unreachable}. Retained terminal stages: "
            f"{terminals}."
        )
