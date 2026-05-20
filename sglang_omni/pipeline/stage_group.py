# SPDX-License-Identifier: Apache-2.0
"""StageGroup manages OS processes backing one topology process group."""
from __future__ import annotations

import asyncio
import logging
import multiprocessing
import os
import queue as queue_mod
import time
from contextlib import contextmanager, suppress
from typing import Any, Sequence

from sglang_omni.pipeline.stage_process import (
    StageProcessSpec,
    StageWorkerProcessSpec,
    get_stage_process_env,
    stage_process_main,
)

logger = logging.getLogger(__name__)


def _get_worker_process_env(spec: StageWorkerProcessSpec) -> dict[str, str]:
    """Return the spawn-time env overrides for *spec*.

    Hard invariant: a TP stage (``tp_size > 1``) must own its OS process
    exclusively — its CUDA env remap and NCCL settings depend on being the
    sole tenant. Mixing a TP stage with any other stage in the same
    process group is a placement bug, not a fallback case.
    """
    tp_stages = [s for s in spec.stage_specs if s.tp_size > 1]
    if not tp_stages:
        return {}
    if len(tp_stages) > 1 or len(spec.stage_specs) > 1:
        raise AssertionError(
            f"Process {spec.process_name!r} mixes a TP stage with other "
            "stages; TP stages must own their OS process exclusively. "
            f"stage_specs={[s.stage_name for s in spec.stage_specs]}"
        )
    return get_stage_process_env(tp_stages[0])


@contextmanager
def _patched_spawn_env(spec: StageWorkerProcessSpec):
    updates = _get_worker_process_env(spec)
    if not updates:
        yield
        return

    backup = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            os.environ[key] = value
        yield
    finally:
        for key, value in backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class StageGroup:
    """Lifecycle manager for one or more OS processes in a topology group."""

    def __init__(
        self,
        group_name: str,
        process_specs: Sequence[StageWorkerProcessSpec],
    ):
        if not process_specs:
            raise ValueError(
                f"StageGroup requires at least one process spec (group={group_name})"
            )
        self.group_name = group_name
        self.process_specs = list(process_specs)
        self._processes: list[multiprocessing.Process] = []
        self._ready_events: list[multiprocessing.Event] = []
        self._startup_error_queues: list[Any] = []

    @property
    def process_count(self) -> int:
        return len(self.process_specs)

    @property
    def specs(self) -> list[StageProcessSpec]:
        return [
            stage_spec
            for process_spec in self.process_specs
            for stage_spec in process_spec.stage_specs
        ]

    @property
    def leader_spec(self) -> StageProcessSpec:
        for spec in self.specs:
            if spec.role in {"single", "leader"}:
                return spec
        raise RuntimeError(f"StageGroup {self.group_name} has no leader-owned spec")

    @property
    def leader_endpoint(self) -> str:
        """Control-plane recv endpoint for tp_rank 0 (used by Coordinator)."""
        return self.leader_spec.recv_endpoint

    @property
    def stage_control_endpoints(self) -> dict[str, str]:
        return {
            spec.stage_name: spec.recv_endpoint
            for spec in self.specs
            if spec.owns_external_io
        }

    @property
    def processes(self) -> list[multiprocessing.Process]:
        return list(self._processes)

    def spawn(self, ctx: multiprocessing.context.SpawnContext) -> None:
        """Spawn the OS process(es) owned by this group."""
        for spec in self.process_specs:
            event = ctx.Event()
            startup_error_queue = ctx.Queue()
            proc_name = _process_name(spec)
            proc = ctx.Process(
                target=stage_process_main,
                args=(spec, event, startup_error_queue),
                name=proc_name,
                daemon=True,
            )
            try:
                with _patched_spawn_env(spec):
                    proc.start()
            except Exception:
                _close_queue(startup_error_queue)
                raise
            self._processes.append(proc)
            self._ready_events.append(event)
            self._startup_error_queues.append(startup_error_queue)

        logger.info(
            "StageGroup %s: spawned %d process(es) (pids=%s)",
            self.group_name,
            len(self._processes),
            [p.pid for p in self._processes],
        )

    async def wait_ready(self, timeout: float) -> None:
        """Block until every TP rank signals ready or *timeout* expires."""
        loop = asyncio.get_running_loop()
        deadline = time.monotonic() + timeout

        for i, event in enumerate(self._ready_events):
            proc = self._processes[i]
            spec = self.process_specs[i]
            process_label = spec.process_name

            while not event.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    startup_error = await loop.run_in_executor(
                        None,
                        self._read_startup_error,
                        i,
                        0.0,
                    )
                    details = _format_startup_error(startup_error)
                    raise TimeoutError(
                        f"Process {process_label} did not become ready "
                        f"within {timeout:.0f}s{details}"
                    )
                if not proc.is_alive():
                    startup_error = await loop.run_in_executor(
                        None,
                        self._read_startup_error,
                        i,
                        0.2,
                    )
                    details = _format_startup_error(startup_error)
                    raise RuntimeError(
                        f"Process {process_label} died during startup "
                        f"(exit code {proc.exitcode}){details}"
                    )
                await loop.run_in_executor(None, event.wait, min(remaining, 1.0))

            logger.info("Process %s ready", process_label)

    def any_dead(self) -> bool:
        """Return True if any process in the group exited while runner is active."""
        return any(not p.is_alive() for p in self._processes)

    def dead_summary(self) -> str:
        """Human-readable summary of dead processes (for error messages)."""
        parts = []
        for i, p in enumerate(self._processes):
            if not p.is_alive():
                process_spec = self.process_specs[i]
                parts.append(
                    f"{process_spec.process_name} " f"(pid={p.pid}, exit={p.exitcode})"
                )
        return ", ".join(parts) if parts else "(none)"

    def _read_startup_error(self, index: int, timeout: float) -> Any | None:
        if index >= len(self._startup_error_queues):
            return None
        q = self._startup_error_queues[index]
        try:
            if timeout <= 0:
                return q.get_nowait()
            return q.get(timeout=timeout)
        except queue_mod.Empty:
            return None
        except Exception:
            logger.debug(
                "Failed to read startup error for group %s index %s",
                self.group_name,
                index,
                exc_info=True,
            )
            return None

    def close_control_channels(self) -> None:
        """Close parent-owned multiprocessing queues for this group."""
        queues: list[Any] = []
        queues.extend(self._startup_error_queues)
        for process_spec in self.process_specs:
            for stage_spec in process_spec.stage_specs:
                queues.extend(stage_spec.follower_work_queues)
                queues.extend(stage_spec.follower_abort_queues)
                if stage_spec.internal_work_queue is not None:
                    queues.append(stage_spec.internal_work_queue)
                if stage_spec.internal_abort_queue is not None:
                    queues.append(stage_spec.internal_abort_queue)
        seen: set[int] = set()
        for q in queues:
            marker = id(q)
            if marker in seen:
                continue
            seen.add(marker)
            _close_queue(q)

    async def shutdown(self, join_timeout: float = 30.0) -> None:

        try:
            for p in self._processes:
                p.join(timeout=join_timeout)
                if p.is_alive():
                    logger.warning(
                        "Terminating stuck process %s (pid=%s)",
                        p.name,
                        p.pid,
                    )
                    p.terminate()
                    p.join(timeout=5)
                    if p.is_alive():
                        p.kill()
                        p.join(timeout=2)
        finally:
            self.close_control_channels()
            self._processes.clear()
            self._ready_events.clear()
            self._startup_error_queues.clear()


def _process_name(spec: StageWorkerProcessSpec) -> str:
    if len(spec.stage_specs) > 1:
        return f"process-{spec.process_name}"
    stage_spec = spec.stage_specs[0]
    if stage_spec.role == "single":
        return f"stage-{stage_spec.stage_name}"
    if stage_spec.role == "leader":
        return f"stage-{stage_spec.stage_name}-leader"
    return f"stage-{stage_spec.stage_name}-tp{stage_spec.tp_rank}-follower"


def _format_startup_error(error: Any | None) -> str:
    if error is None:
        return ""
    if isinstance(error, dict):
        process_name = error.get("process_name")
        stage_names = error.get("stage_names")
        traceback_text = error.get("traceback")
        header = (
            f"\nStartup failure detail from process {process_name!r}, "
            f"stages={stage_names}:"
        )
        if traceback_text:
            return f"{header}\n{traceback_text}"
        return header
    return f"\nStartup failure detail: {error}"


def _close_queue(q: Any) -> None:
    close = getattr(q, "close", None)
    if callable(close):
        with suppress(Exception):
            close()
    join_thread = getattr(q, "join_thread", None)
    if callable(join_thread):
        with suppress(Exception):
            join_thread()
