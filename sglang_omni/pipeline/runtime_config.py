# SPDX-License-Identifier: Apache-2.0
"""Runtime preparation helpers shared by pipeline runners."""

from __future__ import annotations

import logging
import re
import shutil
import socket
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from sglang_omni.config.placement import StagePlacementPlan, build_stage_placement_plan
from sglang_omni.config.schema import PipelineConfig, StageConfig
from sglang_omni.config.topology import ProcessTopologyPlan, build_process_topology_plan

logger = logging.getLogger(__name__)


class IpcRuntimeDir:
    """Runtime-owned IPC directory for one pipeline instance."""

    def __init__(self, path: Path):
        self.path = path
        self._closed = False

    def __enter__(self) -> IpcRuntimeDir:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"IpcRuntimeDir(path={self.path!r}, closed={self._closed})"

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            shutil.rmtree(self.path)
        except FileNotFoundError:
            return
        except OSError as exc:
            logger.warning("Failed to remove IPC runtime dir %s: %s", self.path, exc)


class TcpEndpointReservation:
    """Parent-owned TCP port reservations for one pipeline startup.

    These sockets keep auto-selected TCP ports owned while runtime prep builds
    the launch plan. The runner releases each endpoint when handing ownership
    to the component that binds it.
    """

    def __init__(self, sockets: dict[str, socket.socket]):
        self._sockets = dict(sockets)

    def __repr__(self) -> str:
        return f"TcpEndpointReservation(keys={sorted(self._sockets)})"

    @property
    def keys(self) -> set[str]:
        return set(self._sockets)

    def release(
        self,
        keys: list[str] | tuple[str, ...] | set[str] | None = None,
    ) -> None:
        selected = list(self._sockets) if keys is None else list(keys)
        for key in selected:
            sock = self._sockets.pop(key, None)
            if sock is None:
                continue
            try:
                sock.close()
            except OSError as exc:
                logger.warning("Failed to release TCP endpoint %s: %s", key, exc)

    def close(self) -> None:
        self.release()


@dataclass(frozen=True)
class EndpointAllocation:
    endpoints: dict[str, str]
    reservation: TcpEndpointReservation | None = None


@dataclass(frozen=True)
class PipelineRuntimePrep:
    """Prepared stage, endpoint, placement, and topology state."""

    stages_cfg: list[StageConfig]
    name_map: dict[str, str]
    entry_stage: str
    endpoints: dict[str, str]
    placement_plan: StagePlacementPlan
    process_plan: ProcessTopologyPlan
    runtime_dir: IpcRuntimeDir | None
    runtime_dir_created_here: bool
    endpoint_reservation: TcpEndpointReservation | None = None


def create_ipc_runtime_dir(config: PipelineConfig) -> IpcRuntimeDir | None:
    """Create a per-run IPC namespace for one pipeline instance."""
    if config.endpoints.scheme != "ipc":
        return None

    base_root = Path(config.endpoints.base_path)
    base_root.mkdir(parents=True, exist_ok=True)

    namespace_prefix = re.sub(r"[^0-9a-z]+", "-", config.name.lower()).strip("-")
    if not namespace_prefix:
        namespace_prefix = "pipeline"
    path = Path(tempfile.mkdtemp(prefix=f"{namespace_prefix}-", dir=base_root))
    return IpcRuntimeDir(path)


def prepare_pipeline_runtime(
    config: PipelineConfig,
    *,
    ipc_runtime_dir: IpcRuntimeDir | None = None,
) -> PipelineRuntimePrep:
    """Prepare fused stages, endpoint allocation, and process topology."""
    runtime_dir = ipc_runtime_dir
    created_runtime_dir = None
    if runtime_dir is None:
        runtime_dir = create_ipc_runtime_dir(config)
        created_runtime_dir = runtime_dir
    runtime_dir_created_here = created_runtime_dir is not None

    try:
        stages_cfg, name_map, entry_stage = config.apply_fusion()
        placement_plan = build_stage_placement_plan(config, stages_cfg=stages_cfg)
        process_plan = build_process_topology_plan(
            config,
            placement_plan,
            stages_cfg=stages_cfg,
        )
        endpoint_allocation = allocate_pipeline_endpoints(
            config,
            stages=stages_cfg,
            ipc_base_dir=runtime_dir.path if runtime_dir else None,
        )
        endpoints = endpoint_allocation.endpoints
    except Exception:
        if created_runtime_dir is not None:
            created_runtime_dir.close()
        raise

    return PipelineRuntimePrep(
        stages_cfg=stages_cfg,
        name_map=name_map,
        entry_stage=entry_stage,
        endpoints=endpoints,
        placement_plan=placement_plan,
        process_plan=process_plan,
        runtime_dir=runtime_dir,
        runtime_dir_created_here=runtime_dir_created_here,
        endpoint_reservation=endpoint_allocation.reservation,
    )


def build_relay_config(
    stage_cfg: StageConfig,
    global_cfg: PipelineConfig,
) -> dict:
    relay_cfg = stage_cfg.relay
    if relay_cfg is not None:
        return {
            "relay_type": global_cfg.relay_backend,
            "slot_size_mb": relay_cfg.slot_size_mb,
            "credits": relay_cfg.credits,
            "rank": relay_cfg.rank,
            "world_size": relay_cfg.world_size,
            "gpu_id": parse_gpu_id(relay_cfg.device),
        }

    if global_cfg.relay_backend == "shm":
        gpu_id = None
    else:
        gpu = stage_cfg.gpu
        if gpu is None:
            gpu_id = None
        elif isinstance(gpu, list):
            gpu_id = gpu[0]
        else:
            gpu_id = gpu

    return {
        "relay_type": global_cfg.relay_backend,
        "slot_size_mb": 512,
        "credits": 2,
        "rank": None,
        "world_size": None,
        "gpu_id": gpu_id,
    }


def parse_gpu_id(device: str) -> int | None:
    if device == "cpu":
        return None
    if device == "cuda":
        return 0
    if device.startswith("cuda:"):
        return int(device.split(":", 1)[1])
    raise ValueError(f"Unsupported device string: {device}")


def reserve_free_tcp_ports(
    start: int,
    count: int,
    *,
    exclude_ports: set[int] | None = None,
) -> list[socket.socket]:
    """Reserve *count* available TCP ports starting from *start*.

    The returned sockets remain bound until closed by the caller.
    """

    sockets: list[socket.socket] = []
    excluded = exclude_ports or set()
    port = start
    try:
        while len(sockets) < count:
            if port in excluded:
                port += 1
                continue
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                sock.close()
            else:
                sockets.append(sock)
            port += 1
    except Exception:
        for sock in sockets:
            sock.close()
        raise
    return sockets


def _tcp_endpoint_port(endpoint: str) -> int | None:
    parsed = urlsplit(endpoint)
    if parsed.scheme != "tcp":
        return None
    try:
        return parsed.port
    except ValueError:
        return None


def _explicit_tcp_ports(endpoints: dict[str, str]) -> set[int]:
    return {
        port
        for endpoint in endpoints.values()
        if (port := _tcp_endpoint_port(endpoint)) is not None
    }


def allocate_pipeline_endpoints(
    config: PipelineConfig,
    *,
    stages: list[StageConfig],
    ipc_base_dir: Path | None = None,
) -> EndpointAllocation:
    endpoints: dict[str, str] = {}

    if config.completion_endpoint:
        endpoints["completion"] = config.completion_endpoint
    if config.abort_endpoint:
        endpoints["abort"] = config.abort_endpoint

    if config.endpoints.scheme == "ipc":
        if ipc_base_dir is None:
            raise ValueError("IPC endpoint allocation requires an IPC runtime dir")
        base_dir = ipc_base_dir
        endpoints.setdefault("completion", f"ipc://{base_dir}/completion.sock")
        endpoints.setdefault("abort", f"ipc://{base_dir}/abort.sock")
        for stage in stages:
            endpoints[f"stage_{stage.name}"] = (
                f"ipc://{base_dir}/stage_{stage.name}.sock"
            )
        return EndpointAllocation(endpoints=endpoints)

    if config.endpoints.scheme == "tcp":
        endpoint_keys: list[str] = []
        if "completion" not in endpoints:
            endpoint_keys.append("completion")
        if "abort" not in endpoints:
            endpoint_keys.append("abort")
        for stage in stages:
            endpoint_keys.append(f"stage_{stage.name}")

        reserved_sockets = reserve_free_tcp_ports(
            config.endpoints.base_port,
            len(endpoint_keys),
            exclude_ports=_explicit_tcp_ports(endpoints),
        )
        reservation_sockets: dict[str, socket.socket] = {}
        for key, sock in zip(endpoint_keys, reserved_sockets, strict=True):
            port = sock.getsockname()[1]
            endpoints[key] = f"tcp://127.0.0.1:{port}"
            reservation_sockets[key] = sock
        reservation = (
            TcpEndpointReservation(reservation_sockets) if reservation_sockets else None
        )
        return EndpointAllocation(endpoints=endpoints, reservation=reservation)

    raise ValueError(f"Unknown endpoint scheme: {config.endpoints.scheme}")


def allocate_endpoints(
    config: PipelineConfig,
    *,
    stages: list[StageConfig],
    ipc_base_dir: Path | None = None,
) -> dict[str, str]:
    allocation = allocate_pipeline_endpoints(
        config,
        stages=stages,
        ipc_base_dir=ipc_base_dir,
    )
    if allocation.reservation is not None:
        allocation.reservation.close()
    return allocation.endpoints
