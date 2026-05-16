#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Preflight gate for the MMMU sweep on H200 hosts.

Verifies before GPU time is spent: HF model revisions resolve + snapshots
exist; the two named containers run the expected images and pass /v1/models;
launcher.log references the resolved snapshot path; SGLang accepts an
``image_url`` data-URI request; every dataset repo the sweep will load is
pinned in ``benchmarks/dataset/mmmu_revisions.json``.

Writes a JSON report to ``--output``. Exits non-zero on any failure.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib import error, request


REPO_ROOT = Path(__file__).resolve().parents[2]

EXPECTED_CONTAINERS: dict[str, str] = {
    "sglang-omni-hayden-benchmark": "frankleeeee/sglang-omni:dev",
    "sglang-hayden-benchmark": "lmsysorg/sglang:dev",
}

DEFAULT_MODELS: list[tuple[str, str]] = [
    ("omni", "Qwen/Qwen3-Omni-30B-A3B-Instruct"),
    ("sglang", "Qwen/Qwen3-VL-30B-A3B-Instruct"),
]

DATASET_REPOS_TO_PIN: list[str] = ["MMMU/MMMU", "zhaochenyang20/mmmu-ci-50"]


@dataclass
class PreflightReport:
    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    model_revisions: dict[str, str] = field(default_factory=dict)
    snapshot_paths: dict[str, str] = field(default_factory=dict)
    containers: dict[str, dict[str, Any]] = field(default_factory=dict)
    dataset_revisions: dict[str, str] = field(default_factory=dict)
    sglang_data_uri_probe: dict[str, Any] = field(default_factory=dict)

    def fail(self, message: str) -> None:
        self.ok = False
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)


# ---------------------------------------------------------------- HF lookup


def resolve_hf_revision(repo_id: str) -> str | None:
    """Return the current main-branch commit SHA for an HF model repo.

    Uses the public HF refs API; requires network. Returns None on any
    failure (caller decides whether to treat as fatal).
    """
    url = f"https://huggingface.co/api/models/{repo_id}/refs"
    try:
        with request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (error.URLError, error.HTTPError, json.JSONDecodeError):
        return None
    for branch in data.get("branches", []):
        if branch.get("name") == "main":
            return branch.get("targetCommit")
    return None


def resolve_hf_dataset_revision(repo_id: str) -> str | None:
    url = f"https://huggingface.co/api/datasets/{repo_id}/refs"
    try:
        with request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (error.URLError, error.HTTPError, json.JSONDecodeError):
        return None
    for branch in data.get("branches", []):
        if branch.get("name") == "main":
            return branch.get("targetCommit")
    return None


# ----------------------------------------------------------- container checks


def _docker_inspect(
    container_name: str, fmt: str, host: str | None = None
) -> str | None:
    """Run `docker inspect <name> --format <fmt>` locally or over SSH."""
    prefix = _shell_prefix(host)
    if not prefix and shutil.which("docker") is None:
        return None
    try:
        out = subprocess.check_output(
            prefix + ["docker", "inspect", container_name, "--format", fmt],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except subprocess.CalledProcessError:
        return None


def docker_inspect_image(container_name: str, host: str | None = None) -> str | None:
    return _docker_inspect(container_name, "{{index .Image}}", host)


def docker_inspect_repo_tag(
    container_name: str, host: str | None = None
) -> str | None:
    return _docker_inspect(container_name, "{{.Config.Image}}", host)


def check_container(
    name: str,
    expected_image: str,
    report: PreflightReport,
    host: str | None = None,
) -> None:
    digest = docker_inspect_image(name, host=host)
    image_ref = docker_inspect_repo_tag(name, host=host)
    info: dict[str, Any] = {
        "container_image_digest": digest,
        "container_image": image_ref,
        "name_ok": True,
    }
    if digest is None or image_ref is None:
        info["name_ok"] = False
        report.fail(
            f"Container {name!r} not found or docker unavailable. Start it with the expected "
            f"image ({expected_image}) before re-running the preflight."
        )
    else:
        # Loose match: image_ref may include extra tag suffix (e.g. :dev-sha).
        if expected_image.split(":")[0] not in image_ref:
            report.fail(
                f"Container {name!r} is running image {image_ref!r}, expected {expected_image!r}. "
                f"Stop/remove the container and restart from the contracted image."
            )
    # Merge (not overwrite) so launch_command/snapshot_path written by an
    # earlier launch_named_container() call survives this inspect pass.
    report.containers.setdefault(name, {}).update(info)


# ----------------------------------------------------------- model probes


def probe_v1_models(base_url: str, host: str | None = None) -> dict[str, Any] | None:
    """GET /v1/models from the orchestrator or over SSH from ``host``."""
    url = base_url.rstrip("/") + "/v1/models"
    prefix = _shell_prefix(host)
    if prefix:
        try:
            body = subprocess.check_output(
                prefix + ["curl", "-sS", "--max-time", "15", url],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=30,
            ).strip()
            return json.loads(body) if body else None
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                json.JSONDecodeError, ValueError):
            return None
    try:
        with request.urlopen(url, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (error.URLError, error.HTTPError, json.JSONDecodeError, ValueError):
        return None


def _build_test_image_data_uri() -> str:
    """A 1x1 PNG encoded as a data URI for the SGLang data-URI probe."""
    png_1x1 = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    )
    b64 = base64.b64encode(png_1x1).decode("ascii")
    return f"data:image/png;base64,{b64}"


def probe_sglang_data_uri(
    base_url: str, model: str, host: str | None = None
) -> dict[str, Any]:
    """POST a minimal image_url request to the SGLang server (data-URI probe).

    When ``host`` is set, the curl runs over SSH so the probe verifies the
    same machine that launched the container.
    """
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _build_test_image_data_uri()}},
                    {"type": "text", "text": "ping"},
                ],
            }
        ],
        "max_tokens": 1,
        "temperature": 0.0,
    }
    url = base_url.rstrip("/") + "/v1/chat/completions"
    body = json.dumps(payload)
    prefix = _shell_prefix(host)
    if prefix:
        try:
            status = subprocess.check_output(
                prefix
                + [
                    "curl",
                    "-sS",
                    "-o",
                    "/dev/null",
                    "-w",
                    "%{http_code}",
                    "-X",
                    "POST",
                    "-H",
                    "Content-Type: application/json",
                    "-d",
                    body,
                    url,
                ],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=90,
            ).strip()
            return {"status": int(status) if status.isdigit() else status,
                    "ok": status == "200"}
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            return {"status": None, "ok": False, "error": str(exc)}
    body_bytes = body.encode("utf-8")
    req = request.Request(
        url, data=body_bytes, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with request.urlopen(req, timeout=60) as resp:
            return {"status": resp.status, "ok": resp.status == 200}
    except error.HTTPError as exc:
        return {"status": exc.code, "ok": False, "error": exc.reason}
    except error.URLError as exc:
        return {"status": None, "ok": False, "error": str(exc.reason)}


# ------------------------------------------------------- dataset revisions


def load_dataset_revisions(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"$schema_version": 1, "revisions": {}}
    return json.loads(path.read_text())


def save_dataset_revisions(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def update_dataset_revisions(report: PreflightReport, path: Path, repos: list[str]) -> None:
    data = load_dataset_revisions(path)
    revisions = dict(data.get("revisions") or {})
    for repo in repos:
        sha = resolve_hf_dataset_revision(repo)
        if not sha:
            report.fail(
                f"Could not resolve HF dataset main SHA for {repo!r}. Check network "
                f"or revisit the repo name."
            )
            continue
        revisions[repo] = sha
        report.dataset_revisions[repo] = sha
    data["revisions"] = revisions
    save_dataset_revisions(path, data)


def verify_dataset_revisions(report: PreflightReport, path: Path, repos: list[str]) -> None:
    data = load_dataset_revisions(path)
    revisions = dict(data.get("revisions") or {})
    for repo in repos:
        if repo not in revisions or not revisions[repo]:
            report.fail(
                f"No revision pinned for dataset repo {repo!r} in {path}. "
                f"Run `preflight_mmmu_sweep.py --update-revisions` to populate."
            )
            continue
        report.dataset_revisions[repo] = revisions[repo]


# ---------------------------------------------------------- main entrypoint


def verify_launcher_log_references_snapshot(
    log_path: Path | str, expected_snapshot: str, host: str | None = None
) -> bool:
    """Return True iff the launcher log mentions the expected snapshot path.

    Once a server is launched with ``--model-path <snapshot>`` its launcher
    log must record the path. This grep proves the running server actually
    loaded from the pinned snapshot, not from some other local checkpoint
    or the HF cache via a different revision.

    When ``host`` is set, the log is read over SSH from the host that
    launched the container. Otherwise read from the orchestrator's local
    filesystem.

    Returns False if the file is missing, unreadable, or never mentions the
    expected path string.
    """
    log_path_str = str(log_path)
    prefix = _shell_prefix(host)
    if prefix:
        try:
            text = subprocess.check_output(
                prefix + ["cat", log_path_str],
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=15,
                errors="replace",
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False
    else:
        p = Path(log_path_str)
        if not p.exists():
            return False
        try:
            text = p.read_text(errors="replace")
        except OSError:
            return False
    return expected_snapshot in text


def _shell_prefix(host: str | None) -> list[str]:
    """Empty list for local execution, ``[ssh, host]`` for remote."""
    if host and host not in ("", "localhost", "127.0.0.1"):
        return ["ssh", host]
    return []


def launch_named_container(
    name: str,
    image: str,
    snapshot_path: str,
    server_cmd: list[str],
    log_path: str,
    host: str | None = None,
    record: dict | None = None,
) -> bool:
    """Launch one named container detached and start a `docker logs -f` follower
    that tees stdout/stderr into ``log_path``. Returns True on successful
    ``docker run``; caller runs the readiness probe + log verification.
    """
    prefix = _shell_prefix(host)

    # Idempotent: re-running preflight must not hit "name already in use".
    subprocess.run(
        prefix + ["docker", "rm", "-f", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )

    run_cmd = prefix + [
        "docker",
        "run",
        "-d",
        "--name",
        name,
        "--gpus",
        "all",
        "--network",
        "host",
        "-v",
        f"{snapshot_path}:/snapshot:ro",
        image,
        *server_cmd,
    ]
    # Record the launch tokens so the eval can derive prefix_cache_disabled
    # and mem_fraction_static from real flags, not CLI declarations.
    if record is not None:
        record["launch_command"] = run_cmd
        record["snapshot_path"] = snapshot_path
    try:
        subprocess.check_call(run_cmd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError:
        return False

    # Remote: redirect on the remote side via shell; local: pipe via Popen.
    if prefix:
        remote_shell_cmd = (
            f"docker logs -f {name} > {log_path} 2>&1"
        )
        subprocess.Popen(["ssh", host, remote_shell_cmd])
    else:
        log_file = open(log_path, "wb")
        subprocess.Popen(
            ["docker", "logs", "-f", name],
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    return True


def wait_for_v1_models(
    base_url: str, timeout_s: int = 600, host: str | None = None
) -> bool:
    """Poll /v1/models until 200 or timeout. With ``host`` set, probes over SSH."""
    url = base_url.rstrip("/") + "/v1/models"
    deadline = time.monotonic() + timeout_s
    prefix = _shell_prefix(host)
    while time.monotonic() < deadline:
        if prefix:
            try:
                out = subprocess.check_output(
                    prefix
                    + [
                        "curl",
                        "-sS",
                        "-o",
                        "/dev/null",
                        "-w",
                        "%{http_code}",
                        url,
                    ],
                    stderr=subprocess.DEVNULL,
                    text=True,
                    timeout=10,
                ).strip()
                if out == "200":
                    return True
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                pass
        else:
            try:
                with request.urlopen(url, timeout=5) as resp:
                    if resp.status == 200:
                        return True
            except (error.URLError, error.HTTPError):
                pass
        time.sleep(5)
    return False


def print_launch_commands(
    snapshot_omni: str | None,
    snapshot_sglang: str | None,
    port_omni: int,
    port_sglang: int,
) -> None:
    """Manual-audit fallback: print the docker run commands the operator
    would execute by hand. The sweep itself now invokes `--launch` mode
    over SSH (see `run_mmmu_sweep.sh`), so this flag is kept only for
    operators who want to inspect the commands before allowing
    `--launch` to issue them. For the operational path, prefer
    `--launch` + `--host <H200>` over copy-paste.
    """
    print("# --- Container launch commands (run on the target H200 host) ---")
    if snapshot_omni:
        print(
            f"docker run -d --name sglang-omni-hayden-benchmark "
            f"--gpus all --network host "
            f"-v {snapshot_omni}:/snapshot:ro "
            f"frankleeeee/sglang-omni:dev "
            f"sgl-omni serve --model-path /snapshot --text-only --port {port_omni} "
            f"2>&1 | tee /tmp/sglang-omni-benchmark.log"
        )
    if snapshot_sglang:
        print(
            f"docker run -d --name sglang-hayden-benchmark "
            f"--gpus all --network host "
            f"-v {snapshot_sglang}:/snapshot:ro "
            f"lmsysorg/sglang:dev "
            f"python -m sglang.launch_server --model-path /snapshot --port {port_sglang} "
            f"2>&1 | tee /tmp/sglang-benchmark.log"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preflight gate for the MMMU sweep.")
    p.add_argument(
        "--base-url-omni",
        default="http://localhost:30000",
        help="Local URL of the sglang-omni-hayden-benchmark container.",
    )
    p.add_argument(
        "--base-url-sglang",
        default="http://localhost:30001",
        help="Local URL of the sglang-hayden-benchmark container.",
    )
    p.add_argument(
        "--snapshot-root",
        default="/root/.cache/huggingface/hub",
        help="Where local HF snapshots live (default: HF default cache).",
    )
    p.add_argument(
        "--download",
        action="store_true",
        help="Run `huggingface-cli download --revision <sha>` to materialize the "
        "snapshots locally. Off by default — the gate verifies the snapshot "
        "exists but does not pull weights unless asked.",
    )
    p.add_argument(
        "--update-revisions",
        action="store_true",
        help="Query HF for current dataset main SHAs and write them into "
        "benchmarks/dataset/mmmu_revisions.json before verification.",
    )
    p.add_argument(
        "--skip-container-check",
        action="store_true",
        help="Skip docker inspect / /v1/models probes (for dev-machine dry-runs).",
    )
    p.add_argument(
        "--launcher-log-omni",
        default=None,
        help="Path to the sglang-omni server launcher log. When provided, the "
        "preflight greps it for the resolved snapshot path to prove the "
        "running server actually loaded from the pinned snapshot.",
    )
    p.add_argument(
        "--launcher-log-sglang",
        default=None,
        help="Path to the sglang server launcher log (same purpose).",
    )
    p.add_argument(
        "--print-launch-commands",
        action="store_true",
        help="Emit the docker run commands the operator should execute on "
        "the H200 host to start the two named containers with snapshot "
        "pinning, then exit. Kept as a manual-audit fallback for the "
        "operator who wants to inspect commands before --launch runs them.",
    )
    p.add_argument(
        "--launch",
        action="store_true",
        help="Actually start the two contracted containers (idempotent: "
        "any existing container with the same name is removed first). "
        "Each container's stdout/stderr is captured via "
        "`docker logs -f` redirected to the corresponding --launcher-log-* "
        "path, so the snapshot-path log grep can run afterwards. "
        "Combine with --host <hostname> to launch on a remote H200 over SSH; "
        "the sweep orchestrator (run_mmmu_sweep.sh) uses this mode "
        "automatically with one preflight invocation per host.",
    )
    p.add_argument(
        "--strict-log-check",
        action="store_true",
        help="Treat missing or unverified launcher logs as a hard failure "
        "instead of a warning. Required mode for the sweep runner so a "
        "silent log gap cannot let an unverified server serve benchmark "
        "traffic.",
    )
    p.add_argument(
        "--host",
        default=None,
        help="Host to launch both containers on (passed to ssh). When unset "
        "or 'localhost', launches locally. Mutually exclusive use: a "
        "single preflight invocation can launch one host's pair; for the "
        "parallel-by-lane sweep run preflight twice (once per host).",
    )
    p.add_argument(
        "--ready-timeout-s",
        type=int,
        default=600,
        help="Max seconds to wait for each server's /v1/models to respond "
        "after --launch (default: 600).",
    )
    p.add_argument(
        "--mem-fraction-static",
        type=float,
        default=0.9,
        help="GPU memory fraction reserved for static weights on each "
        "container. Recorded in preflight JSON and passed as a server "
        "flag to both containers (default: 0.9).",
    )
    p.add_argument(
        "--disable-prefix-cache",
        action="store_true",
        default=True,
        help="Pass --disable-radix-cache to both server launch commands "
        "so the metadata's prefix_cache_disabled is enforced by launch flag, not "
        "just declared by metadata (default: True).",
    )
    p.add_argument(
        "--port-omni",
        type=int,
        default=30000,
        help="Port the sglang-omni container should listen on (default: 30000).",
    )
    p.add_argument(
        "--port-sglang",
        type=int,
        default=30001,
        help="Port the sglang container should listen on (default: 30001).",
    )
    p.add_argument(
        "--output",
        default=str(REPO_ROOT / "results" / "preflight_mmmu_sweep.json"),
        help="Where to write the JSON report.",
    )
    return p.parse_args()


def _check_launcher_logs(
    args: argparse.Namespace, report: PreflightReport
) -> None:
    """Verify each launcher log references the resolved snapshot path.

    With ``--strict-log-check`` set, missing or non-matching logs are fatal;
    otherwise they degrade to a warning.
    """
    pairs = [
        ("omni", args.launcher_log_omni, "Qwen/Qwen3-Omni-30B-A3B-Instruct"),
        ("sglang", args.launcher_log_sglang, "Qwen/Qwen3-VL-30B-A3B-Instruct"),
    ]
    strict = bool(getattr(args, "strict_log_check", False))
    for backend_tag, log_path, repo_id in pairs:
        if not log_path:
            msg = (
                f"--launcher-log-{backend_tag} not provided; cannot verify the "
                f"{backend_tag} server actually loaded the pinned snapshot."
            )
            (report.fail if strict else report.warn)(msg)
            continue
        snapshot = report.snapshot_paths.get(repo_id)
        if not snapshot:
            report.fail(
                f"No resolved snapshot path for {repo_id}, so the {backend_tag} "
                f"launcher-log verification cannot run."
            )
            continue
        if not verify_launcher_log_references_snapshot(
            log_path, snapshot, host=getattr(args, "host", None)
        ):
            report.fail(
                f"{backend_tag} launcher log at {log_path} does not reference the "
                f"resolved snapshot path {snapshot}. The container is not loading "
                f"from the pinned model — abort before any benchmark traffic runs."
            )


def main() -> int:
    args = parse_args()
    report = PreflightReport()

    # Optional: emit docker run commands and exit. Used by the operator to
    # sanity-check the launch contract before issuing the commands by hand
    # on the H200 host.
    if args.print_launch_commands:
        # We still need resolved snapshot paths for the commands; do a fast
        # HF lookup pass without container probes.
        snapshots: dict[str, str | None] = {}
        for _, repo_id in DEFAULT_MODELS:
            sha = resolve_hf_revision(repo_id)
            if sha:
                snapshot_dirname = f"models--{repo_id.replace('/', '--')}"
                snapshots[repo_id] = str(
                    Path(args.snapshot_root) / snapshot_dirname / "snapshots" / sha
                )
        print_launch_commands(
            snapshots.get("Qwen/Qwen3-Omni-30B-A3B-Instruct"),
            snapshots.get("Qwen/Qwen3-VL-30B-A3B-Instruct"),
            args.port_omni,
            args.port_sglang,
        )
        return 0

    # 1. Model revision resolution.
    for backend_tag, repo_id in DEFAULT_MODELS:
        sha = resolve_hf_revision(repo_id)
        if not sha:
            report.fail(
                f"Could not resolve HF model main SHA for {repo_id!r}. Check network "
                f"and rerun. The MMMU sweep must run against a pinned revision."
            )
            continue
        report.model_revisions[repo_id] = sha
        snapshot_dirname = f"models--{repo_id.replace('/', '--')}"
        snapshot_path = Path(args.snapshot_root) / snapshot_dirname / "snapshots" / sha
        report.snapshot_paths[repo_id] = str(snapshot_path)
        if not snapshot_path.exists():
            if args.download and shutil.which("huggingface-cli") is not None:
                local_dir = snapshot_path.parent.parent / "local-snapshot" / sha
                local_dir.mkdir(parents=True, exist_ok=True)
                try:
                    subprocess.check_call(
                        [
                            "huggingface-cli",
                            "download",
                            repo_id,
                            "--revision",
                            sha,
                            "--local-dir",
                            str(local_dir),
                        ]
                    )
                    report.snapshot_paths[repo_id] = str(local_dir)
                except subprocess.CalledProcessError as exc:
                    report.fail(
                        f"huggingface-cli download failed for {repo_id} @ {sha}: {exc}"
                    )
            else:
                # The gate fails closed when the snapshot does not exist.
                # The sweep cannot run against unpinned weights and the
                # preflight is the gate that prevents that.
                report.fail(
                    f"Snapshot for {repo_id} @ {sha} not found at {snapshot_path}. "
                    f"Re-run with --download to materialize it, or pre-populate "
                    f"the HF cache before the sweep launches."
                )

    # 1b. Optional: actually launch the contracted containers and capture
    # their launcher logs — replaces the earlier
    # print-commands half-step with real execution. Failure here is fatal
    # because the rest of the gate has no servers to probe.
    if args.launch and report.ok:
        snapshot_omni = report.snapshot_paths.get(
            "Qwen/Qwen3-Omni-30B-A3B-Instruct"
        )
        snapshot_sglang = report.snapshot_paths.get(
            "Qwen/Qwen3-VL-30B-A3B-Instruct"
        )
        if not snapshot_omni or not snapshot_sglang:
            report.fail(
                "--launch requested but at least one snapshot path is missing; "
                "cannot start containers without pinned mounts."
            )
        else:
            log_omni = args.launcher_log_omni or "/tmp/sglang-omni-benchmark.log"
            log_sglang = args.launcher_log_sglang or "/tmp/sglang-benchmark.log"
            mem_frac = f"{args.mem_fraction_static:.3f}"

            omni_server_cmd = [
                "sgl-omni",
                "serve",
                "--model-path",
                "/snapshot",
                "--text-only",
                "--port",
                str(args.port_omni),
                "--mem-fraction-static",
                mem_frac,
            ]
            if args.disable_prefix_cache:
                omni_server_cmd.append("--disable-radix-cache")
            omni_record = report.containers.setdefault("sglang-omni-hayden-benchmark", {})
            ok_omni = launch_named_container(
                name="sglang-omni-hayden-benchmark",
                image="frankleeeee/sglang-omni:dev",
                snapshot_path=snapshot_omni,
                server_cmd=omni_server_cmd,
                log_path=log_omni,
                host=args.host,
                record=omni_record,
            )

            sglang_server_cmd = [
                "python",
                "-m",
                "sglang.launch_server",
                "--model-path",
                "/snapshot",
                "--port",
                str(args.port_sglang),
                "--mem-fraction-static",
                mem_frac,
            ]
            if args.disable_prefix_cache:
                sglang_server_cmd.append("--disable-radix-cache")
            sglang_record = report.containers.setdefault("sglang-hayden-benchmark", {})
            ok_sglang = launch_named_container(
                name="sglang-hayden-benchmark",
                image="lmsysorg/sglang:dev",
                snapshot_path=snapshot_sglang,
                server_cmd=sglang_server_cmd,
                log_path=log_sglang,
                host=args.host,
                record=sglang_record,
            )
            if not ok_omni:
                report.fail("Failed to launch sglang-omni-hayden-benchmark.")
            if not ok_sglang:
                report.fail("Failed to launch sglang-hayden-benchmark.")
            # Wait for both servers to report healthy via /v1/models.
            if ok_omni and not wait_for_v1_models(
                args.base_url_omni, args.ready_timeout_s, host=args.host
            ):
                report.fail(
                    f"sglang-omni-hayden-benchmark did not report ready on "
                    f"{args.base_url_omni}/v1/models within {args.ready_timeout_s}s."
                )
            if ok_sglang and not wait_for_v1_models(
                args.base_url_sglang, args.ready_timeout_s, host=args.host
            ):
                report.fail(
                    f"sglang-hayden-benchmark did not report ready on "
                    f"{args.base_url_sglang}/v1/models within {args.ready_timeout_s}s."
                )
            # The log files now exist (the follower started writing to them
            # when the container began emitting output). Pin them so the
            # downstream log-check step uses these paths.
            args.launcher_log_omni = log_omni
            args.launcher_log_sglang = log_sglang

    # 2. Container checks (skippable for dev-box dry-runs). Run on the
    # same host that launched the containers when --host is set so the
    # digest captured in the report matches the running container.
    if not args.skip_container_check:
        for name, expected_image in EXPECTED_CONTAINERS.items():
            check_container(name, expected_image, report, host=args.host)

    # 3. /v1/models probes.
    if not args.skip_container_check:
        for backend_tag, base_url in (
            ("omni", args.base_url_omni),
            ("sglang", args.base_url_sglang),
        ):
            info = probe_v1_models(base_url, host=args.host)
            if info is None:
                report.fail(
                    f"/v1/models on {base_url} did not respond. Confirm the {backend_tag} "
                    f"server is up and listening."
                )
            else:
                # Record the first model identity returned.
                models = info.get("data") or []
                first = models[0].get("id") if models else None
                report.containers.setdefault(
                    "sglang-omni-hayden-benchmark"
                    if backend_tag == "omni"
                    else "sglang-hayden-benchmark",
                    {},
                )["loaded_model"] = first

    # 4. SGLang data-URI compatibility probe.
    if not args.skip_container_check:
        probe = probe_sglang_data_uri(
            args.base_url_sglang, "qwen3-vl", host=args.host
        )
        report.sglang_data_uri_probe = probe
        if not probe.get("ok"):
            report.fail(
                f"SGLang data-URI image_url probe failed (status={probe.get('status')}). "
                f"This means stock SGLang at the running revision does not accept "
                f"data: URIs in messages[].content image_url parts. The omni-side "
                f"payload translator emits data: URIs; either swap to a tiny local "
                f"file-server fallback or upgrade SGLang."
            )

    # 4b. Launcher log verification — proves the server actually loaded
    # from the pinned snapshot, not from a stale local checkpoint.
    if not args.skip_container_check:
        _check_launcher_logs(args, report)

    # 5. Dataset revision pinning.
    rev_path = REPO_ROOT / "benchmarks" / "dataset" / "mmmu_revisions.json"
    if args.update_revisions:
        update_dataset_revisions(report, rev_path, DATASET_REPOS_TO_PIN)
    else:
        verify_dataset_revisions(report, rev_path, DATASET_REPOS_TO_PIN)

    # Write JSON report.
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "ok": report.ok,
                "errors": report.errors,
                "warnings": report.warnings,
                "model_revisions": report.model_revisions,
                "snapshot_paths": report.snapshot_paths,
                "containers": report.containers,
                "dataset_revisions": report.dataset_revisions,
                "sglang_data_uri_probe": report.sglang_data_uri_probe,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )

    if not report.ok:
        print(f"[preflight] FAILED — see {out_path}", file=sys.stderr)
        for err in report.errors:
            print(f"  - {err}", file=sys.stderr)
        return 2

    print(f"[preflight] OK — report at {out_path}")
    if report.warnings:
        for warn in report.warnings:
            print(f"[preflight] warn: {warn}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
