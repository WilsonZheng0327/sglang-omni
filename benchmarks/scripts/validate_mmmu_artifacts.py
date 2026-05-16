#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate a finished MMMU sweep's retained artifact bundle.

Status-JSONL-driven contract: the source of truth for what
cells the sweep attempted is ``sweep-status.jsonl``. Every status row
must point at a ``cell_dir`` that contains a complete bundle
(``mmmu_results.json`` + ``preflight.json`` + ``launcher.log`` +
``stderr.log``). Every ``mmmu_results.json`` discovered on disk must
have a matching status row. Container digests in the status row must
be non-empty and equal the cell's run_metadata digest. All
``REQUIRED_FIELDS`` from the run-metadata schema must be present under
``run_metadata`` and the live fields must be non-null. Any failed
condition fails the sweep
report path with a non-zero exit.

Failed reps (``status: failed`` in the status JSONL) keep their cell
directory and are still validated for bundle completeness — the goal
is "no silent data loss", not "must pass". But missing bundles or
mismatched digests for an otherwise-successful row are hard errors
because they would let downstream reporting use stale or wrong data.

Exit codes:
  0 = all cells valid
  1 = at least one cell missing artifacts or failed validation

Usage:
  validate_mmmu_artifacts.py <out_root> <status_log>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running this script standalone (`python validate_mmmu_artifacts.py ...`)
# from any cwd by inserting the repo root on sys.path before the package import.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.scripts.run_metadata import REQUIRED_FIELDS  # noqa: E402

LIVE_REQUIRED = (
    "model_revision",
    "container_image_digest",
    "dataset_revisions",
    "mem_fraction_static_configured",
    "kv_cache_capacity_tokens",
    "steady_state_gpu_gb",
)

CELL_BUNDLE_FILES = (
    "mmmu_results.json",
    "preflight.json",
    "launcher.log",
    "stderr.log",
)


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _validate_launch_command(
    preflight: dict, container_name: str, row_label: str
) -> list[str]:
    """Require launch_command tokens to include ``--disable-radix-cache`` and a numeric ``--mem-fraction-static <X>``."""
    issues: list[str] = []
    containers = (preflight.get("containers") or {}) if preflight else {}
    container_record = containers.get(container_name) or {}
    launch_cmd = container_record.get("launch_command")
    if not launch_cmd:
        issues.append(
            f"{row_label}: retained preflight.json container {container_name!r} "
            f"is missing `launch_command`; cannot prove launch policy from evidence"
        )
        return issues
    tokens = list(launch_cmd)
    if "--disable-radix-cache" not in tokens:
        issues.append(
            f"{row_label}: launch_command for {container_name!r} is missing "
            f"--disable-radix-cache; prefix_cache_disabled cannot be proven"
        )
    has_mem_fraction = False
    for i, tok in enumerate(tokens):
        if tok == "--mem-fraction-static" and i + 1 < len(tokens):
            try:
                float(tokens[i + 1])
                has_mem_fraction = True
            except ValueError:
                pass
            break
    if not has_mem_fraction:
        issues.append(
            f"{row_label}: launch_command for {container_name!r} is missing a "
            f"numeric --mem-fraction-static <X>; mem-fraction cannot be proven"
        )
    return issues


def _validate_status_row(row: dict) -> list[str]:
    """Validate one sweep-status.jsonl row + the bundle it points at."""
    issues: list[str] = []
    cell_dir_str = row.get("cell_dir")
    if not cell_dir_str:
        issues.append(f"status row has no cell_dir: {row!r}")
        return issues
    cell_dir = Path(cell_dir_str)
    row_label = (
        f"{row.get('host', '?')}/{row.get('backend', '?')}/"
        f"lane={row.get('lane', '?')}/rep={row.get('rep', '?')}"
    )
    if not cell_dir.exists():
        issues.append(f"{row_label}: cell_dir {cell_dir} does not exist")
        return issues

    for fname in CELL_BUNDLE_FILES:
        if not (cell_dir / fname).exists():
            issues.append(f"{row_label}: missing {fname} in {cell_dir}")

    # Every row must declare failure_log_path; failed-status rows must have it non-empty.
    if "failure_log_path" not in row:
        issues.append(
            f"{row_label}: status row is missing 'failure_log_path' key; the "
            f"plan requires every status row to surface its failure log path"
        )
    elif row.get("status") == "failed" and not (row.get("failure_log_path") or "").strip():
        issues.append(
            f"{row_label}: failed row has empty failure_log_path; the plan "
            f"requires failed reps to surface their failure log path"
        )

    result_path = cell_dir / "mmmu_results.json"
    if not result_path.exists():
        return issues  # already reported above

    try:
        data = json.loads(result_path.read_text())
    except json.JSONDecodeError as exc:
        issues.append(f"{row_label}: invalid JSON in mmmu_results.json ({exc})")
        return issues

    meta = data.get("run_metadata")
    if not isinstance(meta, dict):
        issues.append(f"{row_label}: missing run_metadata block")
        return issues

    for key in REQUIRED_FIELDS:
        if key not in meta:
            issues.append(f"{row_label}: run_metadata missing key {key!r}")

    if row.get("status") == "success":
        # Live fields enforced only on success rows; failed rows keep partial state.
        for key in LIVE_REQUIRED:
            value = meta.get(key)
            if value is None or value == [] or value == {}:
                issues.append(
                    f"{row_label}: live metadata field {key!r} is empty/None ({value!r})"
                )

    # Digest cross-check: row digest non-empty and equal to meta digest.
    row_digest = (row.get("container_image_digest") or "").strip()
    meta_digest = (meta.get("container_image_digest") or "").strip()
    if row.get("status") == "success":
        if not row_digest:
            issues.append(f"{row_label}: status row container_image_digest is empty")
        if not meta_digest:
            issues.append(f"{row_label}: run_metadata container_image_digest is empty")
        if row_digest and meta_digest and row_digest != meta_digest:
            issues.append(
                f"{row_label}: status row digest {row_digest!r} != "
                f"run_metadata digest {meta_digest!r}"
            )

    # Failed-rep consistency: the plan requires `failure_count` to agree
    # between the status row, the run_metadata block, and the count of
    # per-sample records flagged as not-successful. Disagreement implies
    # either silently-retried reps that overwrote failure evidence, or a
    # reporter that miscounted; both are unsafe for downstream reporting.
    if row.get("status") == "success":
        row_failures = row.get("failure_count")
        meta_failures = meta.get("failure_count")
        if row_failures is not None and meta_failures is not None and row_failures != meta_failures:
            issues.append(
                f"{row_label}: status row failure_count={row_failures} != "
                f"run_metadata failure_count={meta_failures}; silently-retried "
                f"failures would produce this kind of disagreement"
            )
        per_sample = data.get("per_sample")
        if isinstance(per_sample, list) and per_sample:
            sample_failures = sum(
                1 for rec in per_sample if not rec.get("is_success", False)
            )
            if meta_failures is not None and meta_failures != sample_failures:
                issues.append(
                    f"{row_label}: run_metadata failure_count={meta_failures} != "
                    f"per_sample non-success count={sample_failures}; the "
                    f"failed-rep accounting contract is broken"
                )

    if row.get("status") == "success":
        preflight_path = cell_dir / "preflight.json"
        if preflight_path.exists():
            try:
                preflight = json.loads(preflight_path.read_text())
            except json.JSONDecodeError as exc:
                issues.append(
                    f"{row_label}: preflight.json is not valid JSON ({exc})"
                )
            else:
                container_name = meta.get("container_name") or row.get("container_name")
                if container_name:
                    issues.extend(
                        _validate_launch_command(preflight, container_name, row_label)
                    )
                else:
                    issues.append(
                        f"{row_label}: cannot determine container_name to check "
                        f"launch_command evidence"
                    )

    return issues


def _find_duplicate_rep_tuples(status_rows: list[dict]) -> list[str]:
    """Reject duplicate ``(host, backend, lane, rep)`` rows (silent-retry hazard)."""
    seen: dict[tuple, int] = {}
    issues: list[str] = []
    for idx, row in enumerate(status_rows):
        key = (
            row.get("host"),
            row.get("backend"),
            row.get("lane"),
            row.get("rep"),
        )
        if any(k is None for k in key):
            continue  # malformed rows are already flagged by row validation
        if key in seen:
            issues.append(
                f"duplicate status row dispatch tuple "
                f"host={key[0]!r}/backend={key[1]!r}/lane={key[2]!r}/rep={key[3]!r}: "
                f"first at row index {seen[key]}, again at row index {idx}; "
                f"silently-retried reps must NOT overwrite prior status rows"
            )
        else:
            seen[key] = idx
    return issues


def _find_orphan_cells(out_root: Path, status_rows: list[dict]) -> list[str]:
    """Discovered mmmu_results.json files that no status row references."""
    issues: list[str] = []
    known = {
        Path(r["cell_dir"]).resolve()
        for r in status_rows
        if r.get("cell_dir")
    }
    for result_path in sorted(out_root.rglob("mmmu_results.json")):
        cell_dir = result_path.parent.resolve()
        if cell_dir not in known:
            issues.append(
                f"orphan cell artifact: {result_path} has no matching status row"
            )
    return issues


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2

    out_root = Path(sys.argv[1])
    status_log = Path(sys.argv[2])

    status_rows = _load_jsonl(status_log)
    if not status_rows:
        print(
            f"[validate] FAILED — status log {status_log} is empty or missing",
            file=sys.stderr,
        )
        return 1

    all_issues: list[str] = []
    for row in status_rows:
        all_issues.extend(_validate_status_row(row))
    all_issues.extend(_find_duplicate_rep_tuples(status_rows))
    all_issues.extend(_find_orphan_cells(out_root, status_rows))

    if all_issues:
        print("[validate] FAILED", file=sys.stderr)
        for issue in all_issues:
            print(f"  - {issue}", file=sys.stderr)
        return 1

    print(
        f"[validate] OK — {len(status_rows)} cells, all bundles complete and "
        f"digests cross-checked."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
