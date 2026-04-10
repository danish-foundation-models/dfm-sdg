from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

from sdg.commons.run_log import activate_run_log, log_event
from sdg.commons.store import ensure_dir
from sdg.commons.utils import (
    artifacts_root,
    git_info,
    hash_spec,
    iso_timestamp,
    read_json,
    write_json,
    write_yaml,
)

RunStatus = Literal["running", "completed", "failed"]


@dataclass(frozen=True)
class Artifact:
    name: str
    path: str
    kind: str
    meta: dict[str, Any]


@dataclass(frozen=True)
class BuildResult:
    run_id: str
    run_dir: str
    spec_hash: str
    pack: str
    entrypoint: str
    status: RunStatus
    artifacts: dict[str, Artifact]


def run(
    fn: Callable[..., dict[str, Artifact]],
    *,
    pack: str,
    entrypoint: str,
    cfg: dict[str, Any],
    seed: int | None = None,
    reuse_completed: bool = True,
    resume_incomplete: bool = True,
) -> BuildResult:
    """Execute one explicit work unit and record it under a run directory."""

    spec = {
        "pack": pack,
        "entrypoint": entrypoint,
        "cfg": cfg,
        "seed": seed,
    }
    spec_hash = hash_spec(spec)

    if reuse_completed:
        reused = resume(spec_hash, pack=pack)
        if reused is not None:
            return reused

    resumed = False
    if resume_incomplete:
        resumed_manifest = _resume_incomplete_manifest(spec_hash, pack=pack)
    else:
        resumed_manifest = None

    if resumed_manifest is None:
        run_id = _new_run_id()
        run_dir = artifacts_root() / "runs" / pack / run_id
        outputs_dir = ensure_dir(run_dir / "outputs")
        ensure_dir(run_dir / "logs")
        git_state = git_info()

        manifest = {
            "run_id": run_id,
            "spec_hash": spec_hash,
            "pack": pack,
            "entrypoint": entrypoint,
            "created_at": iso_timestamp(),
            "started_at": iso_timestamp(),
            "finished_at": None,
            "git_commit": git_state["git_commit"],
            "git_dirty": git_state["git_dirty"],
            "seed": seed,
            "output_artifacts": {},
            "models": _model_manifest(cfg),
            "error": None,
            "resume_count": 0,
        }
        write_json(manifest, run_dir / "manifest.json")
        write_yaml(cfg, run_dir / "config.yaml")
    else:
        run_dir, manifest = resumed_manifest
        run_id = manifest["run_id"]
        outputs_dir = ensure_dir(run_dir / "outputs")
        ensure_dir(run_dir / "logs")
        manifest.pop("status", None)
        manifest["finished_at"] = None
        manifest["error"] = None
        manifest["resumed_at"] = iso_timestamp()
        manifest["resume_count"] = int(manifest.get("resume_count", 0)) + 1
        write_json(manifest, run_dir / "manifest.json")
        resumed = True

    with activate_run_log(run_dir):
        log_event(
            "run",
            "started",
            pack=pack,
            entrypoint=entrypoint,
            run_id=run_id,
            spec_hash=spec_hash,
            resumed=resumed,
        )
        try:
            artifacts = fn(
                cfg=cfg,
                outputs_dir=outputs_dir,
                seed=seed,
            )
        except Exception as error:
            manifest["finished_at"] = iso_timestamp()
            manifest["error"] = {
                "type": error.__class__.__name__,
                "message": str(error),
            }
            write_json(manifest, run_dir / "manifest.json")
            log_event(
                "run",
                "failed",
                pack=pack,
                entrypoint=entrypoint,
                run_id=run_id,
                error_type=error.__class__.__name__,
                error_message=str(error),
            )
            raise

        completion = {
            "completed_at": iso_timestamp(),
            "artifacts": {
                name: _artifact_to_dict(artifact)
                for name, artifact in artifacts.items()
            },
        }
        write_json(completion, outputs_dir / "_artifacts.json")

        manifest["finished_at"] = completion["completed_at"]
        manifest["output_artifacts"] = completion["artifacts"]
        manifest["error"] = None
        write_json(manifest, run_dir / "manifest.json")
        log_event(
            "run",
            "completed",
            pack=pack,
            entrypoint=entrypoint,
            run_id=run_id,
            artifacts=sorted(artifacts),
        )

    return BuildResult(
        run_id=run_id,
        run_dir=str(run_dir),
        spec_hash=spec_hash,
        pack=pack,
        entrypoint=entrypoint,
        status="completed",
        artifacts=artifacts,
    )


def _repair_manifest_from_outputs(run_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    if run_status(manifest) == "completed":
        return manifest

    completion_path = run_dir / "outputs" / "_artifacts.json"
    if not completion_path.exists():
        return manifest

    completion = read_json(completion_path)
    manifest["finished_at"] = completion.get("completed_at") or iso_timestamp()
    manifest["output_artifacts"] = completion.get("artifacts", {})
    manifest["error"] = None
    write_json(manifest, run_dir / "manifest.json")
    return manifest


def load(run_id_or_path: str) -> BuildResult:
    """Load one completed or failed run without executing user code."""

    run_dir = _resolve_run_dir(run_id_or_path)
    manifest = _repair_manifest_from_outputs(run_dir, read_json(run_dir / "manifest.json"))
    return _result_from_manifest(run_dir, manifest)


def compare(left: str, right: str) -> dict[str, Any]:
    """Compare two runs by manifest, metrics, and named outputs."""

    left_run = _resolve_run_dir(left)
    right_run = _resolve_run_dir(right)
    left_manifest = read_json(left_run / "manifest.json")
    right_manifest = read_json(right_run / "manifest.json")

    fields = ["pack", "entrypoint", "status", "seed", "git_commit", "spec_hash"]
    manifest_diff = {
        field: {
            "left": run_status(left_manifest) if field == "status" else left_manifest.get(field),
            "right": run_status(right_manifest) if field == "status" else right_manifest.get(field),
        }
        for field in fields
        if (
            run_status(left_manifest) if field == "status" else left_manifest.get(field)
        ) != (
            run_status(right_manifest) if field == "status" else right_manifest.get(field)
        )
    }

    left_metrics = _read_optional_json(left_run / "outputs" / "metrics.json")
    right_metrics = _read_optional_json(right_run / "outputs" / "metrics.json")

    return {
        "left_run_id": left_manifest["run_id"],
        "right_run_id": right_manifest["run_id"],
        "manifest_diff": manifest_diff,
        "metrics": {
            "left": left_metrics,
            "right": right_metrics,
        },
        "output_artifacts": {
            "left": sorted(left_manifest.get("output_artifacts", {}).keys()),
            "right": sorted(right_manifest.get("output_artifacts", {}).keys()),
        },
    }


def read_events(
    run_id_or_path: str,
    *,
    component: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    run_dir = _resolve_run_dir(run_id_or_path)
    events_path = run_dir / "logs" / "events.jsonl"
    if not events_path.exists():
        return []

    rows = _read_jsonl(events_path)
    if component is not None:
        rows = [row for row in rows if row.get("component") == component]
    if limit is None:
        return rows
    assert limit >= 0, "limit must be non-negative"
    if limit == 0:
        return []
    return rows[-limit:]


def progress(
    run_id_or_path: str,
    *,
    include_model_events: bool = False,
    limit: int = 20,
) -> dict[str, Any]:
    result = load(run_id_or_path)
    run_dir = Path(result.run_dir)
    manifest = read_json(run_dir / "manifest.json")
    outputs_dir = run_dir / "outputs"
    events = read_events(run_id_or_path)
    visible_events = events if include_model_events else [row for row in events if row.get("component") != "model"]

    return {
        "run_id": result.run_id,
        "pack": result.pack,
        "entrypoint": result.entrypoint,
        "status": result.status,
        "started_at": manifest.get("started_at"),
        "finished_at": manifest.get("finished_at"),
        "resume_count": manifest.get("resume_count", 0),
        "error": manifest.get("error"),
        "snapshots": _progress_snapshots(outputs_dir),
        "event_counts": _event_counts(events),
        "recent_events": _tail(visible_events, limit),
    }


def resume(spec_hash: str, *, pack: str | None = None) -> BuildResult | None:
    """Return a previously completed run with the same spec hash, if present."""

    matches: list[tuple[str, Path]] = []

    for manifest_path in _manifest_paths(pack):
        manifest = _repair_manifest_from_outputs(manifest_path.parent, read_json(manifest_path))
        if run_status(manifest) != "completed":
            continue
        if manifest.get("spec_hash") != spec_hash:
            continue
        matches.append((manifest.get("finished_at") or "", manifest_path.parent))

    if not matches:
        return None

    matches.sort(key=lambda item: item[0], reverse=True)
    return load(str(matches[0][1]))


def _resume_incomplete_manifest(spec_hash: str, *, pack: str) -> tuple[Path, dict[str, Any]] | None:
    matches: list[tuple[str, Path, dict[str, Any]]] = []
    for manifest_path in _manifest_paths(pack):
        manifest = _repair_manifest_from_outputs(manifest_path.parent, read_json(manifest_path))
        if run_status(manifest) not in {"running", "failed"}:
            continue
        if manifest.get("spec_hash") != spec_hash:
            continue
        matches.append((manifest.get("created_at") or "", manifest_path.parent, manifest))

    if not matches:
        return None

    matches.sort(key=lambda item: item[0], reverse=True)
    _, run_dir, manifest = matches[0]
    return run_dir, manifest


def _new_run_id() -> str:
    timestamp = iso_timestamp().replace(":", "").replace("+00:00", "Z")
    return f"{timestamp}-{uuid4().hex[:8]}"


def _resolve_run_dir(run_id_or_path: str) -> Path:
    candidate = Path(run_id_or_path).expanduser()
    if candidate.exists():
        return candidate.resolve()

    matches = list((artifacts_root() / "runs").glob(f"*/{run_id_or_path}"))
    if len(matches) != 1:
        raise ValueError(f"Could not resolve run: {run_id_or_path}")
    return matches[0].resolve()


def _manifest_paths(pack: str | None) -> list[Path]:
    runs_root = artifacts_root() / "runs"
    if pack:
        return sorted((runs_root / pack).glob("*/manifest.json"))
    return sorted(runs_root.glob("*/*/manifest.json"))


def _model_manifest(cfg: dict[str, Any]) -> dict[str, Any]:
    models = cfg.get("models", {})
    if not isinstance(models, dict):
        return {}

    return {
        name: {
            key: value
            for key, value in spec.items()
            if key in {"model", "base_url", "api_key_env"}
        }
        for name, spec in models.items()
        if isinstance(spec, dict)
    }


def _artifact_to_dict(artifact: Artifact) -> dict[str, Any]:
    return {
        "name": artifact.name,
        "path": artifact.path,
        "kind": artifact.kind,
        "meta": artifact.meta,
    }


def run_status(manifest: dict[str, Any]) -> RunStatus:
    if manifest.get("error") is not None:
        return "failed"
    if manifest.get("finished_at") is not None:
        return "completed"
    legacy_status = manifest.get("status")
    if legacy_status in {"running", "completed", "failed"}:
        return cast(RunStatus, legacy_status)
    return "running"


def _result_from_manifest(run_dir: Path, manifest: dict[str, Any]) -> BuildResult:
    artifacts = {
        name: Artifact(
            name=name,
            path=spec["path"],
            kind=spec["kind"],
            meta=spec.get("meta", {}),
        )
        for name, spec in manifest.get("output_artifacts", {}).items()
    }

    return BuildResult(
        run_id=manifest["run_id"],
        run_dir=str(run_dir),
        spec_hash=manifest["spec_hash"],
        pack=manifest["pack"],
        entrypoint=manifest["entrypoint"],
        status=run_status(manifest),
        artifacts=artifacts,
    )


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return read_json(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _progress_snapshots(outputs_dir: Path) -> dict[str, Any]:
    snapshots: dict[str, Any] = {}
    for path in sorted(outputs_dir.glob("*.json")):
        if path.name in {"metrics.json", "failure_summary.json"}:
            continue
        if not (path.name.endswith("_progress.json") or path.name.endswith("_metrics.json")):
            continue
        snapshots[path.stem] = read_json(path)
    return snapshots


def _event_counts(events: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in events:
        component = str(row.get("component"))
        counts[component] = counts.get(component, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _tail(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    assert limit >= 0, "limit must be non-negative"
    if limit == 0:
        return []
    return rows[-limit:]
