from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def artifacts_root() -> Path:
    override = os.environ.get("SDG_ARTIFACTS_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return project_root() / "artifacts"


def reports_root() -> Path:
    override = os.environ.get("SDG_REPORTS_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return project_root() / "reports"


def packs_root() -> Path:
    override = os.environ.get("SDG_PACKS_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return project_root() / "sdg" / "packs"


def utc_now() -> datetime:
    return datetime.now(UTC)


def iso_timestamp() -> str:
    return utc_now().replace(microsecond=0).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def hash_spec(value: Any) -> str:
    payload = canonical_json(value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_json(path: str | Path) -> Any:
    with Path(path).open() as handle:
        return json.load(handle)


def write_json(value: Any, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return target


def read_yaml(path: str | Path) -> Any:
    with Path(path).open() as handle:
        return yaml.safe_load(handle)


def write_yaml(value: Any, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w") as handle:
        yaml.safe_dump(value, handle, sort_keys=True)
    return target


def git_info() -> dict[str, Any]:
    root = project_root()

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()

    return {
        "git_commit": commit or None,
        "git_dirty": bool(dirty),
    }
