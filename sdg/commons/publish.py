from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from sdg.commons.store import ensure_dir, write_jsonl
from sdg.commons.utils import write_json


def publish_dataset(artifact, out_dir: str | Path) -> Path:
    target_dir = ensure_dir(out_dir)
    source = Path(artifact.path)
    target = target_dir / source.name

    if source.is_dir():
        shutil.copytree(source, target, dirs_exist_ok=True)
        return target

    shutil.copy2(source, target)
    return target


def write_manifest(run_info: dict[str, Any], path: str | Path) -> Path:
    return write_json(run_info, path)


def write_report(metrics: dict[str, Any], failure_summary: dict[str, Any], path: str | Path) -> Path:
    return write_json({"metrics": metrics, "failure_summary": failure_summary}, path)


def write_preview(rows: list[dict[str, Any]], path: str | Path, n: int = 100) -> Path:
    return write_jsonl(rows[:n], path)
