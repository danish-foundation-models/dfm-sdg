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


def upload_dataset_artifact(
    artifact,
    *,
    repo_id: str,
    split: str = "train",
    private: bool = False,
    commit_message: str | None = None,
) -> dict[str, Any]:
    source = Path(artifact.path)
    if not source.exists():
        raise FileNotFoundError(f"Artifact path does not exist: {source}")
    if source.is_dir():
        raise ValueError(f"Artifact upload only supports file artifacts, got directory: {source}")

    dataset = _load_hub_dataset(source, split=split)
    dataset.push_to_hub(
        repo_id,
        split=split,
        private=private,
        commit_message=commit_message,
    )
    return {
        "repo_id": repo_id,
        "split": split,
        "private": private,
        "source_path": str(source),
        "rows": dataset.num_rows,
        "columns": list(dataset.column_names),
    }


def _load_hub_dataset(source: Path, *, split: str):
    from datasets import load_dataset

    suffix = source.suffix.lower()
    if suffix in {".jsonl", ".json"}:
        dataset_dict = load_dataset("json", data_files={split: str(source)})
        return dataset_dict[split]

    if suffix == ".parquet":
        dataset_dict = load_dataset("parquet", data_files={split: str(source)})
        return dataset_dict[split]

    raise ValueError(
        "HF upload only supports .jsonl, .json, and .parquet artifacts, "
        f"got: {source.name}"
    )
