from __future__ import annotations

from pathlib import Path
from typing import Any

from sdg.commons import publish as common_publish
from sdg.commons import store
from sdg.commons.run import BuildResult
from sdg.commons.utils import read_json, read_yaml, reports_root, write_json


def publish_run(
    result: BuildResult,
    *,
    metrics: dict[str, Any],
    failure_summary: dict[str, Any],
    out_dir: str | None = None,
) -> dict[str, Any]:
    target_dir = _publish_dir(result, out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    for artifact in result.artifacts.values():
        common_publish.publish_dataset(artifact, target_dir)

    write_json(metrics, target_dir / "metrics.json")
    write_json(failure_summary, target_dir / "failure_summary.json")
    common_publish.write_report(metrics, failure_summary, target_dir / "report.json")
    artifact_names = sorted(path.name for path in target_dir.iterdir())
    common_publish.write_manifest(
        {
            "pack": result.pack,
            "run_id": result.run_id,
            "source_run_dir": result.run_dir,
            "published_artifacts": sorted({*artifact_names, "manifest.json"}),
        },
        target_dir / "manifest.json",
    )

    return {
        "run_id": result.run_id,
        "out_dir": str(target_dir),
        "artifacts": sorted(path.name for path in target_dir.iterdir()),
    }


def load_existing_quality(result: BuildResult) -> tuple[dict[str, Any], dict[str, Any]]:
    outputs_dir = Path(result.run_dir) / "outputs"
    metrics = read_json(outputs_dir / "metrics.json") if (outputs_dir / "metrics.json").exists() else {}
    failure_summary = (
        read_json(outputs_dir / "failure_summary.json")
        if (outputs_dir / "failure_summary.json").exists()
        else {}
    )
    return metrics, failure_summary


def _publish_dir(result: BuildResult, out_dir: str | None) -> Path:
    if out_dir:
        return Path(out_dir).expanduser().resolve()
    return reports_root() / result.pack / result.run_id


def publish_generated_dataset(result: BuildResult, target_dir: Path) -> dict[str, int]:
    outputs_dir = Path(result.run_dir) / "outputs"
    verified_path = outputs_dir / "memorization_verified.jsonl"
    rows_path = outputs_dir / "memorization_rows.jsonl"
    if verified_path.exists():
        rows = store.read_jsonl(verified_path)
    elif rows_path.exists():
        rows = store.read_jsonl(rows_path)
    else:
        return {"rows": 0, "train_rows": 0, "eval_rows": 0, "failure_rows": 0}

    export_rows = [strip_hidden(row) for row in rows]
    failures = [row for row in export_rows if any(not passed for passed in row.get("checks", {}).values())]

    cfg = read_yaml(Path(result.run_dir) / "config.yaml")
    train_fraction = cfg.get("generation", {}).get("train_fraction", 0.8)
    train_rows, eval_rows = split_train_eval(export_rows, train_fraction)

    store.write_parquet(train_rows, target_dir / "train.parquet")
    store.write_parquet(eval_rows, target_dir / "eval.parquet")
    store.write_parquet(failures, target_dir / "failures.parquet")
    common_publish.write_preview(export_rows, target_dir / "sample_preview.jsonl", n=100)

    return {
        "rows": len(export_rows),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "failure_rows": len(failures),
    }


def split_train_eval(rows: list[dict[str, Any]], train_fraction: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    split_at = int(len(rows) * train_fraction)
    return rows[:split_at], rows[split_at:]


def strip_hidden(row: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in row.items():
        if key == "hidden":
            continue
        if isinstance(value, dict) and not value:
            continue
        cleaned[key] = value
    return cleaned
