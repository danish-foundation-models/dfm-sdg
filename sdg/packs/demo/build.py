from __future__ import annotations

from pathlib import Path
from random import Random
from typing import Any

from sdg.commons import Artifact, BuildResult, store
from sdg.commons import eval as common_eval
from sdg.commons import publish as common_publish
from sdg.commons.run import load, run
from sdg.commons.utils import read_json, read_yaml, reports_root, write_json


def build(cfg: dict[str, Any]) -> BuildResult:
    """Build a tiny deterministic arithmetic dataset."""

    return run(
        _build_run,
        pack="demo",
        entrypoint="build",
        cfg=cfg,
        seed=cfg.get("seed"),
        reuse_completed=cfg.get("reuse_completed", True),
    )


def verify(run_id_or_path: str) -> dict[str, Any]:
    """Verify arithmetic targets against the hidden answer state."""

    result = load(run_id_or_path)
    rows = _load_rows(result)
    verified_rows = common_eval.verify(rows, _check_answer, name="answer_correct")
    failures = [row for row in verified_rows if not row["checks"]["answer_correct"]]

    outputs_dir = Path(result.run_dir) / "outputs"
    store.write_jsonl(verified_rows, outputs_dir / "verified.jsonl")
    store.write_jsonl(failures, outputs_dir / "failures.jsonl")

    metrics = common_eval.aggregate_metrics(verified_rows)
    failure_summary = common_eval.summarize_failures(verified_rows)
    write_json(metrics, outputs_dir / "metrics.json")
    write_json(failure_summary, outputs_dir / "failure_summary.json")
    common_publish.write_preview(verified_rows, outputs_dir / "sample_preview.jsonl", n=20)

    return {
        "run_id": result.run_id,
        "verified_rows": len(verified_rows),
        "failed_rows": len(failures),
        "metrics": metrics,
        "failure_summary": failure_summary,
    }


def summarize(run_id_or_path: str) -> dict[str, Any]:
    """Summarize a run using available metrics or raw row counts."""

    result = load(run_id_or_path)
    rows = _load_rows(result)
    outputs_dir = Path(result.run_dir) / "outputs"
    metrics_path = outputs_dir / "metrics.json"
    metrics = read_json(metrics_path) if metrics_path.exists() else common_eval.aggregate_metrics(rows)

    return {
        "pack": result.pack,
        "run_id": result.run_id,
        "status": result.status,
        "rows": len(rows),
        "artifacts": sorted(result.artifacts),
        "metrics": metrics,
    }


def publish(run_id_or_path: str, out_dir: str | None = None) -> dict[str, Any]:
    """Export train/eval parquet plus manifest, metrics, failures, and preview."""

    result = load(run_id_or_path)
    rows = _load_verified_rows(result)
    cfg = _load_cfg(result)
    export_rows = [_strip_hidden(row) for row in rows]
    failures = [row for row in export_rows if not row["checks"]["answer_correct"]]
    generation = cfg["generation"]
    train_rows, eval_rows = _split_rows(export_rows, generation["train_fraction"])

    target_dir = _publish_dir(result, out_dir)
    store.ensure_dir(target_dir)
    store.write_parquet(train_rows, target_dir / "train.parquet")
    store.write_parquet(eval_rows, target_dir / "eval.parquet")
    store.write_parquet(failures, target_dir / "failures.parquet")
    common_publish.write_preview(export_rows, target_dir / "sample_preview.jsonl", n=20)

    outputs_dir = Path(result.run_dir) / "outputs"
    metrics = _load_or_compute(outputs_dir / "metrics.json", common_eval.aggregate_metrics(rows))
    failure_summary = _load_or_compute(outputs_dir / "failure_summary.json", common_eval.summarize_failures(rows))

    write_json(metrics, target_dir / "metrics.json")
    write_json(failure_summary, target_dir / "failure_summary.json")
    common_publish.write_report(metrics, failure_summary, target_dir / "report.json")
    common_publish.write_manifest(
        {
            "pack": result.pack,
            "run_id": result.run_id,
            "source_run_dir": result.run_dir,
            "source_artifacts": sorted(result.artifacts),
            "published_artifacts": [
                "train.parquet",
                "eval.parquet",
                "failures.parquet",
                "sample_preview.jsonl",
                "manifest.json",
                "metrics.json",
                "failure_summary.json",
                "report.json",
            ],
        },
        target_dir / "manifest.json",
    )

    return {
        "run_id": result.run_id,
        "out_dir": str(target_dir),
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "failure_rows": len(failures),
    }


def _build_run(
    *,
    cfg: dict[str, Any],
    outputs_dir: Path,
    seed: int | None,
) -> dict[str, Artifact]:
    rows = _make_rows(cfg, seed)
    dataset_path = store.write_jsonl(rows, outputs_dir / "dataset.jsonl")
    common_publish.write_preview(rows, outputs_dir / "sample_preview.jsonl", n=20)

    return {
        "dataset": Artifact(
            name="dataset",
            path=str(dataset_path),
            kind="jsonl",
            meta={"rows": len(rows), "family": "addition"},
        )
    }


def _make_rows(cfg: dict[str, Any], seed: int | None) -> list[dict[str, Any]]:
    generation = cfg["generation"]
    rng = Random(seed if seed is not None else 0)
    count = generation["count"]
    min_value = generation["min_value"]
    max_value = generation["max_value"]

    rows: list[dict[str, Any]] = []
    for index in range(count):
        left = rng.randint(min_value, max_value)
        right = rng.randint(min_value, max_value)
        answer = left + right
        rows.append(
            {
                "id": f"demo-{index:05d}",
                "prompt": f"Add {left} and {right}.",
                "target": str(answer),
                "hidden": {"operands": [left, right], "answer": answer},
                "sources": [],
                "meta": {"family": "addition"},
            }
        )
    return rows


def _load_rows(result: BuildResult) -> list[dict[str, Any]]:
    dataset_path = Path(result.artifacts["dataset"].path)
    return store.read_jsonl(dataset_path)


def _load_verified_rows(result: BuildResult) -> list[dict[str, Any]]:
    verified_path = Path(result.run_dir) / "outputs" / "verified.jsonl"
    if verified_path.exists():
        return store.read_jsonl(verified_path)
    rows = _load_rows(result)
    return common_eval.verify(rows, _check_answer, name="answer_correct")


def _load_cfg(result: BuildResult) -> dict[str, Any]:
    return read_yaml(Path(result.run_dir) / "config.yaml")


def _check_answer(row: dict[str, Any]) -> bool:
    return str(row["hidden"]["answer"]) == str(row["target"])


def _strip_hidden(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "hidden"}


def _split_rows(rows: list[dict[str, Any]], train_fraction: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    split_at = int(len(rows) * train_fraction)
    return rows[:split_at], rows[split_at:]


def _publish_dir(result: BuildResult, out_dir: str | None) -> Path:
    if out_dir:
        return Path(out_dir).expanduser().resolve()
    return reports_root() / result.pack / result.run_id


def _load_or_compute(path: Path, fallback: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        return read_json(path)
    return fallback
