from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any

from sdg.commons import Artifact, BuildResult, store
from sdg.commons import eval as common_eval
from sdg.commons import publish as common_publish
from sdg.commons.run import load, run
from sdg.commons.utils import read_json, read_yaml, reports_root, write_json


@dataclass(frozen=True)
class StarterRowSpec:
    title: str
    prompt: str
    target: str
    primary_subset: str
    subset_inspirations: tuple[str, ...]
    verification: str


@dataclass(frozen=True)
class StarterPackSpec:
    name: str
    family: str
    description: str
    goal: str
    getting_started: tuple[str, ...]
    starter_rows: tuple[StarterRowSpec, ...]


def build_starter_pack(cfg: dict[str, Any], *, spec: StarterPackSpec) -> BuildResult:
    def _build_run(
        *,
        cfg: dict[str, Any],
        outputs_dir: Path,
        seed: int | None,
    ) -> dict[str, Artifact]:
        rows = _materialize_rows(cfg, spec=spec, seed=seed)
        dataset_path = store.write_jsonl(rows, outputs_dir / "dataset.jsonl")
        guide = _guide_payload(spec, rows)
        guide_path = write_json(guide, outputs_dir / "starter_guide.json")
        common_publish.write_preview(rows, outputs_dir / "sample_preview.jsonl", n=20)

        return {
            "dataset": Artifact(
                name="dataset",
                path=str(dataset_path),
                kind="jsonl",
                meta={"rows": len(rows), "family": spec.family},
            ),
            "starter_guide": Artifact(
                name="starter_guide",
                path=str(guide_path),
                kind="json",
                meta={
                    "goal": spec.goal,
                    "recommended_subsets": guide["recommended_subsets"],
                },
            ),
        }

    return run(
        _build_run,
        pack=spec.name,
        entrypoint="build",
        cfg=cfg,
        seed=cfg.get("seed"),
        reuse_completed=cfg.get("reuse_completed", True),
    )


def verify_starter_pack(run_id_or_path: str, *, spec: StarterPackSpec) -> dict[str, Any]:
    result = load(run_id_or_path)
    rows = _load_rows(result)
    verified_rows = common_eval.verify(rows, _has_prompt_target, name="has_prompt_target")
    verified_rows = common_eval.verify(verified_rows, _has_sources, name="has_sources")
    verified_rows = common_eval.verify(verified_rows, _has_starter_metadata, name="has_starter_metadata")
    failures = [row for row in verified_rows if _row_failed(row)]

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
        "goal": spec.goal,
        "verified_rows": len(verified_rows),
        "failed_rows": len(failures),
        "metrics": metrics,
        "failure_summary": failure_summary,
    }


def summarize_starter_pack(run_id_or_path: str) -> dict[str, Any]:
    result = load(run_id_or_path)
    rows = _load_rows(result)
    outputs_dir = Path(result.run_dir) / "outputs"
    metrics_path = outputs_dir / "metrics.json"
    metrics = read_json(metrics_path) if metrics_path.exists() else common_eval.aggregate_metrics(rows)
    guide = read_json(outputs_dir / "starter_guide.json")

    return {
        "pack": result.pack,
        "run_id": result.run_id,
        "status": result.status,
        "rows": len(rows),
        "artifacts": sorted(result.artifacts),
        "goal": guide["goal"],
        "recommended_subsets": guide["recommended_subsets"],
        "source_inspirations": guide["source_inspirations"],
        "getting_started": guide["getting_started"],
        "metrics": metrics,
    }


def publish_starter_pack(run_id_or_path: str, out_dir: str | None = None) -> dict[str, Any]:
    result = load(run_id_or_path)
    rows = _load_verified_rows(result)
    cfg = _load_cfg(result)
    failures = [row for row in rows if _row_failed(row)]
    generation = cfg["generation"]
    train_rows, eval_rows = _split_rows(rows, generation["train_fraction"])

    target_dir = _publish_dir(result, out_dir)
    store.ensure_dir(target_dir)
    store.write_parquet(train_rows, target_dir / "train.parquet")
    store.write_parquet(eval_rows, target_dir / "eval.parquet")
    store.write_parquet(failures, target_dir / "failures.parquet")
    common_publish.write_preview(rows, target_dir / "sample_preview.jsonl", n=20)

    outputs_dir = Path(result.run_dir) / "outputs"
    guide = read_json(outputs_dir / "starter_guide.json")
    metrics = _load_or_compute(outputs_dir / "metrics.json", common_eval.aggregate_metrics(rows))
    failure_summary = _load_or_compute(outputs_dir / "failure_summary.json", common_eval.summarize_failures(rows))

    write_json(guide, target_dir / "starter_guide.json")
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
                "starter_guide.json",
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


def _materialize_rows(
    cfg: dict[str, Any],
    *,
    spec: StarterPackSpec,
    seed: int | None,
) -> list[dict[str, Any]]:
    generation = cfg["generation"]
    sample_count = generation["sample_count"]
    shuffle_examples = generation.get("shuffle_examples", False)
    templates = list(spec.starter_rows)
    assert templates, "starter packs require at least one starter row"

    if shuffle_examples:
        rng = Random(seed if seed is not None else 0)
        rng.shuffle(templates)

    rows: list[dict[str, Any]] = []
    for index in range(sample_count):
        template = templates[index % len(templates)]
        rows.append(
            {
                "id": f"{spec.name}-{index:05d}",
                "title": template.title,
                "prompt": template.prompt,
                "target": template.target,
                "sources": [
                    {"kind": "dolci_subset", "value": subset}
                    for subset in template.subset_inspirations
                ],
                "meta": {
                    "family": spec.family,
                    "goal": spec.goal,
                    "primary_subset": template.primary_subset,
                    "verification": template.verification,
                },
            }
        )
    return rows


def _guide_payload(spec: StarterPackSpec, rows: list[dict[str, Any]]) -> dict[str, Any]:
    recommended_subsets = list(dict.fromkeys(row["meta"]["primary_subset"] for row in rows))
    source_inspirations = sorted(
        {
            source["value"]
            for row in rows
            for source in row["sources"]
        }
    )
    return {
        "pack": spec.name,
        "family": spec.family,
        "description": spec.description,
        "goal": spec.goal,
        "recommended_subsets": recommended_subsets,
        "source_inspirations": source_inspirations,
        "getting_started": list(spec.getting_started),
    }


def _load_rows(result: BuildResult) -> list[dict[str, Any]]:
    dataset_path = Path(result.artifacts["dataset"].path)
    return store.read_jsonl(dataset_path)


def _load_verified_rows(result: BuildResult) -> list[dict[str, Any]]:
    verified_path = Path(result.run_dir) / "outputs" / "verified.jsonl"
    if verified_path.exists():
        return store.read_jsonl(verified_path)
    return _load_rows(result)


def _load_cfg(result: BuildResult) -> dict[str, Any]:
    return read_yaml(Path(result.run_dir) / "config.yaml")


def _has_prompt_target(row: dict[str, Any]) -> bool:
    prompt = str(row.get("prompt", "")).strip()
    target = str(row.get("target", "")).strip()
    return bool(prompt and target)


def _has_sources(row: dict[str, Any]) -> bool:
    sources = row.get("sources")
    if not isinstance(sources, list) or not sources:
        return False
    return all(isinstance(source, dict) and bool(source.get("value")) for source in sources)


def _has_starter_metadata(row: dict[str, Any]) -> bool:
    meta = row.get("meta")
    if not isinstance(meta, dict):
        return False

    return all(
        bool(meta.get(key))
        for key in ("family", "goal", "primary_subset", "verification")
    )


def _row_failed(row: dict[str, Any]) -> bool:
    checks = row.get("checks")
    if not isinstance(checks, dict):
        return False
    return any(not _check_passed(value) for value in checks.values())


def _check_passed(value: object) -> bool:
    if isinstance(value, bool):
        return value

    assert isinstance(value, dict), "check value must be a bool or a status mapping"
    if "passed" in value:
        return bool(value["passed"])
    if "ok" in value:
        return bool(value["ok"])
    raise AssertionError("check status mapping must contain 'passed' or 'ok'")


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
