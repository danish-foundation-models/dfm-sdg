from __future__ import annotations

from pathlib import Path

from sdg.commons.run import load, read_events
from sdg.commons.utils import read_yaml
from sdg.packs.demo.build import build, publish, summarize, verify


def test_demo_pack_end_to_end(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SDG_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("SDG_REPORTS_ROOT", str(tmp_path / "reports"))

    cfg_path = Path(__file__).resolve().parents[1] / "sdg" / "packs" / "demo" / "configs" / "base.yaml"
    cfg = read_yaml(cfg_path)

    first = build(cfg)
    second = build(cfg)

    assert second.run_id == first.run_id

    loaded = load(first.run_id)
    assert loaded.pack == "demo"
    assert "dataset" in loaded.artifacts
    run_events = read_events(first.run_id, component="run")
    assert [event["event"] for event in run_events] == ["started", "completed"]
    assert read_events(first.run_id, component="run", limit=1)[0]["event"] == "completed"

    verification = verify(first.run_id)
    assert verification["failed_rows"] == 0

    summary = summarize(first.run_id)
    assert summary["rows"] == cfg["generation"]["count"]
    assert summary["metrics"]["checks"]["answer_correct"]["failed"] == 0

    published = publish(first.run_id)
    out_dir = Path(published["out_dir"])

    assert (out_dir / "train.parquet").exists()
    assert (out_dir / "eval.parquet").exists()
    assert (out_dir / "failures.parquet").exists()
    assert (out_dir / "manifest.json").exists()
