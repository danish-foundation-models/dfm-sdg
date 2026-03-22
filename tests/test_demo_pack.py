from __future__ import annotations

from pathlib import Path

import httpx

from sdg.commons.run import load, progress, read_events
from sdg.commons.utils import read_yaml
from sdg.commons.viewer import render_run_view, start_viewer_server
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
    run_progress = progress(first.run_id)
    assert run_progress["status"] == "completed"
    assert run_progress["event_counts"]["run"] == 2
    assert run_progress["recent_events"][-1]["event"] == "completed"

    verification = verify(first.run_id)
    assert verification["failed_rows"] == 0

    summary = summarize(first.run_id)
    assert summary["rows"] == cfg["generation"]["count"]
    assert summary["metrics"]["checks"]["answer_correct"]["failed"] == 0

    view = render_run_view(first.run_id, limit=10)
    viewer_path = Path(view["out_path"])
    assert view["default_artifact"] == "dataset"
    assert viewer_path.exists()
    viewer_html = viewer_path.read_text()
    assert "Add " in viewer_html
    assert "Rows per page" in viewer_html
    assert "section-markdown" in viewer_html

    running = start_viewer_server(first.run_id, host="127.0.0.1", port=0)
    try:
        with httpx.Client(base_url=running.base_url, timeout=5.0) as client:
            run_payload = client.get("/api/run").json()
            assert run_payload["default_artifact"] == "dataset"
            progress_payload = client.get("/api/progress").json()
            assert progress_payload["status"] == "completed"
            assert "artifacts" in progress_payload
            page = client.get(
                "/api/artifact",
                params={"name": "dataset", "page": 1, "page_size": 5},
            ).json()
            assert page["artifact"]["name"] == "dataset"
            assert page["filtered_count"] == cfg["generation"]["count"]
            assert len(page["items"]) == 5
            assert page["items"][0]["sections"][0]["format"] in {"plain", "code", "markdown"}
            html = client.get("/").text
            assert "Copy Row JSON" in html
            assert "section-markdown" in html
    finally:
        running.close()

    published = publish(first.run_id)
    out_dir = Path(published["out_dir"])

    assert (out_dir / "train.parquet").exists()
    assert (out_dir / "eval.parquet").exists()
    assert (out_dir / "failures.parquet").exists()
    assert (out_dir / "manifest.json").exists()
