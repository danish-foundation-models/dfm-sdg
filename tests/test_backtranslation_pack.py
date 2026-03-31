from __future__ import annotations

from pathlib import Path

import httpx

from sdg.commons import store
from sdg.commons.run import load, progress, read_events
from sdg.commons.viewer import render_run_view, start_viewer_server
from sdg.packs.backtranslation.build import (
    _count_articles,
    _iter_articles,
    _load_resume_state,
    build,
    publish,
    summarize,
    verify,
)


class FakeInstructionWriter:
    def chat(self, messages, temperature=0.0):
        del temperature
        user = messages[1]["content"]
        title = _line_value(user, "Article title")
        return f"Write a concise encyclopedia article about {title}."


def _line_value(text: str, label: str) -> str:
    prefix = f"{label}: "
    for line in text.splitlines():
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    raise AssertionError(f"Missing {label} in prompt:\n{text}")


def test_backtranslation_pack_end_to_end(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SDG_ARTIFACTS_ROOT", str(tmp_path / "artifacts"))
    monkeypatch.setenv("SDG_REPORTS_ROOT", str(tmp_path / "reports"))
    monkeypatch.setattr(
        "sdg.packs.backtranslation.build._load_instruction_writer",
        lambda cfg: FakeInstructionWriter(),
    )

    source_path = tmp_path / "articles.jsonl"
    store.write_jsonl(
        [
            {
                "id": "a1",
                "title": "Aarhus",
                "url": "https://example.com/aarhus",
                "text": (
                    "Aarhus is Denmark's second-largest city. "
                    "It is known for its port, university, and cultural institutions."
                ),
            },
            {
                "id": "a2",
                "title": "Short entry",
                "url": "https://example.com/short",
                "text": "Too short.",
            },
            {
                "id": "a3",
                "title": "Odense",
                "url": "https://example.com/odense",
                "text": (
                    "Odense is a city on the island of Funen. "
                    "It is closely associated with Hans Christian Andersen and has a long urban history."
                ),
            },
        ],
        source_path,
    )

    cfg = {
        "pack": "backtranslation",
        "reuse_completed": True,
        "models": {"instruction_writer": "openai"},
        "source": {
            "path": str(source_path),
            "text_field": "text",
            "title_field": "title",
            "id_field": "id",
            "url_field": "url",
        },
        "generation": {
            "min_article_chars": 80,
            "max_articles": 10,
            "temperature": 0.0,
            "train_fraction": 0.5,
        },
    }

    first = build(cfg)
    second = build(cfg)

    assert second.run_id == first.run_id

    loaded = load(first.run_id)
    assert loaded.pack == "backtranslation"
    assert "dataset" in loaded.artifacts

    dataset_rows = store.read_jsonl(Path(loaded.artifacts["dataset"].path))
    assert len(dataset_rows) == 2
    assert dataset_rows[0]["prompt"] == "Write a concise encyclopedia article about Aarhus."
    assert dataset_rows[0]["target"].startswith("Aarhus is Denmark")
    assert dataset_rows[1]["meta"]["title"] == "Odense"

    run_events = read_events(first.run_id, component="run")
    assert [event["event"] for event in run_events] == ["started", "completed"]
    run_progress = progress(first.run_id)
    assert run_progress["status"] == "completed"

    verification = verify(first.run_id)
    assert verification["failed_rows"] == 0

    summary = summarize(first.run_id)
    assert summary["rows"] == 2
    assert summary["metrics"]["checks"]["target_min_chars"]["failed"] == 0

    view = render_run_view(first.run_id, limit=10)
    viewer_path = Path(view["out_path"])
    assert view["default_artifact"] == "dataset"
    assert viewer_path.exists()
    viewer_html = viewer_path.read_text()
    assert "Write a concise encyclopedia article about Aarhus." in viewer_html
    assert "Rows per page" in viewer_html

    running = start_viewer_server(first.run_id, host="127.0.0.1", port=0)
    try:
        with httpx.Client(base_url=running.base_url, timeout=5.0) as client:
            run_payload = client.get("/api/run").json()
            assert run_payload["default_artifact"] == "dataset"
            progress_payload = client.get("/api/progress").json()
            assert progress_payload["status"] == "completed"
            page = client.get(
                "/api/artifact",
                params={"name": "dataset", "page": 1, "page_size": 5},
            ).json()
            assert page["artifact"]["name"] == "dataset"
            assert page["filtered_count"] == 2
            assert len(page["items"]) == 2
            html = client.get("/").text
            assert "Copy Row JSON" in html
    finally:
        running.close()

    published = publish(first.run_id)
    out_dir = Path(published["out_dir"])

    assert (out_dir / "train.parquet").exists()
    assert (out_dir / "eval.parquet").exists()
    assert (out_dir / "failures.parquet").exists()
    assert (out_dir / "manifest.json").exists()


def test_backtranslation_resume_state_respects_max_articles(tmp_path) -> None:
    source_path = tmp_path / "articles.jsonl"
    store.write_jsonl(
        [
            {"id": "a1", "title": "One", "text": "A" * 100},
            {"id": "a2", "title": "Two", "text": "B" * 100},
            {"id": "a3", "title": "Three", "text": "C" * 100},
            {"id": "a4", "title": "Four", "text": "D" * 100},
        ],
        source_path,
    )

    dataset_path = tmp_path / "dataset.jsonl"
    failures_path = tmp_path / "generation_failures.jsonl"
    store.write_jsonl(
        [
            {
                "id": "backtranslation-000000",
                "prompt": "Prompt 1",
                "target": "A" * 100,
                "meta": {"source_id": "a1"},
            }
        ],
        dataset_path,
    )
    store.write_jsonl(
        [
            {
                "id": "backtranslation-failure-000001",
                "source_id": "a2",
                "error_type": "RuntimeError",
                "error_message": "boom",
            }
        ],
        failures_path,
    )

    cfg = {
        "source": {
            "path": str(source_path),
            "text_field": "text",
            "title_field": "title",
            "id_field": "id",
        },
        "generation": {
            "min_article_chars": 80,
            "max_articles": 3,
        },
    }

    resume_state = _load_resume_state(
        dataset_path=dataset_path,
        failures_path=failures_path,
    )

    assert resume_state["completed_rows"] == 1
    assert resume_state["failed_rows"] == 1
    assert resume_state["processed_source_ids"] == {"a1", "a2"}

    stats = _count_articles(cfg, processed_source_ids=resume_state["processed_source_ids"])

    assert stats["resumed_rows"] == 2
    assert stats["pending_rows"] == 1
    assert stats["max_articles"] == 3

    pending_articles = list(
        _iter_articles(cfg, processed_source_ids=resume_state["processed_source_ids"])
    )

    assert [article["source_id"] for article in pending_articles] == ["a3"]
    assert [article["row_index"] for article in pending_articles] == [2]
