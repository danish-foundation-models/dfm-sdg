from __future__ import annotations

from pathlib import Path

from sdg.commons import store
from sdg.commons.run import Artifact
from sdg.commons.viewer import (
    _build_jsonl_offsets,
    _compact_summary,
    _default_artifact_name,
    _discover_live_artifacts,
    _resolve_artifact_view,
    _slice_jsonl_rows_with_offsets,
    _value_at_path,
    _viewer_item,
)
from sdg.packs.multi_turn_dialogue.viewer import (
    viewer_spec as multi_turn_dialogue_viewer_spec,
)
from sdg.packs.verifiable_reasoning.viewer import (
    viewer_spec as verifiable_reasoning_viewer_spec,
)


def test_jsonl_offsets_support_direct_paging(tmp_path) -> None:
    path = tmp_path / "rows.jsonl"
    rows = [{"id": f"row-{index}", "value": index} for index in range(6)]
    store.write_jsonl(rows, path)

    offsets = _build_jsonl_offsets(path)

    assert len(offsets) == 6
    assert _slice_jsonl_rows_with_offsets(path, offsets, offset=0, limit=2) == rows[:2]
    assert _slice_jsonl_rows_with_offsets(path, offsets, offset=2, limit=2) == rows[2:4]
    assert _slice_jsonl_rows_with_offsets(path, offsets, offset=5, limit=3) == rows[5:]


def test_discover_live_artifacts_finds_jsonl_outputs(tmp_path) -> None:
    run_dir = tmp_path / "run"
    outputs_dir = run_dir / "outputs"
    outputs_dir.mkdir(parents=True)
    rows_path = outputs_dir / "grounded_qa_rows.jsonl"
    store.write_jsonl([{"id": "row-1"}], rows_path)

    artifacts = _discover_live_artifacts(Path(run_dir), {})

    assert "grounded_qa_rows" in artifacts
    assert artifacts["grounded_qa_rows"] == Artifact(
        name="grounded_qa_rows",
        path=str(rows_path),
        kind="jsonl",
        meta={},
    )


def test_viewer_item_has_stable_item_and_section_keys() -> None:
    row = {
        "id": "row-7",
        "prompt": "Question",
        "target": "Answer",
        "reasoning": "Reasoning",
    }

    view = _resolve_artifact_view([row], {})
    item = _viewer_item(row, view)

    assert item["key"] == "row-7"
    assert [section["key"] for section in item["sections"]] == ["prompt", "reasoning", "target"]


def test_viewer_message_section_renders_chat_transcript() -> None:
    row = {
        "id": "dialogue-1",
        "messages": [
            {"role": "user", "content": "Hej"},
            {"role": "assistant", "content": "Svar med <forsigtighed>."},
        ],
    }
    view = _resolve_artifact_view(
        [row],
        {
            "detail_sections": [
                {"path": "messages", "label": "Conversation", "format": "messages"},
            ]
        },
    )

    item = _viewer_item(row, view)
    section = item["sections"][0]

    assert section["format"] == "messages"
    assert section["text"] == "user: Hej\n\nassistant: Svar med <forsigtighed>."
    assert 'data-role="user"' in section["html"]
    assert "Svar med &lt;forsigtighed&gt;." in section["html"]


def test_viewer_path_supports_message_indices() -> None:
    row = {
        "messages": [
            {"role": "user", "content": "første"},
            {"role": "assistant", "content": "sidste"},
        ]
    }

    assert _value_at_path(row, "messages.0.content") == "første"
    assert _value_at_path(row, "messages.last.content") == "sidste"
    assert _value_at_path(row, "messages.2.content") is None


def test_default_artifact_name_prefers_pack_viewer_order() -> None:
    artifacts = {
        "dataset": Artifact(name="dataset", path="/tmp/dataset.jsonl", kind="jsonl", meta={}),
        "memory_chunks": Artifact(name="memory_chunks", path="/tmp/memory_chunks.jsonl", kind="jsonl", meta={}),
    }
    spec = {
        "artifacts": {
            "memory_chunks": {"label": "Chunks"},
            "dataset": {"label": "Rows"},
        }
    }

    assert _default_artifact_name(spec, artifacts) == "memory_chunks"


def test_compact_summary_handles_non_string_dict_keys() -> None:
    summary = {
        "family_counts": {
            4: 12,
            "kept_preview": ["omit me"],
        }
    }

    assert _compact_summary(summary) == {"family_counts": {4: 12}}


def test_verifiable_reasoning_viewer_uses_short_titles_and_response_first() -> None:
    row = {
        "id": "verifiable-reasoning-00001",
        "prompt": "A very long prompt that should not become the card title.",
        "target": "Svar:\n1, 2, 3",
        "reasoning": "Short reasoning.",
        "meta": {
            "family": "zebra_logic",
            "prompt_language": "da",
            "difficulty": "medium",
            "target_source": "answer_teacher",
        },
    }

    view = _resolve_artifact_view([row], verifiable_reasoning_viewer_spec()["artifacts"]["dataset"])
    item = _viewer_item(row, view)

    assert item["title"] == "verifiable-reasoning-00001"
    assert item["subtitle"] == "zebra_logic"
    assert item["excerpt"] == "Svar: 1, 2, 3"
    assert item["sections"][0]["label"] == "Response"


def test_multi_turn_dialogue_viewer_uses_conversation_first() -> None:
    row = {
        "id": "multi-turn-dialogue-candidate-00001",
        "messages": [
            {"role": "user", "content": "Kan du hjælpe?"},
            {"role": "assistant", "content": "Ja."},
        ],
        "meta": {
            "family": "general_chat",
            "domain": "home_network_troubleshooting",
            "intent_trajectory": "troubleshooting_interaction",
        },
        "hidden": {
            "review": {"review_decision": "accept"},
            "selection": {"accepted": True},
        },
        "sources": [],
    }

    view = _resolve_artifact_view([row], multi_turn_dialogue_viewer_spec()["artifacts"]["dataset"])
    item = _viewer_item(row, view)

    assert item["title"] == "multi-turn-dialogue-candidate-00001"
    assert item["subtitle"] == "home_network_troubleshooting"
    assert item["excerpt"] == "Kan du hjælpe?"
    assert item["sections"][0]["label"] == "Conversation"
    assert item["sections"][0]["format"] == "messages"
    assert "assistant: Ja." in item["sections"][0]["text"]
    assert "Final Assistant Turn" not in [section["label"] for section in item["sections"]]
