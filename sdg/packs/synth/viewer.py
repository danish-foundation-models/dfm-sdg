from __future__ import annotations

from typing import Any


def viewer_spec() -> dict[str, Any]:
    return {
        "title": "Synth Viewer",
        "default_artifact": "grounded_qa_rows",
        "default_page_size": 30,
        "artifacts": {
            "grounded_qa_rows": _grounded_row_view("Grounded QA kept"),
            "grounded_qa_candidates": _grounded_row_view("Grounded QA candidates"),
            "grounded_qa_rejected": _grounded_row_view("Grounded QA rejected"),
            "memorization_rows": _memorization_view("Memorization kept"),
            "memorization_candidates": _memorization_view("Memorization candidates"),
            "memorization_rejected": _memorization_view("Memorization rejected"),
            "source_table": {
                "label": "Sources",
                "list_title": "title",
                "list_subtitle": "id",
                "list_excerpt": "meta.wikidata.description",
                "badges": [
                    {"path": "meta.language", "label": "Language", "tone": "blue"},
                    {"path": "meta.dataset", "label": "Dataset", "tone": "slate"},
                ],
                "filters": ["meta.language", "meta.dataset"],
                "detail_sections": [
                    _section("id", "Id", "plain"),
                    _section("title", "Title", "plain"),
                    _section("text", "Text", "plain"),
                    _section("meta", "Meta", "code"),
                    _section("url", "Url", "plain"),
                ],
                "search_fields": ["id", "title", "text", "meta.dataset"],
                "preview_limit": 40,
                "page_size": 20,
            },
            "memory_chunks": {
                "label": "Memory chunks",
                "list_title": "title",
                "list_subtitle": "id",
                "list_excerpt": "text",
                "badges": [
                    {"path": "source_id", "label": "Source", "tone": "slate"},
                    {"path": "meta.word_count", "label": "Words", "tone": "amber"},
                ],
                "filters": ["source_id"],
                "detail_sections": [
                    _section("id", "Id", "plain"),
                    _section("title", "Title", "plain"),
                    _section("text", "Text", "plain"),
                    _section("meta", "Meta", "code"),
                    _section("source_id", "Source Id", "plain"),
                    _section("url", "Url", "plain"),
                ],
                "search_fields": ["id", "title", "text", "source_id"],
                "preview_limit": 20,
                "page_size": 20,
            },
        },
    }


def _grounded_row_view(label: str) -> dict[str, Any]:
    return {
        "label": label,
        "list_title": "prompt",
        "list_subtitle": "id",
        "list_excerpt": "target",
        "badges": [
            {"path": "meta.question_type", "label": "Type", "tone": "blue"},
            {"path": "meta.query_angle", "label": "Angle", "tone": "slate"},
            {"path": "meta.required_cited_sources", "label": "Need", "tone": "amber"},
            {"path": "hidden.generation_filter.reasons", "label": "Reject", "tone": "rose"},
        ],
        "filters": [
            "meta.question_type",
            "meta.query_angle",
            "meta.required_cited_sources",
            "hidden.generation_filter.reasons",
        ],
        "detail_sections": [
            _section("prompt", "Prompt", "plain"),
            _section("messages", "Messages", "code"),
            _section("reasoning", "Reasoning", "markdown"),
            _section("target", "Target", "code"),
            _section("sources", "Sources", "code"),
            _section("scores", "Scores", "code"),
            _section("meta", "Meta", "code"),
            _section("hidden.generation_filter", "Generation Filter", "code"),
        ],
        "search_fields": [
            "id",
            "prompt",
            "reasoning",
            "target",
            "meta.question_type",
            "meta.query_angle",
            "hidden.generation_filter.reasons",
        ],
        "preview_limit": 30,
    }


def _memorization_view(label: str) -> dict[str, Any]:
    return {
        "label": label,
        "list_title": "prompt",
        "list_subtitle": "id",
        "list_excerpt": "target",
        "badges": [
            {"path": "meta.question_type", "label": "Type", "tone": "blue"},
            {"path": "meta.query_angle", "label": "Angle", "tone": "slate"},
            {"path": "hidden.generation_filter.reasons", "label": "Reject", "tone": "rose"},
        ],
        "filters": [
            "meta.question_type",
            "meta.query_angle",
            "hidden.generation_filter.reasons",
        ],
        "detail_sections": [
            _section("prompt", "Prompt", "plain"),
            _section("messages", "Messages", "code"),
            _section("reasoning", "Reasoning", "markdown"),
            _section("target", "Target", "plain"),
            _section("sources", "Sources", "code"),
            _section("scores", "Scores", "code"),
            _section("meta", "Meta", "code"),
            _section("hidden.generation_filter", "Generation Filter", "code"),
        ],
        "search_fields": [
            "id",
            "prompt",
            "reasoning",
            "target",
            "meta.question_type",
            "meta.query_angle",
            "hidden.generation_filter.reasons",
        ],
        "preview_limit": 30,
    }


def _section(path: str, label: str, format_name: str) -> dict[str, str]:
    return {
        "path": path,
        "label": label,
        "format": format_name,
    }
