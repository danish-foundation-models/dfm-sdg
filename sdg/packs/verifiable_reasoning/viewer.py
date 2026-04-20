from __future__ import annotations

from typing import Any


def viewer_spec() -> dict[str, Any]:
    row_view = _row_view("Accepted rows", excerpt_path="target")
    rejection_view = _row_view("Rejected rows", excerpt_path="hidden.generation_error")

    return {
        "title": "Verifiable Reasoning Viewer",
        "default_artifact": "dataset",
        "default_page_size": 20,
        "artifacts": {
            "dataset": row_view,
            "verified": row_view,
            "rejections": rejection_view,
            "failures": rejection_view,
            "sample_preview": {
                **row_view,
                "label": "Accepted preview",
                "preview_limit": 20,
            },
            "rejections_preview": {
                **rejection_view,
                "label": "Rejected preview",
                "preview_limit": 20,
            },
        },
    }


def _row_view(label: str, *, excerpt_path: str) -> dict[str, Any]:
    return {
        "label": label,
        "list_title": "id",
        "list_subtitle": "meta.family",
        "list_excerpt": excerpt_path,
        "badges": [
            {"path": "meta.prompt_language", "label": "Lang", "tone": "blue"},
            {"path": "meta.difficulty", "label": "Difficulty", "tone": "amber"},
            {"path": "meta.target_source", "label": "Source", "tone": "slate"},
        ],
        "filters": [
            "meta.family",
            "meta.prompt_language",
            "meta.difficulty",
            "meta.target_source",
        ],
        "detail_sections": [
            _section("target", "Response", "code"),
            _section("reasoning", "Reasoning", "plain"),
            _section("hidden.generation_error", "Generation Error", "plain"),
            _section("hidden.answer_teacher_raw_response", "Raw Response", "plain"),
            _section("prompt", "Prompt", "plain"),
            _section("checks", "Checks", "code"),
            _section("sources", "Sources", "code"),
            _section("meta", "Meta", "code"),
            _section("hidden", "Hidden", "code"),
        ],
        "search_fields": [
            "id",
            "prompt",
            "target",
            "reasoning",
            "meta.family",
            "meta.prompt_language",
            "meta.difficulty",
            "meta.target_source",
            "hidden.generation_error",
        ],
        "preview_limit": 40,
    }


def _section(path: str, label: str, format_name: str) -> dict[str, str]:
    return {
        "path": path,
        "label": label,
        "format": format_name,
    }
