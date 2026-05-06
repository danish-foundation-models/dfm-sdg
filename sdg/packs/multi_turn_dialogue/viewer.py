from __future__ import annotations

from typing import Any


def viewer_spec() -> dict[str, Any]:
    row_view = _row_view("Selected dialogues")
    candidate_view = _row_view("Candidate dialogues")
    rejected_view = _row_view("Rejected dialogues")

    return {
        "title": "Multi-Turn Dialogue Viewer",
        "default_artifact": "dataset",
        "default_page_size": 20,
        "artifacts": {
            "dataset": row_view,
            "verified": row_view,
            "sample_preview": {
                **row_view,
                "label": "Selected preview",
                "preview_limit": 20,
            },
            "candidates": candidate_view,
            "rejected_candidates": rejected_view,
            "failures": rejected_view,
        },
    }


def _row_view(label: str) -> dict[str, Any]:
    return {
        "label": label,
        "list_title": "id",
        "list_subtitle": "meta.domain",
        "list_excerpt": "messages.0.content",
        "badges": [
            {"path": "meta.family", "label": "Family", "tone": "blue"},
            {"path": "meta.domain", "label": "Domain", "tone": "slate"},
            {"path": "meta.intent_trajectory", "label": "Intent", "tone": "amber"},
            {"path": "hidden.review.review_decision", "label": "Review", "tone": "rose"},
        ],
        "filters": [
            "meta.family",
            "meta.domain",
            "meta.intent_trajectory",
            "hidden.review.review_decision",
        ],
        "detail_sections": [
            _section("messages", "Conversation", "messages"),
            _section("hidden.review", "Review", "code"),
            _section("hidden.selection", "Selection", "code"),
            _section("sources", "Sources", "code"),
            _section("meta", "Meta", "code"),
            _section("hidden", "Hidden", "code"),
        ],
        "search_fields": [
            "id",
            "messages",
            "meta.family",
            "meta.domain",
            "meta.intent_trajectory",
            "hidden.review.review_decision",
        ],
        "preview_limit": 40,
    }


def _section(path: str, label: str, format_name: str) -> dict[str, str]:
    return {
        "path": path,
        "label": label,
        "format": format_name,
    }
