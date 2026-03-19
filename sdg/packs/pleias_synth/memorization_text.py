from __future__ import annotations

import re
from typing import Any


def split_sentences(text: str) -> list[str]:
    collapsed = re.sub(r"\s+", " ", text.strip())
    if not collapsed:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", collapsed) if part.strip()]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def normalize_text(text: str) -> str:
    return " ".join(tokenize(text))


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    return [text]


def clean_recall_list(value: Any) -> list[str]:
    cleaned_items: list[str] = []
    for item in as_list(value):
        cleaned = clean_recall_text(item)
        if cleaned:
            cleaned_items.append(cleaned)
    return cleaned_items


def clean_recall_text(text: str) -> str:
    cleaned = str(text).strip()
    if not cleaned:
        return ""

    replacements = [
        (r"\baccording to the provided texts\b", "from what is known"),
        (r"\baccording to the provided text\b", "from what is known"),
        (r"\baccording to the retrieved support\b", "from what is known"),
        (r"\baccording to the support\b", "from what is known"),
        (r"\baccording to the evidence\b", "from what is known"),
        (r"\bthe provided texts?\b", "what is known"),
        (r"\bthe hidden facts\b", "what is known"),
        (r"\bthe hidden grounding facts\b", "what is known"),
        (r"\bthe hidden grounding details\b", "the known details"),
        (r"\bthe teacher bundle\b", "the remembered facts"),
        (r"\bthe source claim\b", "the core claim"),
        (r"\bthe source\b", "the topic"),
        (r"\bretrieved support\b", "known details"),
        (r"\bprovided support\b", "known details"),
        (r"\bthe evidence\b", "the known facts"),
        (r"\bevidence\b", "known facts"),
        (r"\bthis is supported by\b", "this is consistent with"),
        (r"\bthe text states that\b", ""),
        (r"\bthe text says that\b", ""),
        (r"\bthe passage states that\b", ""),
        (r"\bthe passage says that\b", ""),
    ]

    for pattern, replacement in replacements:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s+([.,;:!?])", r"\1", cleaned)
    return cleaned.strip(" -")


def meaningful_tokens(text: str) -> set[str]:
    return {
        token
        for token in normalize_text(text).split()
        if len(token) > 2 and token not in {"and", "the"}
    }


def title_variants(title: str) -> list[str]:
    variants = [title]
    if not title.lower().startswith("the "):
        variants.append(f"The {title}")
    return sorted(variants, key=len, reverse=True)


def is_person(doc: dict[str, Any]) -> bool:
    meta = doc.get("meta") or {}
    structured = meta.get("structured_wikipedia") or {}
    categories = [category.lower() for category in structured.get("categories", [])]
    description = str((meta.get("wikidata") or {}).get("description") or meta.get("wikibase_shortdesc") or "").lower()
    infoboxes = [name.lower() for name in structured.get("infobox_templates", [])]

    person_markers = [
        "births",
        "deaths",
        "people",
        "actors",
        "actresses",
        "writers",
        "scientists",
        "philosophers",
        "politicians",
        "artists",
    ]
    if any(
        marker in description
        for marker in [
            "physicist",
            "chemist",
            "writer",
            "politician",
            "artist",
            "actor",
            "philosopher",
            "scientist",
        ]
    ):
        return True
    if any(marker in category for category in categories for marker in person_markers):
        return True
    if any("infobox person" in name or "infobox scientist" in name for name in infoboxes):
        return True
    return False
