from __future__ import annotations

from sdg.packs.pleias_synth.build_memory_core import build_index
from sdg.packs.pleias_synth.memorization_text import is_person, lexical_tokens, meaningful_tokens, tokenize


def test_tokenize_keeps_danish_letters() -> None:
    assert tokenize("Ærø har blåbær og ål.") == ["ærø", "har", "blåbær", "og", "ål"]


def test_meaningful_tokens_use_danish_stopwords() -> None:
    assert meaningful_tokens("Det og den blå fisk", language="da") == {"blå", "fisk"}


def test_build_index_uses_unicode_tokenizer() -> None:
    index = build_index(
        [
            {
                "id": "chunk-1",
                "title": "Ærø",
                "text": "Ærø har blåbær og ål.",
                "url": "https://example.com",
            }
        ]
    )

    assert index["type"] == "lexical-v2"
    assert index["chunks"]["chunk-1"]["tokens"] == lexical_tokens("Ærø har blåbær og ål.", limit=128)


def test_is_person_supports_danish_metadata() -> None:
    doc = {
        "meta": {
            "language": "da",
            "wikidata": {"description": "dansk forfatter og politiker"},
            "structured_wikipedia": {
                "categories": ["Danske forfattere", "Danske politikere"],
                "infobox_templates": [],
            },
        }
    }

    assert is_person(doc)
